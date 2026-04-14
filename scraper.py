# ─────────────────────────────────────────────────────────────────────────────
# scraper.py
# Stage 0: Fetch papers for a specific Semantic Scholar author.
# Enriches each paper with:
#   - Abstract        (Semantic Scholar API)
#   - Full text       (arXiv PDF → PyMuPDF, or Semantic Scholar open-access PDF)
#
# Full-text fallback chain per paper:
#   1. arXiv API (title search → PDF download → text extraction)
#   2. Semantic Scholar openAccessPdf URL → PDF download → text extraction
#   3. Abstract only (if no open-access PDF found)
#
# Dependencies: requests, arxiv, pymupdf, pandas, tqdm, config.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import html
import re
import time
import tempfile
from requests import Response
import requests
import arxiv
import fitz                          # PyMuPDF
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
from scholarly import scholarly

import config


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a PDF given as raw bytes.
    Uses PyMuPDF (fitz). Returns empty string on failure.
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:
        print(f"    [PyMuPDF] Could not extract text: {e}")
        return ""


def _download_pdf_bytes(url: str) -> bytes | None:
    """
    Download a PDF from a URL and return its raw bytes.
    Returns None on failure.
    """
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        if "pdf" in resp.headers.get("Content-Type", "").lower() or url.endswith(".pdf"):
            return resp.content
    except Exception as e:
        print(f"    [PDF Download] Failed for {url[:80]}: {e}")
    return None


def _semantic_scholar_headers() -> dict:
    """Build request headers for Semantic Scholar, including an API key if present."""
    headers = {"User-Agent": "Mozilla/5.0"}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    return headers


def _semantic_scholar_get(url: str, params: dict, timeout: int = 20, max_retries: int = 3) -> Response:
    """GET Semantic Scholar with simple retry/backoff for 429 and transient 5xx errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=_semantic_scholar_headers())
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
                print(f"    [Semantic Scholar] Rate limited (429); retrying in {wait_seconds:.0f}s...")
                time.sleep(wait_seconds)
                last_exc = requests.HTTPError(f"429 Client Error for url: {resp.url}")
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait_seconds = 2 ** attempt
                print(f"    [Semantic Scholar] Request failed; retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)
            else:
                break
    raise RuntimeError("Semantic Scholar request failed after retries") from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Full-text source 1: arXiv
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fulltext_from_pdf_url(pdf_url: str) -> tuple[str, str]:
    """Download and extract text from a known PDF URL."""
    if not pdf_url:
        return "", ""
    pdf_bytes = _download_pdf_bytes(pdf_url)
    if not pdf_bytes:
        return "", pdf_url
    text = _extract_text_from_pdf_bytes(pdf_bytes)
    return text, pdf_url


def _fetch_fulltext_arxiv_by_id(arxiv_id: str) -> tuple[str, str]:
    """Fetch full text from arXiv using an explicit arXiv identifier."""
    if not arxiv_id:
        return "", ""
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        result = next(client.results(search), None)
        if result is None:
            return "", ""

        pdf_url = result.pdf_url
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        result.download_pdf(filename=tmp_path)
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(tmp_path)

        text = _extract_text_from_pdf_bytes(pdf_bytes)
        return text, pdf_url
    except Exception as e:
        print(f"    [arXiv] Could not fetch id '{arxiv_id}': {e}")
        return "", ""

def _fetch_fulltext_arxiv(title: str) -> tuple[str, str]:
    """
    Search arXiv by title, download the PDF, and extract full text.

    Returns
    -------
    (full_text, pdf_url) — both empty strings if not found.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        result = next(client.results(search), None)
        if result is None:
            return "", ""

        pdf_url = result.pdf_url
        # Download to a temp file (arxiv library requires a file path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        result.download_pdf(filename=tmp_path)
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        os.remove(tmp_path)

        text = _extract_text_from_pdf_bytes(pdf_bytes)
        return text, pdf_url

    except Exception as e:
        print(f"    [arXiv] Could not fetch '{title[:60]}': {e}")
        return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Full-text source 2: Semantic Scholar open-access PDF
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fulltext_semantic_scholar(title: str) -> tuple[str, str]:
    """
    Look up a paper on Semantic Scholar, find its open-access PDF URL,
    download the PDF, and extract full text.

    Returns
    -------
    (full_text, pdf_url) — both empty strings if not found.
    """
    try:
        # Step 1: search by title to get paperId
        search_url = config.SEMANTIC_SCHOLAR_API
        params = {
            "query": title,
            "fields": "title,abstract,openAccessPdf",
            "limit": 1,
        }
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("total", 0) == 0:
            return "", ""

        paper = data["data"][0]
        oa_pdf = paper.get("openAccessPdf")
        if not oa_pdf or not oa_pdf.get("url"):
            return "", ""

        pdf_url = oa_pdf["url"]
        pdf_bytes = _download_pdf_bytes(pdf_url)
        if not pdf_bytes:
            return "", pdf_url

        text = _extract_text_from_pdf_bytes(pdf_bytes)
        return text, pdf_url

    except Exception as e:
        print(f"    [Semantic Scholar PDF] Could not fetch '{title[:60]}': {e}")
        return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Abstract-only fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_abstract_semantic_scholar(title: str) -> str:
    """
    Fetch only the abstract from Semantic Scholar.
    Used when no full text is available.
    """
    params = {"query": title, "fields": "title,abstract", "limit": 1}
    try:
        resp = _semantic_scholar_get(config.SEMANTIC_SCHOLAR_API, params=params, timeout=10, max_retries=3)
        data = resp.json()
        if data.get("total", 0) > 0:
            return (data["data"][0].get("abstract") or "").strip()
    except Exception as e:
        print(f"    [Semantic Scholar Abstract] Warning for '{title[:60]}': {e}")
    return ""


def _resolve_semantic_scholar_author(person_identifier: str) -> dict:
    """Resolve a Semantic Scholar author from a name, author URL, or author ID."""
    raw = str(person_identifier).strip()
    if not raw:
        raise ValueError("person_identifier cannot be empty")

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc and "/author/" in parsed.path:
        raw = parsed.path.rstrip("/").split("/")[-1]

    if raw.isdigit():
        author_url = f"https://api.semanticscholar.org/graph/v1/author/{raw}"
        params = {
            "fields": "name,affiliations,paperCount,papers.title,papers.year,papers.venue,papers.authors,papers.abstract,papers.citationCount,papers.openAccessPdf,papers.externalIds",
        }
        resp = _semantic_scholar_get(author_url, params=params, timeout=20, max_retries=3)
        author = resp.json()
        author.setdefault("authorId", raw)
        return author

    search_url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": raw,
        "fields": "name,affiliations,paperCount,url",
        "limit": 10,
    }
    resp = _semantic_scholar_get(search_url, params=params, timeout=20, max_retries=3)
    results = resp.json().get("data", [])
    if not results:
        raise RuntimeError(f"No Semantic Scholar author found for '{person_identifier}'")

    normalized = raw.casefold()

    def sort_key(candidate: dict) -> tuple[int, int]:
        name = str(candidate.get("name", "")).casefold()
        exact = 1 if name == normalized else 0
        paper_count = int(candidate.get("paperCount", 0) or 0)
        return exact, paper_count

    chosen = sorted(results, key=sort_key, reverse=True)[0]
    author_id = chosen.get("authorId")
    if not author_id:
        raise RuntimeError(f"Semantic Scholar author search returned no authorId for '{person_identifier}'")

    author_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    params = {
        "fields": "name,affiliations,paperCount,papers.title,papers.year,papers.venue,papers.authors,papers.abstract,papers.citationCount,papers.openAccessPdf,papers.externalIds",
    }
    resp = _semantic_scholar_get(author_url, params=params, timeout=20, max_retries=3)
    author = resp.json()
    author.setdefault("authorId", author_id)
    return author


def _choose_paper_limit(available_count: int, preferred_count: int) -> int:
    """Prefer 20 papers, but if unavailable use 10 when possible."""
    if available_count >= preferred_count:
        return preferred_count
    if available_count >= 10:
        return 10
    return available_count


def _fetch_papers_from_semantic_scholar(query: str, n_papers: int) -> pd.DataFrame:
    """
    Fallback source for paper lists when Google Scholar lookup is unavailable.

    Uses Semantic Scholar paper search to retrieve the top `n_papers` matches for
    the configured query, then enriches each result with the same PDF/abstract
    fallback chain used for Google Scholar results.
    """
    print("\n" + "=" * 60)
    print("STAGE 0 — Fetching papers from Semantic Scholar fallback")
    print("=" * 60)

    params = {
        "query": query,
        "fields": "title,year,venue,authors,abstract,citationCount,openAccessPdf,externalIds",
        "limit": n_papers,
    }

    try:
        resp = _semantic_scholar_get(config.SEMANTIC_SCHOLAR_API, params=params, timeout=20, max_retries=3)
    except Exception as exc:
        print(f"[Stage 0] Semantic Scholar search failed: {exc}")
        print("[Stage 0] Falling back to arXiv query search.")
        return _fetch_papers_from_arxiv(query, n_papers)

    data = resp.json()
    papers = data.get("data", [])[:n_papers]

    records = []
    for i, paper in enumerate(tqdm(papers, desc="Fetching papers")):
        title = (paper.get("title") or "").strip()
        abstract_raw = (paper.get("abstract") or "").strip()
        oa_pdf = paper.get("openAccessPdf") or {}
        preferred_pdf_url = oa_pdf.get("url") or ""
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("ARXIV") or ""
        authors = paper.get("authors") or []
        author_names = ", ".join(a.get("name", "") for a in authors if a.get("name"))
        author_affiliations = []
        for a in authors:
            name = str(a.get("name", "")).strip()
            raw_aff = a.get("affiliations") or a.get("affiliation") or ""
            if isinstance(raw_aff, (list, tuple, set)):
                aff = ", ".join(str(x).strip() for x in raw_aff if str(x).strip())
            else:
                aff = str(raw_aff).strip()
            if name:
                author_affiliations.append(f"{name}: {aff or 'N/A'}")
        base = {
            "rank_original": i + 1,
            "title": title,
            "year": paper.get("year", ""),
            "venue": paper.get("venue") or "",
            "authors": author_names,
            "citations": paper.get("citationCount", 0),
        }

        enriched = _enrich_paper(
            title,
            abstract_raw,
            preferred_pdf_url=preferred_pdf_url,
            arxiv_id=arxiv_id,
        )
        source_label = enriched["text_source"]
        has_full = "✅" if enriched["full_text"] else "❌"
        print(f"  [{i+1:2d}] {has_full} full text ({source_label}) — {title[:60]}")
        print(f"       Authors       : {author_names or 'N/A'}")
        print(f"       Affiliations  : {'; '.join(author_affiliations) if author_affiliations else 'N/A'}")

        records.append({**base, **enriched})

    df = pd.DataFrame(records)
    n_full = (df["full_text"].str.len() > 0).sum() if not df.empty else 0
    print(f"\n✅ Fetched {len(df)} papers | Full text available: {n_full}/{len(df)}")
    return df


def _fetch_papers_from_arxiv(query: str, n_papers: int) -> pd.DataFrame:
    """Fallback source for paper lists when Semantic Scholar is rate limited."""
    print("\n" + "=" * 60)
    print("STAGE 0 — Fetching papers from arXiv fallback")
    print("=" * 60)

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=n_papers,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))[:n_papers]
    except Exception as exc:
        raise RuntimeError("arXiv fallback also failed") from exc

    records = []
    for i, paper in enumerate(tqdm(results, desc="Fetching papers")):
        title = (paper.title or "").strip()
        abstract_raw = (getattr(paper, "summary", "") or "").strip()
        author_names = ", ".join(a.name for a in getattr(paper, "authors", []) if getattr(a, "name", ""))
        base = {
            "rank_original": i + 1,
            "title": title,
            "year": getattr(paper, "published", None).year if getattr(paper, "published", None) else "",
            "venue": "arXiv",
            "authors": author_names,
            "citations": 0,
        }

        enriched = _enrich_paper(title, abstract_raw)
        source_label = enriched["text_source"]
        has_full = "✅" if enriched["full_text"] else "❌"
        print(f"  [{i+1:2d}] {has_full} full text ({source_label}) — {title[:60]}")
        print(f"       Authors       : {author_names or 'N/A'}")
        print("       Affiliations  : N/A (arXiv API result does not include affiliations here)")

        records.append({**base, **enriched})

    df = pd.DataFrame(records)
    n_full = (df["full_text"].str.len() > 0).sum() if not df.empty else 0
    print(f"\n✅ Fetched {len(df)} papers | Full text available: {n_full}/{len(df)}")
    return df


def _fetch_papers_from_semantic_scholar_author(
    person_identifier: str,
    preferred_count: int,
    sort_by: str = "recent",
) -> pd.DataFrame:
    """Fetch papers for one Semantic Scholar author, sorted by recency or citations."""
    author = _resolve_semantic_scholar_author(person_identifier)
    author_name = str(author.get("name", "Unknown")).strip() or "Unknown"
    author_affiliations = _format_affiliations(author)
    papers = author.get("papers") or []

    print("\n" + "=" * 60)
    sort_label = "most recent" if sort_by == "recent" else "most cited"
    print(f"STAGE 0 — Fetching {sort_label} papers for one Semantic Scholar author")
    print("=" * 60)
    print(f"Person          : {author_name}")
    print(f"Affiliations    : {author_affiliations or 'N/A'}")
    print(f"Papers available: {len(papers)}")

    if not papers:
        raise RuntimeError(f"Semantic Scholar author '{author_name}' has no papers in the returned profile")

    target_count = _choose_paper_limit(len(papers), preferred_count)
    if target_count != preferred_count:
        print(f"[Stage 0] Requested {preferred_count} papers, using {target_count} because only {len(papers)} are available.")

    if sort_by == "cited":
        sorted_papers = sorted(
            papers,
            key=lambda p: (
                int(p.get("citationCount", 0) or 0),
                int(p.get("year", 0) or 0),
            ),
            reverse=True,
        )[:target_count]
    else:
        sorted_papers = sorted(
            papers,
            key=lambda p: int(p.get("year", 0) or 0),
            reverse=True,
        )[:target_count]

    records = []
    for i, paper in enumerate(tqdm(sorted_papers, desc="Fetching papers")):
        title = (paper.get("title") or "").strip()
        abstract_raw = (paper.get("abstract") or "").strip()
        oa_pdf = paper.get("openAccessPdf") or {}
        preferred_pdf_url = oa_pdf.get("url") or ""
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("ARXIV") or ""
        paper_authors = paper.get("authors") or []
        paper_author_names = ", ".join(a.get("name", "") for a in paper_authors if a.get("name")) or author_name
        paper_affiliations = []
        for a in paper_authors:
            name = str(a.get("name", "")).strip()
            raw_aff = a.get("affiliations") or a.get("affiliation") or ""
            if isinstance(raw_aff, (list, tuple, set)):
                aff = ", ".join(str(x).strip() for x in raw_aff if str(x).strip())
            else:
                aff = str(raw_aff).strip()
            if name:
                paper_affiliations.append(f"{name}: {aff or 'N/A'}")

        base = {
            "rank_original": i + 1,
            "title": title,
            "year": paper.get("year", ""),
            "venue": paper.get("venue") or "",
            "authors": paper_author_names,
            "citations": paper.get("citationCount", 0),
        }

        enriched = _enrich_paper(
            title,
            abstract_raw,
            preferred_pdf_url=preferred_pdf_url,
            arxiv_id=arxiv_id,
        )
        source_label = enriched["text_source"]
        has_full = "✅" if enriched["full_text"] else "❌"
        print(f"  [{i+1:2d}] {has_full} full text ({source_label}) — {title[:60]}")
        print(f"       Authors       : {paper_author_names or 'N/A'}")
        print(f"       Affiliations  : {'; '.join(paper_affiliations) if paper_affiliations else 'N/A'}")

        records.append({**base, **enriched})

    df = pd.DataFrame(records)
    n_full = (df["full_text"].str.len() > 0).sum() if not df.empty else 0
    print(f"\n✅ Fetched {len(df)} papers | Full text available: {n_full}/{len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-paper enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_paper(
    title: str,
    existing_abstract: str,
    preferred_pdf_url: str = "",
    arxiv_id: str = "",
) -> dict:
    """
        Attempt to fetch full text for a paper using the fallback chain:
            1. Provided open-access PDF URL (paper-specific)
            2. arXiv by explicit arXiv id
            3. arXiv by title search
            4. Semantic Scholar open-access PDF by title
            5. Abstract only

    Parameters
    ----------
    title             : str  — paper title
    existing_abstract : str  — abstract already retrieved (e.g. from scholarly)

    Returns
    -------
    dict with keys:
        abstract    : str  — best available abstract
        full_text   : str  — full paper text (empty if unavailable)
        pdf_url     : str  — URL of the PDF used (empty if unavailable)
        text_source : str  — one of "arxiv", "semantic_scholar_pdf", "abstract_only"
    """
    time.sleep(config.SEMANTIC_SCHOLAR_DELAY)  # rate limiting

    # ── Try paper-specific open-access PDF first ─────────────────────────────
    full_text, pdf_url = _fetch_fulltext_from_pdf_url(preferred_pdf_url)
    if full_text:
        abstract = existing_abstract or full_text[:1000]
        return {
            "abstract":    abstract,
            "full_text":   full_text,
            "pdf_url":     pdf_url,
            "text_source": "semantic_scholar_pdf",
        }

    # ── Try arXiv by explicit ID (more precise than title search) ───────────
    full_text, pdf_url = _fetch_fulltext_arxiv_by_id(arxiv_id)
    if full_text:
        abstract = existing_abstract or full_text[:1000]
        return {
            "abstract":    abstract,
            "full_text":   full_text,
            "pdf_url":     pdf_url,
            "text_source": "arxiv",
        }

    # ── Try arXiv title search ────────────────────────────────────────────────
    full_text, pdf_url = _fetch_fulltext_arxiv(title)
    if full_text:
        abstract = existing_abstract or full_text[:1000]  # use first 1000 chars as abstract fallback
        return {
            "abstract":    abstract,
            "full_text":   full_text,
            "pdf_url":     pdf_url,
            "text_source": "arxiv",
        }

    # ── Try Semantic Scholar open-access PDF ──────────────────────────────────
    full_text, pdf_url = _fetch_fulltext_semantic_scholar(title)
    if full_text:
        abstract = existing_abstract or full_text[:1000]
        return {
            "abstract":    abstract,
            "full_text":   full_text,
            "pdf_url":     pdf_url,
            "text_source": "semantic_scholar_pdf",
        }

    # ── Fall back to abstract only ────────────────────────────────────────────
    abstract = existing_abstract or _fetch_abstract_semantic_scholar(title)
    return {
        "abstract":    abstract,
        "full_text":   "",
        "pdf_url":     pdf_url or "",
        "text_source": "abstract_only",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Scholar author parsing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_author_id(url: str) -> str:
    """Extract the author ID from an author profile URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "user" not in params:
        raise ValueError(
            f"Could not find 'user' parameter in URL: {url}\n"
            "Expected format: https://scholar.google.com/citations?user=XXXX"
        )
    return params["user"][0]


def _parse_publication(pub: dict, index: int) -> dict:
    """Extract base metadata fields from a scholarly publication object."""
    bib = pub.get("bib", {})
    return {
        "rank_original": index + 1,
        "title":         bib.get("title", "").strip(),
        "year":          bib.get("pub_year", ""),
        "venue":         (
            bib.get("venue")
            or bib.get("journal")
            or bib.get("booktitle")
            or ""
        ),
        "authors":       bib.get("author", ""),
        "citations":     pub.get("num_citations", 0),
        # abstract from scholarly (often empty — will be enriched below)
        "_abstract_raw": bib.get("abstract", "").strip(),
    }


def _format_affiliations(author: dict) -> str:
    """Return a readable affiliation string from a scholarly author object."""
    affiliations = author.get("affiliation") or author.get("affiliations") or ""
    if isinstance(affiliations, (list, tuple, set)):
        values = [str(item).strip() for item in affiliations if str(item).strip()]
        return ", ".join(values)
    if isinstance(affiliations, dict):
        values = [str(item).strip() for item in affiliations.values() if str(item).strip()]
        return ", ".join(values)
    return str(affiliations).strip()


def _strip_html(value: str) -> str:
    """Remove simple HTML markup from a Scholar snippet."""
    return re.sub(r"<[^>]+>", "", html.unescape(value)).strip()


def _parse_google_scholar_publications(page_html: str) -> list[dict]:
    """Parse publication rows from a Google Scholar profile page."""
    records: list[dict] = []
    rows = re.findall(r'<tr class="gsc_a_tr".*?</tr>', page_html, flags=re.S)
    for row in rows:
        title_match = re.search(r'class="gsc_a_at"[^>]*>(.*?)</a>', row, flags=re.S)
        year_match = re.search(r'class="gsc_a_y"[^>]*>.*?(\d{4})', row, flags=re.S)
        cited_match = re.search(r'class="gsc_a_ac"[^>]*>(.*?)</a>', row, flags=re.S)
        gray_blocks = re.findall(r'class="gs_gray"[^>]*>(.*?)</div>', row, flags=re.S)

        title = _strip_html(title_match.group(1)) if title_match else ""
        if not title:
            continue

        year_text = year_match.group(1) if year_match else ""
        citation_text = _strip_html(cited_match.group(1)) if cited_match else ""
        citation_digits = re.sub(r"[^0-9]", "", citation_text)
        venue = _strip_html(gray_blocks[1]) if len(gray_blocks) > 1 else ""
        venue = re.sub(r"\s*[-–—]\s*\d{4}\s*$", "", venue).strip()
        authors = _strip_html(gray_blocks[0]) if len(gray_blocks) > 0 else ""

        records.append(
            {
                "title": title,
                "year": int(year_text) if year_text.isdigit() else 0,
                "citations": int(citation_digits) if citation_digits.isdigit() else 0,
                "venue": venue,
                "authors": authors,
            }
        )

    return records


def _normalize_text(value: str) -> str:
    """Normalize text for approximate author-name comparisons."""
    return " ".join(str(value).lower().strip().split())


def _resolve_google_scholar_author(person_identifier: str) -> dict:
    """Resolve the best Google Scholar author candidate for a name or profile URL."""
    raw = str(person_identifier).strip()
    if not raw:
        raise ValueError("person_identifier cannot be empty")

    if "scholar.google.com" in raw and "user=" in raw:
        scholar_id = _extract_author_id(raw)
        try:
            author = scholarly.search_author_id(scholar_id)
            if author:
                return author
        except Exception:
            pass

    matches: list[dict] = []
    try:
        search = scholarly.search_author(raw)
        for _ in range(10):
            try:
                candidate = next(search)
            except StopIteration:
                break
            if candidate:
                matches.append(candidate)
    except Exception as exc:
        raise RuntimeError(f"Google Scholar author search failed for '{person_identifier}': {exc}") from exc

    if not matches:
        raise RuntimeError(f"No Google Scholar author found for '{person_identifier}'")

    normalized = _normalize_text(raw)
    chosen = next((candidate for candidate in matches if _normalize_text(candidate.get("name", "")) == normalized), matches[0])
    return chosen


def _extract_scholar_publications(author: dict, n_papers: int) -> list[dict]:
    """Return the author's publications sorted by year, newest first."""
    try:
        filled = scholarly.fill(
            author,
            sections=["publications"],
            sortby="pubyear",
            publication_limit=max(n_papers, 100),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Google Scholar publications: {exc}") from exc

    publications = filled.get("publications", []) or []
    parsed = [_parse_publication(pub, index) for index, pub in enumerate(publications)]
    parsed = [record for record in parsed if record.get("title")]
    parsed.sort(
        key=lambda record: (
            int(record.get("year", 0) or 0),
            int(record.get("citations", 0) or 0),
        ),
        reverse=True,
    )
    return parsed[:n_papers]


def _fetch_publications_from_profile_url(profile_url: str, n_papers: int) -> list[dict]:
    """Fetch publication metadata directly from a Google Scholar profile URL."""
    profile_url = (profile_url or "").strip()
    if not profile_url:
        raise RuntimeError("SCHOLAR_PROFILE_URL is not configured")

    base_url = profile_url.split("#", 1)[0]
    if "view_op=list_works" not in base_url:
        separator = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{separator}view_op=list_works&sortby=pubdate&hl=en"

    params = {"cstart": 0, "pagesize": max(n_papers, 100)}
    records: list[dict] = []
    seen_titles: set[str] = set()

    while len(records) < n_papers:
        resp = requests.get(
            base_url,
            params=params,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()

        page_records = _parse_google_scholar_publications(resp.text)
        if not page_records:
            break

        for row in page_records:
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            key = title.casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)

            records.append(
                {
                    "title": title,
                    "year": row.get("year", 0),
                    "venue": row.get("venue", ""),
                    "authors": row.get("authors", ""),
                    "citations": row.get("citations", 0),
                    "_abstract_raw": "",
                }
            )
            if len(records) >= n_papers:
                break

        if len(page_records) < params["pagesize"]:
            break
        params["cstart"] += params["pagesize"]

    records.sort(
        key=lambda record: (
            int(record.get("year", 0) or 0),
            int(record.get("citations", 0) or 0),
        ),
        reverse=True,
    )
    return records[:n_papers]


def _fetch_publications_from_google_scholar_query(person_identifier: str, n_papers: int) -> list[dict]:
    """Fetch Google Scholar publications using pub search and filter by author name."""
    raw_name = str(person_identifier).strip()
    if not raw_name:
        raise RuntimeError("person_identifier is empty")

    # Prefer an explicit author query to keep results scoped to the person.
    query = f'author:"{raw_name}"'
    candidates: list[dict] = []

    if hasattr(scholarly, "search_pubs"):
        iterator = scholarly.search_pubs(query)
    elif hasattr(scholarly, "search_pubs_query"):
        iterator = scholarly.search_pubs_query(query)
    else:
        raise RuntimeError("scholarly does not expose publication search methods")

    normalized_name = _normalize_text(raw_name)
    max_scan = max(120, n_papers * 8)
    scanned = 0

    while scanned < max_scan and len(candidates) < n_papers * 3:
        try:
            pub = next(iterator)
        except StopIteration:
            break

        scanned += 1
        bib = pub.get("bib", {})
        title = str(bib.get("title", "")).strip()
        if not title:
            continue

        authors = str(bib.get("author", "")).strip()
        # Keep only records where the target name appears among authors.
        if normalized_name not in _normalize_text(authors):
            continue

        year_raw = str(bib.get("pub_year", "")).strip()
        year = int(year_raw) if year_raw.isdigit() else 0
        venue = str(
            bib.get("venue")
            or bib.get("journal")
            or bib.get("booktitle")
            or ""
        ).strip()

        candidates.append(
            {
                "title": title,
                "year": year,
                "venue": venue,
                "authors": authors,
                "citations": int(pub.get("num_citations", 0) or 0),
                "_abstract_raw": str(bib.get("abstract", "")).strip(),
            }
        )

    deduped: list[dict] = []
    seen: set[str] = set()
    for row in candidates:
        key = row["title"].casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    deduped.sort(
        key=lambda record: (
            int(record.get("year", 0) or 0),
            int(record.get("citations", 0) or 0),
        ),
        reverse=True,
    )
    return deduped[:n_papers]


def _fetch_papers_from_google_scholar(person_identifier: str, n_papers: int) -> pd.DataFrame:
    """Fetch the most recent papers from Google Scholar, sorted by publication year."""
    author_name = str(person_identifier).strip() or person_identifier
    author_affiliations = "N/A"
    publications: list[dict] = []

    try:
        author = _resolve_google_scholar_author(person_identifier)
        author_name = str(author.get("name", person_identifier)).strip() or person_identifier
        author_affiliations = _format_affiliations(author)
        publications = _extract_scholar_publications(author, n_papers)
    except Exception as exc:
        print(f"[Stage 0] Google Scholar author lookup failed: {exc}")
        print("[Stage 0] Trying configured SCHOLAR_PROFILE_URL as fallback.")
        try:
            publications = _fetch_publications_from_profile_url(config.SCHOLAR_PROFILE_URL, n_papers)
        except Exception as profile_exc:
            print(f"[Stage 0] Profile URL fallback failed: {profile_exc}")
            print("[Stage 0] Trying Google Scholar publication query fallback.")
            publications = _fetch_publications_from_google_scholar_query(person_identifier, n_papers)

    print("\n" + "=" * 60)
    print("STAGE 0 — Fetching most recent papers from Google Scholar")
    print("=" * 60)
    print(f"Target person   : {author_name}")
    print(f"Affiliations    : {author_affiliations or 'N/A'}")
    print(f"Requesting      : {n_papers} papers")

    records: list[dict] = []
    for index, row in enumerate(publications, start=1):
        title = row["title"]
        abstract_raw = row.get("_abstract_raw", "")
        base = {
            "rank_original": index,
            "title": title,
            "year": row.get("year", ""),
            "venue": row.get("venue", ""),
            "authors": row.get("authors", ""),
            "citations": row.get("citations", 0),
        }

        enriched = _enrich_paper(title, abstract_raw)
        source_label = enriched["text_source"]
        has_full = "✅" if enriched["full_text"] else "❌"
        print(f"  [{index:2d}] {has_full} full text ({source_label}) — {title[:60]}")
        print(f"       Authors       : {base['authors'] or 'N/A'}")
        print(f"       Venue         : {base['venue'] or 'N/A'}")

        records.append({**base, **enriched})

    if not records:
        raise RuntimeError("Google Scholar profile did not return any publications")

    df = pd.DataFrame(records)
    n_full = (df["full_text"].str.len() > 0).sum() if not df.empty else 0
    print(f"\n✅ Fetched {len(df)} papers | Full text available: {n_full}/{len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_papers(person_identifier: str, n_papers: int) -> pd.DataFrame:
    """
    Fetch the most recent papers for one person using Google Scholar.
    Enriches each paper with abstract + full text via open-access PDF / arXiv.

    Parameters
    ----------
    person_identifier : str
        Semantic Scholar author name, author URL, or author ID.
    n_papers : int
        Number of most-recent papers to retrieve.

    Returns
    -------
    pd.DataFrame with columns:
        rank_original, title, year, venue, authors, citations,
        abstract, full_text, pdf_url, text_source
    """
    try:
        return _fetch_papers_from_google_scholar(person_identifier, n_papers)
    except Exception as exc:
        print(f"[Stage 0] Google Scholar recent pipeline failed completely: {exc}")
        print("[Stage 0] Falling back to Semantic Scholar recency so the pipeline can continue.")
        return _fetch_papers_from_semantic_scholar_author(person_identifier, n_papers, sort_by="recent")


def fetch_most_cited_papers(person_identifier: str, n_papers: int) -> pd.DataFrame:
    """
    Fetch the most cited papers for one person from Semantic Scholar.
    Enriches each paper with abstract + full text via open-access PDF / arXiv.

    Parameters
    ----------
    person_identifier : str
        Semantic Scholar author name, author URL, or author ID.
    n_papers : int
        Number of top-cited papers to retrieve.

    Returns
    -------
    pd.DataFrame with columns:
        rank_original, title, year, venue, authors, citations,
        abstract, full_text, pdf_url, text_source
    """
    return _fetch_papers_from_semantic_scholar_author(person_identifier, n_papers, sort_by="cited")


def fetch_papers_by_title_search(title: str, n_papers: int) -> pd.DataFrame:
    """
    Fetch papers by searching Semantic Scholar for a title (e.g., a reference paper's title).
    Returns related papers enriched with full text via the standard fallback chain.

    Parameters
    ----------
    title : str
        The paper title to search for.
    n_papers : int
        Number of papers to retrieve from search results.

    Returns
    -------
    pd.DataFrame with columns:
        rank_original, title, year, venue, authors, citations,
        abstract, full_text, pdf_url, text_source
    """
    print("\n" + "=" * 60)
    print("STAGE 0 — Searching Semantic Scholar by title")
    print("=" * 60)
    print(f"Search query    : {title[:70]}...")
    print(f"Requesting      : {n_papers} papers")

    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,year,venue,authors,abstract,citationCount,openAccessPdf,externalIds",
        "limit": n_papers,
    }

    try:
        resp = _semantic_scholar_get(search_url, params=params, timeout=20, max_retries=3)
    except Exception as exc:
        print(f"[Stage 0] Semantic Scholar search failed: {exc}")
        print("[Stage 0] Falling back to arXiv title search.")
        return _fetch_papers_from_arxiv(title, n_papers)

    data = resp.json()
    papers = data.get("data", [])[:n_papers]

    if not papers:
        print(f"[Stage 0] No papers found for '{title[:60]}'. Trying arXiv fallback.")
        return _fetch_papers_from_arxiv(title, n_papers)

    records = []
    for i, paper in enumerate(tqdm(papers, desc="Fetching papers")):
        paper_title = (paper.get("title") or "").strip()
        abstract_raw = (paper.get("abstract") or "").strip()
        oa_pdf = paper.get("openAccessPdf") or {}
        preferred_pdf_url = oa_pdf.get("url") or ""
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("ARXIV") or ""
        authors = paper.get("authors") or []
        author_names = ", ".join(a.get("name", "") for a in authors if a.get("name")) or "Unknown"
        author_affiliations = []
        for a in authors:
            name = str(a.get("name", "")).strip()
            raw_aff = a.get("affiliations") or a.get("affiliation") or ""
            if isinstance(raw_aff, (list, tuple, set)):
                aff = ", ".join(str(x).strip() for x in raw_aff if str(x).strip())
            else:
                aff = str(raw_aff).strip()
            if name:
                author_affiliations.append(f"{name}: {aff or 'N/A'}")

        base = {
            "rank_original": i + 1,
            "title": paper_title,
            "year": paper.get("year", ""),
            "venue": paper.get("venue") or "",
            "authors": author_names,
            "citations": paper.get("citationCount", 0),
        }

        enriched = _enrich_paper(
            paper_title,
            abstract_raw,
            preferred_pdf_url=preferred_pdf_url,
            arxiv_id=arxiv_id,
        )
        source_label = enriched["text_source"]
        has_full = "✅" if enriched["full_text"] else "❌"
        print(f"  [{i+1:2d}] {has_full} full text ({source_label}) — {paper_title[:60]}")
        print(f"       Authors       : {author_names or 'N/A'}")
        print(f"       Affiliations  : {'; '.join(author_affiliations) if author_affiliations else 'N/A'}")

        records.append({**base, **enriched})

    df = pd.DataFrame(records)
    n_full = (df["full_text"].str.len() > 0).sum() if not df.empty else 0
    print(f"\n✅ Fetched {len(df)} papers | Full text available: {n_full}/{len(df)}")
    return df


def save_papers(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame of papers to a CSV file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True,
        lineterminator="\n",
    )
    print(f"💾 Saved to '{path}'")


def load_papers(path: str) -> pd.DataFrame:
    """Load a previously saved papers CSV."""
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df)} papers from '{path}'")
    return df
