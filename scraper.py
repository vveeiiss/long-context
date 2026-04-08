# ─────────────────────────────────────────────────────────────────────────────
# scraper.py
# Stage 0: Fetch papers from a Google Scholar profile.
# Enriches each paper with:
#   - Abstract        (Semantic Scholar API)
#   - Full text       (arXiv PDF → PyMuPDF, or Semantic Scholar open-access PDF)
#
# Full-text fallback chain per paper:
#   1. arXiv API (title search → PDF download → text extraction)
#   2. Semantic Scholar openAccessPdf URL → PDF download → text extraction
#   3. Abstract only (if no open-access PDF found)
#
# Dependencies: scholarly, requests, arxiv, pymupdf, pandas, tqdm, config.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import tempfile
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


# ─────────────────────────────────────────────────────────────────────────────
# Full-text source 1: arXiv
# ─────────────────────────────────────────────────────────────────────────────

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
        resp = requests.get(config.SEMANTIC_SCHOLAR_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("total", 0) > 0:
            return (data["data"][0].get("abstract") or "").strip()
    except Exception as e:
        print(f"    [Semantic Scholar Abstract] Warning for '{title[:60]}': {e}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-paper enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_paper(title: str, existing_abstract: str) -> dict:
    """
    Attempt to fetch full text for a paper using the fallback chain:
      1. arXiv
      2. Semantic Scholar open-access PDF
      3. Abstract only

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

    # ── Try arXiv first ───────────────────────────────────────────────────────
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
# Google Scholar profile parsing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_author_id(url: str) -> str:
    """Extract the Google Scholar author ID from a profile URL."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_papers(profile_url: str, n_papers: int) -> pd.DataFrame:
    """
    Scrape the N most-recent papers from a Google Scholar author profile.
    Enriches each paper with abstract + full text via arXiv / Semantic Scholar.

    Parameters
    ----------
    profile_url : str
        Full Google Scholar profile URL.
    n_papers : int
        Number of most-recent papers to retrieve.

    Returns
    -------
    pd.DataFrame with columns:
        rank_original, title, year, venue, authors, citations,
        abstract, full_text, pdf_url, text_source
    """
    print("\n" + "="*60)
    print("STAGE 0 — Fetching papers from Google Scholar")
    print("="*60)

    author_id = _extract_author_id(profile_url)
    print(f"Author ID : {author_id}")

    author = scholarly.search_author_id(author_id)
    author = scholarly.fill(author, sections=["publications"])

    publications = author.get("publications", [])
    print(f"Total publications found : {len(publications)}")

    publications_sorted = sorted(
        publications,
        key=lambda p: int(p.get("bib", {}).get("pub_year", 0) or 0),
        reverse=True,
    )[:n_papers]

    records = []
    for i, pub in enumerate(tqdm(publications_sorted, desc="Fetching papers")):
        base = _parse_publication(pub, i)
        title         = base.pop("title")
        abstract_raw  = base.pop("_abstract_raw")

        enriched = _enrich_paper(title, abstract_raw)

        source_label = enriched["text_source"]
        has_full = "✅" if enriched["full_text"] else "❌"
        print(f"  [{i+1:2d}] {has_full} full text ({source_label}) — {title[:60]}")

        records.append({**base, "title": title, **enriched})

    df = pd.DataFrame(records)

    # Summary
    n_full = (df["full_text"].str.len() > 0).sum()
    print(f"\n✅ Fetched {len(df)} papers | Full text available: {n_full}/{len(df)}")
    return df


def save_papers(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame of papers to a CSV file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved to '{path}'")


def load_papers(path: str) -> pd.DataFrame:
    """Load a previously saved papers CSV."""
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df)} papers from '{path}'")
    return df
