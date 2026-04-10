# ─────────────────────────────────────────────────────────────────────────────
# paper_url_parser.py
# Extract paper metadata (title, abstract, arxiv_id) from URLs.
# Supports: arXiv, Semantic Scholar, and generic HTTP URLs.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import requests
from urllib.parse import urlparse
import arxiv

import config


def _get_arxiv_id_from_url(url: str) -> str | None:
    """Extract arXiv ID from an arXiv URL."""
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_semantic_scholar_id_from_url(url: str) -> str | None:
    """Extract Semantic Scholar paper ID from a URL."""
    match = re.search(r"semanticscholar\.org/paper/[a-z0-9\-]+/([a-f0-9]{40})", url)
    if match:
        return match.group(1)
    return None


def _fetch_arxiv_metadata(arxiv_id: str) -> tuple[str, str] | None:
    """
    Fetch title and abstract from arXiv using the arXiv ID.
    
    Returns
    -------
    (title, abstract) or None on failure.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        result = next(client.results(search), None)
        if result:
            return result.title, result.summary
    except Exception as e:
        print(f"    [arXiv API] Could not fetch metadata for {arxiv_id}: {e}")
    return None


def _fetch_semantic_scholar_metadata(paper_id: str) -> tuple[str, str] | None:
    """
    Fetch title and abstract from Semantic Scholar using paper ID.
    
    Returns
    -------
    (title, abstract) or None on failure.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        if config.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
        
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {"fields": "title,abstract"}
        
        resp = requests.get(url, params=params, timeout=20, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        title = data.get("title", "").strip()
        abstract = data.get("abstract", "").strip()
        
        if title and abstract:
            return title, abstract
    except Exception as e:
        print(f"    [Semantic Scholar API] Could not fetch metadata for {paper_id}: {e}")
    return None


def _fetch_html_abstract_from_url(url: str) -> tuple[str, str] | None:
    """
    Try to extract title and abstract from the HTML of a generic paper URL.
    Looks for meta tags (og:title, description) and common abstract patterns.
    
    Returns
    -------
    (title, abstract) or None on failure.
    """
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
        
        # Try meta tags
        title_match = re.search(r'<meta\s+(?:name|property)="(?:og:)?title"\s+content="([^"]+)"', html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        
        abstract_match = re.search(r'<meta\s+(?:name|property)="(?:og:)?description"\s+content="([^"]+)"', html, re.IGNORECASE)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        if not abstract:
            abstract_match = re.search(r'<meta\s+(?:name|property)="abstract"\s+content="([^"]+)"', html, re.IGNORECASE)
            abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        if title and abstract:
            return title, abstract
    except Exception as e:
        print(f"    [HTML Scrape] Could not fetch metadata from {url}: {e}")
    return None


def extract_paper_metadata(paper_url: str) -> dict:
    """
    Extract title and abstract from a paper URL.
    Supports: arXiv, Semantic Scholar, generic HTTP URLs.
    
    Parameters
    ----------
    paper_url : str
        The URL of the paper (arXiv, Semantic Scholar, or generic).
    
    Returns
    -------
    dict with keys: title, abstract, arxiv_id, paper_url
        Returns all fields; title/abstract may be empty if extraction fails.
    
    Raises
    ------
    ValueError
        If the URL is invalid or no metadata could be extracted.
    """
    paper_url = paper_url.strip()
    
    print(f"[Paper URL] Extracting metadata from: {paper_url}")
    
    # Try arXiv
    arxiv_id = _get_arxiv_id_from_url(paper_url)
    if arxiv_id:
        print(f"  Detected arXiv ID: {arxiv_id}")
        metadata = _fetch_arxiv_metadata(arxiv_id)
        if metadata:
            title, abstract = metadata
            print(f"  ✓ Title: {title[:70]}...")
            print(f"  ✓ Abstract: {abstract[:100]}...")
            return {
                "title": title,
                "abstract": abstract,
                "arxiv_id": arxiv_id,
                "paper_url": paper_url,
            }
    
    # Try Semantic Scholar
    s2_id = _get_semantic_scholar_id_from_url(paper_url)
    if s2_id:
        print(f"  Detected Semantic Scholar ID: {s2_id}")
        metadata = _fetch_semantic_scholar_metadata(s2_id)
        if metadata:
            title, abstract = metadata
            print(f"  ✓ Title: {title[:70]}...")
            print(f"  ✓ Abstract: {abstract[:100]}...")
            return {
                "title": title,
                "abstract": abstract,
                "arxiv_id": None,
                "paper_url": paper_url,
            }
    
    # Try HTML scrape (generic URL)
    print(f"  Attempting HTML metadata extraction...")
    metadata = _fetch_html_abstract_from_url(paper_url)
    if metadata:
        title, abstract = metadata
        print(f"  ✓ Title: {title[:70]}...")
        print(f"  ✓ Abstract: {abstract[:100]}...")
        return {
            "title": title,
            "abstract": abstract,
            "arxiv_id": arxiv_id,
            "paper_url": paper_url,
        }
    
    raise ValueError(f"Could not extract metadata from {paper_url}")
