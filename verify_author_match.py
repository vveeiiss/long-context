"""Small utility to verify an author's name against Semantic Scholar and Google Scholar matches.

Usage:
  python verify_author_match.py --name "Yann LeCun"
  python verify_author_match.py --name "Yann LeCun" --semantic-scholar-id 1741107
"""

from __future__ import annotations

import argparse
import difflib
from typing import Iterable
from urllib.parse import urlparse

import requests
from scholarly import scholarly

import config


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _semantic_scholar_headers() -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    return headers


def _semantic_scholar_get(url: str, params: dict, timeout: int = 20) -> dict:
    response = requests.get(url, params=params, timeout=timeout, headers=_semantic_scholar_headers())
    response.raise_for_status()
    return response.json()


def _semantic_scholar_matches(name: str, limit: int = 10) -> list[dict]:
    search_url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": name,
        "fields": "authorId,name,affiliations,paperCount,url",
        "limit": limit,
    }
    data = _semantic_scholar_get(search_url, params=params)
    return data.get("data", [])


def _candidate_queries(name: str) -> list[str]:
    """Generate a small set of fallback queries for a person name."""
    tokens = [token for token in _normalize(name).split() if token]
    if not tokens:
        return []

    candidates = [name.strip()]
    if len(tokens) >= 2:
        candidates.append(tokens[-1])
        candidates.append(" ".join(tokens[:2]))
    if len(tokens) >= 3:
        candidates.append(" ".join(tokens[-2:]))

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        key = _normalize(candidate)
        if key and key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _google_scholar_matches(name: str, limit: int = 10) -> list[dict]:
    matches: list[dict] = []
    try:
        search = scholarly.search_author(name)
        for _ in range(limit):
            try:
                candidate = next(search)
            except StopIteration:
                break
            if not candidate:
                continue
            matches.append(
                {
                    "name": candidate.get("name", ""),
                    "affiliation": candidate.get("affiliation", ""),
                    "scholar_id": candidate.get("scholar_id", ""),
                }
            )
    except Exception as exc:
        print(f"Google Scholar search failed: {exc}")
    return matches


def _best_match(requested_name: str, matches: Iterable[dict]) -> dict | None:
    best: dict | None = None
    best_score = -1.0
    for match in matches:
        candidate_name = str(match.get("name", ""))
        score = _similarity(requested_name, candidate_name)
        if score > best_score:
            best_score = score
            best = {**match, "score": score}
    return best


def resolve_best_semantic_scholar_author(name: str, limit: int = 3) -> dict:
    """Resolve the best Semantic Scholar author match from the top search results."""
    matches = _semantic_scholar_matches(name, limit=limit)
    if not matches:
        raise RuntimeError(f"No Semantic Scholar matches found for '{name}'")

    best = _best_match(name, matches)
    if not best:
        raise RuntimeError(f"Could not score Semantic Scholar matches for '{name}'")

    return best


def _print_matches(source_name: str, requested_name: str, matches: list[dict]) -> None:
    print(f"\n{source_name} matches")
    print("-" * 60)
    if not matches:
        print("No matches found.")
        return

    for i, match in enumerate(matches, start=1):
        candidate_name = str(match.get("name", ""))
        aff = match.get("affiliations") or match.get("affiliation") or "N/A"
        if isinstance(aff, list):
            aff = ", ".join(str(item) for item in aff if str(item).strip()) or "N/A"
        score = _similarity(requested_name, candidate_name)
        print(f"{i:2d}. {candidate_name} | score={score:.3f} | affiliations={aff}")

    best = _best_match(requested_name, matches)
    if best:
        candidate_name = str(best.get("name", ""))
        print(f"Best match: {candidate_name} (score={best['score']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a person name against Semantic Scholar and Google Scholar matches.")
    parser.add_argument("--name", required=True, help="Person name to verify.")
    parser.add_argument(
        "--semantic-scholar-id",
        default="",
        help="Optional Semantic Scholar author ID. If provided, this will also verify the exact author profile.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of matches to print per source.")
    args = parser.parse_args()

    requested_name = args.name.strip()
    print(f"Requested name: {requested_name}")

    if args.semantic_scholar_id:
        author_id = args.semantic_scholar_id.strip()
        author_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
        try:
            data = _semantic_scholar_get(
                author_url,
                params={
                    "fields": "name,affiliations,paperCount",
                },
            )
            resolved_name = data.get("name", "")
            aff = data.get("affiliations") or "N/A"
            if isinstance(aff, list):
                aff = ", ".join(str(item) for item in aff if str(item).strip()) or "N/A"
            print("\nSemantic Scholar exact author")
            print("-" * 60)
            print(f"ID            : {author_id}")
            print(f"Name          : {resolved_name}")
            print(f"Affiliations  : {aff}")
            print(f"Name matches?  : {'yes' if _normalize(requested_name) == _normalize(resolved_name) else 'no'}")
        except Exception as exc:
            print(f"Semantic Scholar exact author lookup failed: {exc}")

    try:
        s2_matches = _semantic_scholar_matches(requested_name, limit=args.limit)
        _print_matches("Semantic Scholar", requested_name, s2_matches)
    except Exception as exc:
        print(f"Semantic Scholar search failed: {exc}")

    gs_matches = _google_scholar_matches(requested_name, limit=args.limit)
    _print_matches("Google Scholar", requested_name, gs_matches)

    s2_best = _best_match(requested_name, s2_matches) if 's2_matches' in locals() else None
    gs_best = _best_match(requested_name, gs_matches) if gs_matches else None

    print("\nSummary")
    print("-" * 60)
    if s2_best:
        print(f"Semantic Scholar best: {s2_best.get('name', 'N/A')} ({s2_best['score']:.3f})")
    if gs_best:
        print(f"Google Scholar best  : {gs_best.get('name', 'N/A')} ({gs_best['score']:.3f})")


def print_and_resolve_best_author(name: str, limit: int = 3) -> dict:
    """Print the top matches and return the selected Semantic Scholar author."""
    matches: list[dict] = []
    used_query = name.strip()
    for query in _candidate_queries(name):
        try:
            query_matches = _semantic_scholar_matches(query, limit=limit)
        except Exception as exc:
            print(f"Semantic Scholar search failed for '{query}': {exc}")
            continue

        if query_matches:
            matches = query_matches
            used_query = query
            break

    if not matches:
        raise RuntimeError(f"No Semantic Scholar author matches found for '{name}'")

    if _normalize(used_query) != _normalize(name):
        print(f"Using fallback Semantic Scholar query: {used_query}")

    _print_matches("Semantic Scholar", name, matches)

    best = _best_match(name, matches)
    if not best:
        raise RuntimeError(f"No Semantic Scholar author selected for '{name}'")

    author_id = str(best.get("authorId", "")).strip()
    if not author_id:
        raise RuntimeError(f"Selected Semantic Scholar author for '{name}' has no authorId")

    aff = best.get("affiliations") or "N/A"
    if isinstance(aff, list):
        aff = ", ".join(str(item) for item in aff if str(item).strip()) or "N/A"

    print("\nSelected author")
    print("-" * 60)
    print(f"Name         : {best.get('name', 'N/A')}")
    print(f"Affiliations : {aff}")
    print(f"Author ID    : {author_id}")
    return best


if __name__ == "__main__":
    main()
