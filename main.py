# main.py
#
# Run the full pipeline (Stage 0 -> Stage 1 -> Stage 2).
#
# Usage (by author):
#   python main.py --person "Author Name"
#   python main.py --person 123456789 --n-papers 20
#
# Usage (by paper URL):
#   python main.py --paper-url "https://arxiv.org/abs/2345.12345"
#   python main.py --paper-url "https://www.semanticscholar.org/paper/..."
#
# Combined:
#   python main.py --person "Author Name" --paper-url "https://arxiv.org/abs/..."

from __future__ import annotations

import argparse
import os

import config
import scraper
import reranker_1
import LLM_ranker
import verify_author_match
import paper_url_parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Scientific IR pipeline (Stage 0 -> Stage 1 -> Stage 2) "
                    "for a specific person and/or paper URL."
    )
    parser.add_argument(
        "--person",
        default=None,
        help="Semantic Scholar author name, author URL, or author ID. "
             "If omitted, must provide --paper-url.",
    )
    parser.add_argument(
        "--paper-url",
        default=None,
        help="URL of a paper (arXiv, Semantic Scholar, or generic HTTP). "
             "If provided, its abstract will be used as the query for ranking.",
    )
    parser.add_argument(
        "--n-papers",
        type=int,
        default=config.N_PAPERS,
        help="Preferred number of papers to fetch before fallback logic is applied.",
    )
    return parser.parse_args()


def run_pipeline(
    person_identifier: str | None,
    paper_url: str | None,
    n_papers: int,
) -> None:
    """
    Run Stage 0 -> Stage 1 -> Stage 2 end-to-end.
    
    Parameters
    ----------
    person_identifier : str | None
        Semantic Scholar author name, author URL, or author ID.
        If None, must provide paper_url.
    paper_url : str | None
        URL of a paper. If provided, its abstract will be used as the query.
    n_papers : int
        Preferred number of papers to fetch.
    """
    if not person_identifier and not paper_url:
        raise ValueError("Must provide either --person or --paper-url (or both).")
    
    os.makedirs("data", exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Extract paper metadata if provided
    # ─────────────────────────────────────────────────────────────────────────────
    query = config.QUERY
    paper_info = None
    
    if paper_url:
        print("\n" + "=" * 60)
        print("EXTRACTING PAPER METADATA")
        print("=" * 60)
        try:
            paper_info = paper_url_parser.extract_paper_metadata(paper_url)
            query = paper_info["abstract"]
            print(f"\n✓ Extracted query from paper: {paper_info['title'][:60]}...")
        except ValueError as e:
            print(f"\n✗ Failed to extract paper metadata: {e}")
            if not person_identifier:
                raise
            print("  Falling back to --person mode and config.QUERY...")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Resolve author if provided
    # ─────────────────────────────────────────────────────────────────────────────
    resolved_name = "N/A"
    selected_affiliations = "N/A"
    resolved_person = person_identifier
    
    if person_identifier:
        selected_author = verify_author_match.print_and_resolve_best_author(person_identifier, limit=3)
        resolved_person = str(selected_author.get("authorId", person_identifier)).strip() or person_identifier
        resolved_name = str(selected_author.get("name", person_identifier)).strip() or person_identifier
        selected_affiliations = selected_author.get("affiliations") or "N/A"
        if isinstance(selected_affiliations, list):
            selected_affiliations = ", ".join(str(item) for item in selected_affiliations if str(item).strip()) or "N/A"
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Pipeline start
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCIENTIFIC IR PIPELINE — START")
    print("=" * 60)
    if person_identifier:
        print(f"Target person : {person_identifier}")
        print(f"Selected name : {resolved_name}")
        print(f"Selected aff. : {selected_affiliations}")
    if paper_url:
        print(f"Paper URL     : {paper_url[:70]}...")
        if paper_info:
            print(f"Paper title   : {paper_info['title'][:60]}...")
    print(f"Query (first)  : {query[:70]}...")
    print(f"Preferred N    : {n_papers}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Stage 0: Fetch papers for one person or based on title search
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Stage 0] Fetching papers and full-text sources...")
    if person_identifier:
        df_papers = scraper.fetch_papers(
            person_identifier=resolved_person,
            n_papers=n_papers,
        )
    else:
        # No person provided; search by paper title if available
        if paper_info and paper_info.get("title"):
            print(f"  Searching Semantic Scholar for papers similar to: {paper_info['title'][:60]}...")
            df_papers = scraper.fetch_papers_by_title_search(
                title=paper_info["title"],
                n_papers=n_papers,
            )
        else:
            raise RuntimeError("Cannot fetch papers without --person or paper title.")
    
    scraper.save_papers(df_papers, config.CSV_RAW_PATH)
    print(f"[Stage 0] Output shape: {df_papers.shape}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Stage 1: Coarse ranking -> Top-k
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Stage 1] Cross-encoder reranking (coarse)...")
    top_k_df = reranker_1.run_stage1(
        df=df_papers,
        query=query,
        top_k=config.TOP_K,
    )
    scraper.save_papers(top_k_df, config.CSV_STAGE1_PATH)
    print(f"[Stage 1] Output shape: {top_k_df.shape}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Stage 2: Fine LLM ranking + rationale extraction
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Stage 2] LLM-based listwise ranking and rationale extraction...")
    final_df = LLM_ranker.run_stage2(
        top_k_df=top_k_df,
        query=query,
    )
    scraper.save_papers(final_df, config.CSV_STAGE2_PATH)
    print(f"[Stage 2] Output shape: {final_df.shape}")

    n_full = (final_df["full_text"].fillna("").str.len() > 0).sum()

    # ─────────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCIENTIFIC IR PIPELINE — COMPLETE")
    print("=" * 60)
    if person_identifier:
        print(f"  Person             : {person_identifier}")
        print(f"  Selected name      : {resolved_name}")
    if paper_url:
        print(f"  Paper URL          : {paper_url[:60]}...")
    print(f"  Query              : {query[:70]}...")
    print(f"  Papers scraped     : {len(df_papers)}")
    print(f"  After Stage 1      : {len(top_k_df)}")
    print(f"  Final ranking      : {len(final_df)}")
    print(f"  Stage 2 input mode : {config.STAGE2_INPUT_TEXT}")
    print(f"  Papers w/ full text: {n_full}/{len(final_df)}")
    print("\n  Output files:")
    print(f"    {config.CSV_RAW_PATH}")
    print(f"    {config.CSV_STAGE1_PATH}")
    print(f"    {config.CSV_STAGE2_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.person, args.paper_url, args.n_papers)
