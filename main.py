# ─────────────────────────────────────────────────────────────────────────────
# main.py
# Entry point for the Scientific IR Pipeline.
#
# Runs Stage 0 → Stage 1 → Stage 2 in sequence.
# Each stage saves its output to a CSV so you can resume from any point.
#
# Usage:
#   python main.py                  # run full pipeline
#   python main.py --skip-scrape    # skip Stage 0, load from existing CSV
#
# File layout:
#   main.py
#   config.py
#   scraper.py
#   stage1_reranker.py
#   stage2_llm_ranker.py
#   requirements.txt
#   data/
#     papers_raw.csv      ← Stage 0 output
#     papers_stage1.csv   ← Stage 1 output
#     papers_stage2.csv   ← Stage 2 output
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os

import config
import scraper
import stage1_reranker
import stage2_llm_ranker


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scientific IR Pipeline — Stage 0 → 1 → 2"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help=(
            f"Skip Stage 0 and load papers from '{config.CSV_RAW_PATH}'. "
            "Use this if you have already scraped the profile."
        ),
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help=(
            f"Skip Stage 1 and load top-k papers from '{config.CSV_STAGE1_PATH}'. "
            "Use this to re-run Stage 2 with a different LLM or query."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    os.makedirs("data", exist_ok=True)

    # ── Stage 0: Scrape Google Scholar profile ────────────────────────────────
    if args.skip_scrape or args.skip_stage1:
        # Load whichever CSV is most appropriate
        raw_path = config.CSV_RAW_PATH
        if os.path.exists(raw_path):
            df_papers = scraper.load_papers(raw_path)
        else:
            raise FileNotFoundError(
                f"--skip-scrape was set but '{raw_path}' does not exist. "
                "Run without --skip-scrape first."
            )
    else:
        df_papers = scraper.fetch_papers(
            profile_url=config.SCHOLAR_PROFILE_URL,
            n_papers=config.N_PAPERS,
        )
        scraper.save_papers(df_papers, config.CSV_RAW_PATH)

    _print_column_summary("Stage 0 output", df_papers)

    # ── Stage 1: Coarse ranking → Top-k ──────────────────────────────────────
    if args.skip_stage1:
        stage1_path = config.CSV_STAGE1_PATH
        if os.path.exists(stage1_path):
            top_k_df = scraper.load_papers(stage1_path)
        else:
            raise FileNotFoundError(
                f"--skip-stage1 was set but '{stage1_path}' does not exist. "
                "Run without --skip-stage1 first."
            )
    else:
        top_k_df = stage1_reranker.run_stage1(
            df=df_papers,
            query=config.QUERY,
            top_k=config.TOP_K,
        )
        scraper.save_papers(top_k_df, config.CSV_STAGE1_PATH)

    _print_column_summary("Stage 1 output", top_k_df)

    # ── Stage 2: Fine LLM ranking + rationale extraction ─────────────────────
    final_df = stage2_llm_ranker.run_stage2(
        top_k_df=top_k_df,
        query=config.QUERY,
    )
    scraper.save_papers(final_df, config.CSV_STAGE2_PATH)

    _print_column_summary("Stage 2 output", final_df)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"  Query              : {config.QUERY}")
    print(f"  Papers scraped     : {len(df_papers)}")
    print(f"  After Stage 1      : {len(top_k_df)}")
    print(f"  Final ranking      : {len(final_df)}")
    print(f"  Stage 2 input mode : {config.STAGE2_INPUT_TEXT}")
    n_full = (final_df["full_text"].fillna("").str.len() > 0).sum()
    print(f"  Papers w/ full text: {n_full}/{len(final_df)}")
    print(f"\n  Output files:")
    print(f"    {config.CSV_RAW_PATH}")
    print(f"    {config.CSV_STAGE1_PATH}")
    print(f"    {config.CSV_STAGE2_PATH}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_column_summary(label: str, df) -> None:
    """Print a brief summary of a DataFrame's columns and shape."""
    print(f"\n[{label}] shape={df.shape}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
