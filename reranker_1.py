# ─────────────────────────────────────────────────────────────────────────────
# stage1_reranker.py
# Stage 1: Coarse ranking of N papers using a cross-encoder reranker.
#
# Input  : DataFrame from scraper.py (columns include abstract, full_text,
#          pdf_url, text_source from the updated scraper)
# Scoring: (query, title + abstract) pairs — abstract only, intentionally.
#          full_text is carried through untouched for Stage 2 / Stage 3.
# Output : Top-k papers sorted by stage1_score, all original columns preserved.
#
# Dependencies: sentence-transformers, pandas, config.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from sentence_transformers import CrossEncoder

import config


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_document_text(row: pd.Series) -> str:
    """
    Build the document string fed to the cross-encoder.
    Uses title + abstract only (cross-encoder max_length = 512 tokens).
    full_text is intentionally excluded here — it is preserved in the
    DataFrame for use in Stage 2 and Stage 3.
    """
    title    = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    if abstract:
        return f"{title}. {abstract}"
    return title


def _build_pairs(df: pd.DataFrame, query: str) -> list[tuple[str, str]]:
    """Build a list of (query, document) pairs for the cross-encoder."""
    return [(query, _build_document_text(row)) for _, row in df.iterrows()]


def _load_reranker(model_name: str, max_length: int) -> CrossEncoder:
    """Load and return a CrossEncoder reranker model."""
    print(f"Loading reranker : {model_name}")
    return CrossEncoder(model_name, max_length=max_length)


def _print_results(df: pd.DataFrame, top_k: int) -> None:
    """Print a formatted summary of the top-k Stage 1 results."""
    print(f"\n{'─'*60}")
    print(f"{'Rank':<6} {'Score':>7}  {'Source':<22}  Title")
    print(f"{'─'*60}")
    for _, row in df.head(top_k).iterrows():
        source = str(row.get("text_source", "unknown"))
        print(
            f"  [{int(row['stage1_rank']):2d}]  "
            f"{row['stage1_score']:>6.4f}  "
            f"{source:<22}  "
            f"{str(row['title'])[:55]}"
        )
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(df: pd.DataFrame, query: str, top_k: int) -> pd.DataFrame:
    """
    Score all papers against the query with a cross-encoder and return top-k.

    Parameters
    ----------
    df : pd.DataFrame
        Papers DataFrame from scraper.fetch_papers().
        Expected columns: title, abstract, full_text, pdf_url, text_source
        (plus rank_original, year, venue, authors, citations).
    query : str
        The search query string.
    top_k : int
        Number of top papers to keep.

    Returns
    -------
    pd.DataFrame
        Top-k papers sorted by stage1_score descending.
        All original columns (including full_text, pdf_url, text_source)
        are preserved unchanged.
        Two new columns added:
          - stage1_score : raw cross-encoder relevance score
          - stage1_rank  : rank position (1 = most relevant)
    """
    print("\n" + "="*60)
    print("STAGE 1 — Coarse ranking with cross-encoder reranker")
    print("="*60)
    print(f"Query : \"{query[:80]}\"")
    print(f"Papers: {len(df)} | Keeping top: {top_k}")

    # Warn if many papers lack abstracts (scoring will be title-only)
    n_no_abstract = (df["abstract"].fillna("").str.strip() == "").sum()
    if n_no_abstract > 0:
        print(f"  ⚠️  {n_no_abstract} papers have no abstract — scoring on title only.")

    reranker = _load_reranker(config.RERANKER_MODEL_NAME, config.RERANKER_MAX_LENGTH)
    pairs    = _build_pairs(df, query)

    print(f"Scoring {len(pairs)} (query, document) pairs...")
    scores = reranker.predict(pairs, show_progress_bar=True)

    df = df.copy()
    df["stage1_score"] = scores
    df_ranked = (
        df.sort_values("stage1_score", ascending=False)
        .reset_index(drop=True)
    )
    df_ranked["stage1_rank"] = df_ranked.index + 1

    _print_results(df_ranked, top_k)

    top_k_df = df_ranked.head(top_k).reset_index(drop=True)

    # Summary: how many of the top-k have full text available
    n_full = (top_k_df["full_text"].fillna("").str.len() > 0).sum()
    print(f"\n✅ Stage 1 complete — kept top {top_k} papers.")
    print(f"   Full text available in top-{top_k}: {n_full}/{top_k}")

    return top_k_df
