# ─────────────────────────────────────────────────────────────────────────────
# reranker_1_bm25_hybrid.py
# Stage 1: Hybrid BM25 + Cross-Encoder ranking (NEW)
#
# This variant combines keyword-based BM25 ranking with semantic cross-encoder
# scoring for more robust relevance detection. BM25 catches exact/near keyword
# matches; cross-encoder captures semantic relevance.
#
# Input  : DataFrame from scraper.py (columns: abstract, full_text, pdf_url, 
#          text_source, plus metadata)
# Scoring: Hybrid = α * norm(cross_encoder) + (1-α) * norm(BM25)
#          where α is configured in config.RERANKER_HYBRID_ALPHA
# Output : Top-k papers sorted by hybrid_score, all original columns preserved.
#
# Dependencies: sentence-transformers, rank-bm25, pandas, config.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Plus

import config


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_document_text(row: pd.Series) -> str:
    """
    Build the document string fed to the cross-encoder.
    Uses title + abstract (intentionally excludes full_text for Stage 1).
    """
    title    = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    if abstract:
        return f"{title}. {abstract}"
    return title


def _build_pairs(df: pd.DataFrame, query: str) -> list[tuple[str, str]]:
    """Build (query, document) pairs for cross-encoder."""
    return [(query, _build_document_text(row)) for _, row in df.iterrows()]


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25 scoring.
    Simple whitespace + lowercase tokenization.
    In production, use NLTK or spaCy for better tokenization.
    """
    # Simple lowercasing + splitting on whitespace/punctuation
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def _build_bm25_corpus(df: pd.DataFrame) -> tuple[BM25Plus, list[list[str]]]:
    """
    Build a BM25 corpus from the DataFrame.
    Return the BM25 object and tokenized documents.
    """
    corpus = []
    for _, row in df.iterrows():
        doc_text = _build_document_text(row)
        tokens = _tokenize_for_bm25(doc_text)
        corpus.append(tokens)
    
    bm25 = BM25Plus(corpus)
    return bm25, corpus


def _score_bm25(query: str, bm25: BM25Plus, corpus: list[list[str]]) -> np.ndarray:
    """
    Score all documents in corpus using BM25.
    Returns array of BM25 scores (normalized to [0, 1]).
    """
    query_tokens = _tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)
    
    # Normalize to [0, 1]
    scores_arr = np.array(scores, dtype=np.float32)
    max_score = scores_arr.max()
    if max_score > 0:
        scores_arr = scores_arr / max_score
    
    return scores_arr


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize a score array to [0, 1]."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-6:
        return np.ones_like(scores) * 0.5
    return (scores - min_val) / (max_val - min_val)


def _load_reranker(model_name: str, max_length: int) -> CrossEncoder:
    """Load and return a CrossEncoder reranker model."""
    print(f"Loading cross-encoder : {model_name}")
    return CrossEncoder(model_name, max_length=max_length)


def _print_results(df: pd.DataFrame, top_k: int) -> None:
    """Print Stage 1 hybrid results with component scores."""
    print(f"\n{'─'*80}")
    print(
        f"{'Rank':<6} {'Hybrid':>8} {'CrossEnc':>8} {'BM25':>8} "
        f"{'Source':<22} Title"
    )
    print(f"{'─'*80}")
    for _, row in df.head(top_k).iterrows():
        source = str(row.get("text_source", "unknown"))
        print(
            f"  [{int(row['stage1_rank']):2d}]  "
            f"{row.get('stage1_score', 0.0):>7.4f}  "
            f"{row.get('stage1_cross_encoder_score', 0.0):>7.4f}  "
            f"{row.get('stage1_bm25_score', 0.0):>7.4f}  "
            f"{source:<22} "
            f"{str(row['title'])[:40]}"
        )
    print(f"{'─'*80}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(df: pd.DataFrame, query: str, top_k: int) -> pd.DataFrame:
    """
    Score papers using BM25 + Cross-Encoder hybrid and return top-k.

    Parameters
    ----------
    df : pd.DataFrame
        Papers from scraper.fetch_papers().
    query : str
        Search query.
    top_k : int
        Number of top papers to keep.

    Returns
    -------
    pd.DataFrame
        Top-k papers sorted by stage1_score (hybrid) descending.
        New columns:
          - stage1_score                    : hybrid score
          - stage1_cross_encoder_score      : normalized CE score
          - stage1_bm25_score               : normalized BM25 score
          - stage1_rank                     : rank position
        All original columns preserved.
    """
    print("\n" + "="*80)
    print("STAGE 1 — Hybrid BM25 + Cross-Encoder Ranking")
    print("="*80)
    print(f"Query : \"{query[:80]}\"")
    print(f"Papers: {len(df)} | Keeping top: {top_k}")
    print(f"Hybrid weight (α=cross-encoder): {getattr(config, 'RERANKER_HYBRID_ALPHA', 0.6)}")

    # Build BM25 corpus
    print("\nBuilding BM25 index...")
    bm25, corpus = _build_bm25_corpus(df)
    bm25_scores = _score_bm25(query, bm25, corpus)
    bm25_scores_norm = _normalize_scores(bm25_scores)

    # Score with cross-encoder
    print("Loading and scoring with cross-encoder...")
    reranker = _load_reranker(config.RERANKER_MODEL_NAME, config.RERANKER_MAX_LENGTH)
    pairs = _build_pairs(df, query)
    cross_encoder_scores = reranker.predict(pairs, show_progress_bar=True)
    cross_encoder_scores_norm = _normalize_scores(np.array(cross_encoder_scores, dtype=np.float32))

    # Compute hybrid score
    alpha = getattr(config, 'RERANKER_HYBRID_ALPHA', 0.6)
    hybrid_scores = alpha * cross_encoder_scores_norm + (1 - alpha) * bm25_scores_norm

    df = df.copy()
    df["stage1_cross_encoder_score"] = cross_encoder_scores_norm
    df["stage1_bm25_score"] = bm25_scores_norm
    df["stage1_score"] = hybrid_scores

    df_ranked = (
        df.sort_values("stage1_score", ascending=False)
        .reset_index(drop=True)
    )
    df_ranked["stage1_rank"] = df_ranked.index + 1

    _print_results(df_ranked, top_k)

    top_k_df = df_ranked.head(top_k).reset_index(drop=True)

    # Summary
    n_full = (top_k_df["full_text"].fillna("").str.len() > 0).sum()
    print(f"\n✅ Stage 1 complete — kept top {top_k} papers (hybrid scoring).")
    print(f"   Full text available in top-{top_k}: {n_full}/{top_k}")
    print(f"   Hybrid weight: α={alpha:.2f} (cross-encoder) + (1-α)={1-alpha:.2f} (BM25)")

    return top_k_df
