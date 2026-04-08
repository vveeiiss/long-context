# ─────────────────────────────────────────────────────────────────────────────
# stage2_llm_ranker.py
# Stage 2: Fine listwise ranking of top-k papers using an LLM.
#
# Input text per paper is controlled by config.STAGE2_INPUT_TEXT:
#   "abstract"  → title + abstract (~400 tokens/paper, fast)
#   "full_text" → full paper text  (~7500 tokens/paper, richer context)
#                 Falls back to abstract if full_text is empty for a paper.
#
# All original columns (including full_text, pdf_url, text_source) are
# preserved in the output DataFrame for downstream Stage 3 use.
#
# Dependencies: transformers, torch, pandas, config.py
# ─────────────────────────────────────────────────────────────────────────────

import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import config


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the LLM and tokenizer specified in config.
    Supports optional 4-bit quantization via bitsandbytes.

    Returns
    -------
    tokenizer : AutoTokenizer
    model     : AutoModelForCausalLM
    """
    print(f"Loading LLM : {config.LLM_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)

    load_kwargs: dict = dict(
        torch_dtype=torch.float16,
        device_map="auto",  # distributes across all available GPUs automatically
    )

    if config.LLM_LOAD_IN_4BIT:
        print("  → 4-bit quantization enabled (bitsandbytes)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL_NAME, **load_kwargs)
    model.eval()
    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Input text selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_paper_text(row: pd.Series) -> str:
    """
    Select the text to represent a paper in the Stage 2 prompt.

    Respects config.STAGE2_INPUT_TEXT:
      - "full_text" : use full_text if available, else fall back to abstract
      - "abstract"  : always use abstract (title + abstract)

    Parameters
    ----------
    row : pd.Series — one row from the top-k DataFrame

    Returns
    -------
    str — the text to embed in the prompt for this paper
    """
    title    = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    full_text = str(row.get("full_text", "")).strip()

    if config.STAGE2_INPUT_TEXT == "full_text" and full_text:
        return full_text

    # abstract mode, or full_text requested but unavailable
    return f"{title}. {abstract}" if abstract else title


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(query: str, papers: pd.DataFrame) -> str:
    """
    Build a listwise ranking prompt containing all papers.
    Text per paper is selected according to config.STAGE2_INPUT_TEXT.

    Parameters
    ----------
    query  : str
    papers : pd.DataFrame  (top-k from Stage 1)

    Returns
    -------
    str — the full prompt string ready for the LLM
    """
    n = len(papers)
    paper_blocks = []

    for i, (_, row) in enumerate(papers.iterrows(), start=1):
        text = _select_paper_text(row)
        source = str(row.get("text_source", "unknown"))
        block = (
            f"[Paper {i}]\n"
            f"Title   : {row['title']}\n"
            f"Year    : {row.get('year', 'N/A')} | Venue: {row.get('venue', 'N/A')}\n"
            f"Source  : {source}\n"
            f"Content : {text}\n"
        )
        paper_blocks.append(block)

    papers_text = "\n".join(paper_blocks)

    return (
        f"You are an expert scientific literature analyst.\n\n"
        f"Rank the following {n} papers by relevance to the query below.\n\n"
        f"QUERY: \"{query}\"\n\n"
        f"PAPERS:\n{papers_text}\n"
        f"INSTRUCTIONS:\n"
        f"1. Rank all {n} papers from most relevant (rank 1) to least relevant (rank {n}).\n"
        f"2. For each paper provide:\n"
        f"   - Its rank\n"
        f"   - A relevance score from 0.0 to 1.0\n"
        f"   - A concise rationale (1–2 sentences) grounded in the paper content\n\n"
        f"OUTPUT FORMAT — follow exactly:\n"
        f"RANK 1 | Paper [X] | Score: 0.XX\n"
        f"Rationale: ...\n\n"
        f"RANK 2 | Paper [X] | Score: 0.XX\n"
        f"Rationale: ...\n\n"
        f"(continue for all {n} papers)\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM inference
# ─────────────────────────────────────────────────────────────────────────────

def generate_ranking(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """
    Run the LLM on the prompt and return only the newly generated text.

    Parameters
    ----------
    prompt    : str
    tokenizer : AutoTokenizer
    model     : AutoModelForCausalLM

    Returns
    -------
    str — raw LLM output (prompt tokens stripped)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"Prompt length : {prompt_len} tokens")

    if prompt_len > 120_000:
        print(
            f"  ⚠️  Prompt is very long ({prompt_len} tokens). "
            "Consider switching config.STAGE2_INPUT_TEXT to 'abstract'."
        )

    print("Running LLM inference...")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────

_RANK_RE      = re.compile(
    r"RANK\s+(\d+)\s*\|\s*Paper\s+(\d+)\s*\|\s*Score:\s*([\d.]+)",
    re.IGNORECASE,
)
_RATIONALE_RE = re.compile(r"Rationale:\s*(.+)", re.IGNORECASE)


def parse_llm_output(llm_output: str, papers: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the LLM's listwise output into a structured DataFrame.
    All original columns (full_text, pdf_url, text_source, stage1_score,
    stage1_rank, etc.) are preserved in the output rows.

    Falls back to Stage 1 order with empty rationales if parsing fails.

    Parameters
    ----------
    llm_output : str
        Raw text generated by the LLM.
    papers : pd.DataFrame
        The top-k papers passed to the LLM.

    Returns
    -------
    pd.DataFrame with all original columns plus:
        stage2_rank      : int
        stage2_score     : float
        stage2_rationale : str
    """
    results = []
    current: dict = {}

    for line in llm_output.strip().splitlines():
        rank_match = _RANK_RE.search(line)
        if rank_match:
            current = {
                "stage2_rank":  int(rank_match.group(1)),
                "paper_idx":    int(rank_match.group(2)) - 1,  # 0-indexed
                "stage2_score": float(rank_match.group(3)),
            }
            continue

        rationale_match = _RATIONALE_RE.search(line)
        if rationale_match and current:
            idx = current["paper_idx"]
            if 0 <= idx < len(papers):
                row = papers.iloc[idx].to_dict()   # preserves ALL columns
                row["stage2_rank"]      = current["stage2_rank"]
                row["stage2_score"]     = current["stage2_score"]
                row["stage2_rationale"] = rationale_match.group(1).strip()
                results.append(row)
            current = {}

    if not results:
        print("  ⚠️  Could not parse LLM output — falling back to Stage 1 order.")
        fallback = papers.copy()
        fallback["stage2_rank"]      = range(1, len(papers) + 1)
        fallback["stage2_score"]     = None
        fallback["stage2_rationale"] = ""
        return fallback

    return (
        pd.DataFrame(results)
        .sort_values("stage2_rank")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(top_k_df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Run fine listwise LLM ranking on the top-k papers from Stage 1.

    Parameters
    ----------
    top_k_df : pd.DataFrame
        Output of stage1_reranker.run_stage1().
        Must contain: title, abstract, full_text, pdf_url, text_source,
        stage1_score, stage1_rank (plus rank_original, year, venue, etc.)
    query : str
        The search query string.

    Returns
    -------
    pd.DataFrame
        Final ranked papers. All columns from Stage 0 and Stage 1 are
        preserved. Three new columns added:
          - stage2_rank      : final rank (1 = most relevant)
          - stage2_score     : LLM relevance score (0.0–1.0)
          - stage2_rationale : natural language explanation per paper
    """
    print("\n" + "="*60)
    print("STAGE 2 — Fine ranking with LLM (listwise)")
    print("="*60)
    print(f"Input text mode : {config.STAGE2_INPUT_TEXT}")

    n_full = (top_k_df["full_text"].fillna("").str.len() > 0).sum()
    n_abstract_only = len(top_k_df) - n_full
    print(f"Papers with full text : {n_full}/{len(top_k_df)}")
    if n_abstract_only > 0:
        print(f"  ℹ️  {n_abstract_only} papers will use abstract as fallback.")

    tokenizer, model = load_llm()
    prompt           = build_prompt(query, top_k_df)
    raw_output       = generate_ranking(prompt, tokenizer, model)

    print("\n--- Raw LLM Output (first 2000 chars) ---")
    print(raw_output[:2000])

    ranked_df = parse_llm_output(raw_output, top_k_df)
    _print_results(ranked_df)

    print("\n✅ Stage 2 complete.")
    return ranked_df


def _print_results(df: pd.DataFrame) -> None:
    """Print a formatted summary of the Stage 2 ranking."""
    print(f"\n{'─'*60}")
    print("Stage 2 Final Ranking")
    print(f"{'─'*60}")
    for _, row in df.iterrows():
        score  = f"{row['stage2_score']:.2f}" if row.get("stage2_score") is not None else "N/A"
        source = str(row.get("text_source", "unknown"))
        print(f"\n  Rank {int(row['stage2_rank']):2d} | Score: {score} | Source: {source}")
        print(f"  Title     : {str(row['title'])[:80]}")
        print(f"  Rationale : {str(row.get('stage2_rationale', ''))[:120]}")
    print(f"{'─'*60}")
