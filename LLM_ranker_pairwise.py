# ─────────────────────────────────────────────────────────────────────────────
# LLM_ranker_pairwise.py
# Stage 2: Pairwise LLM Ranking (NEW)
#
# Instead of listwise ranking (comparing all ~10 papers at once), uses pairwise
# comparisons to build a ranking via a tournament or sorting approach.
# More stable predictions, better calibration, easier to interpret.
#
# Modes:
#   "tournament"  : Single-elimination tournament (3-4 LLM calls)
#   "bubble-sort" : Bubble sort style comparisons (up to n*(n-1)/2 calls)
#   "quicksort"   : Quicksort-based approach (~n*log(n) LLM calls)
#
# Dependencies: transformers, torch, pandas, config.py
# ─────────────────────────────────────────────────────────────────────────────

import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import config


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (same as listwise)
# ─────────────────────────────────────────────────────────────────────────────

def load_llm() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load LLM with optional 4-bit quantization."""
    model_candidates = [
        config.LLM_MODEL_NAME,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    model_candidates = list(dict.fromkeys(model_candidates))

    load_kwargs: dict = dict(
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if config.LLM_LOAD_IN_4BIT:
        print("  → 4-bit quantization enabled (bitsandbytes)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    last_exc: Exception | None = None
    for model_name in model_candidates:
        try:
            print(f"Loading LLM : {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model.eval()
            return tokenizer, model
        except Exception as exc:
            print(f"  -> Failed to load '{model_name}': {exc}")
            last_exc = exc

    raise RuntimeError(
        "Could not load any configured/fallback Hugging Face model. "
    ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Input text selection (same as listwise)
# ─────────────────────────────────────────────────────────────────────────────

def _select_paper_text(row: pd.Series) -> str:
    """Select abstract or full_text per config.STAGE2_INPUT_TEXT."""
    title    = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    full_text = str(row.get("full_text", "")).strip()

    if config.STAGE2_INPUT_TEXT == "full_text" and full_text:
        return full_text

    return f"{title}. {abstract}" if abstract else title


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise comparison prompt and inference
# ─────────────────────────────────────────────────────────────────────────────

def build_pairwise_prompt(query: str, paper_a: pd.Series, paper_b: pd.Series) -> str:
    """
    Build a pairwise comparison prompt.
    Ask the LLM which of two papers is more relevant to the query.
    """
    text_a = _select_paper_text(paper_a)
    text_b = _select_paper_text(paper_b)
    source_a = str(paper_a.get("text_source", "unknown"))
    source_b = str(paper_b.get("text_source", "unknown"))

    return (
        f"You are an expert scientific literature analyst.\n\n"
        f"Compare the following two papers for relevance to the query.\n\n"
        f"QUERY: \"{query}\"\n\n"
        f"PAPER A:\n"
        f"Title    : {paper_a['title']}\n"
        f"Year     : {paper_a.get('year', 'N/A')} | Venue: {paper_a.get('venue', 'N/A')}\n"
        f"Source   : {source_a}\n"
        f"Content  : {text_a}\n\n"
        f"PAPER B:\n"
        f"Title    : {paper_b['title']}\n"
        f"Year     : {paper_b.get('year', 'N/A')} | Venue: {paper_b.get('venue', 'N/A')}\n"
        f"Source   : {source_b}\n"
        f"Content  : {text_b}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Which paper is MORE relevant to the query?\n"
        f"2. Provide a relevance score for each (0.0–1.0)\n"
        f"3. Explain your reasoning (1–2 sentences)\n\n"
        f"OUTPUT FORMAT — follow exactly:\n"
        f"MORE RELEVANT: [A or B]\n"
        f"SCORE A: X.XX\n"
        f"SCORE B: Y.YY\n"
        f"Reasoning: ...\n"
    )


def generate_pairwise_comparison(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> str:
    """Run LLM on pairwise prompt and return generated text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    
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


_WINNER_RE = re.compile(r"MORE\s+RELEVANT\s*:\s*([AB])", re.IGNORECASE)
_SCORE_A_RE = re.compile(r"SCORE\s+A\s*:\s*([\d.]+)", re.IGNORECASE)
_SCORE_B_RE = re.compile(r"SCORE\s+B\s*:\s*([\d.]+)", re.IGNORECASE)


def parse_pairwise_output(llm_output: str) -> tuple[str, float, float, str]:
    """
    Parse pairwise LLM output.
    Returns: (winner, score_a, score_b, reasoning)
    """
    winner_match = _WINNER_RE.search(llm_output)
    winner = winner_match.group(1).upper() if winner_match else "A"

    score_a_match = _SCORE_A_RE.search(llm_output)
    score_a = float(score_a_match.group(1)) if score_a_match else 0.5

    score_b_match = _SCORE_B_RE.search(llm_output)
    score_b = float(score_b_match.group(1)) if score_b_match else 0.5

    reasoning_match = re.search(r"Reasoning\s*:\s*(.+)", llm_output, re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Model output unavailable."

    return winner, score_a, score_b, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Sorting strategies
# ─────────────────────────────────────────────────────────────────────────────

def _bubble_sort_pairwise(
    papers: pd.DataFrame,
    query: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> pd.DataFrame:
    """
    Bubble-sort style pairwise ranking.
    Makes O(n²) comparisons in worst case, but typically faster for small n.
    """
    n = len(papers)
    sorted_papers = papers.copy()
    comparison_count = 0

    print(f"Starting bubble-sort pairwise ranking (n={n})...")

    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            paper_j = sorted_papers.iloc[j]
            paper_j_plus_1 = sorted_papers.iloc[j + 1]

            prompt = build_pairwise_prompt(query, paper_j, paper_j_plus_1)
            raw_output = generate_pairwise_comparison(prompt, tokenizer, model)
            winner, score_a, score_b, reasoning = parse_pairwise_output(raw_output)
            comparison_count += 1

            if winner == "B":
                # Swap
                sorted_papers.iloc[[j, j + 1]] = sorted_papers.iloc[[j + 1, j]].values
                swapped = True
                status = f"Swapped ({comparison_count})"
            else:
                status = f"Keep ({comparison_count})"

            print(
                f"  [{status}] Pos {j+1} vs {j+2}: "
                f"{paper_j['title'][:45]} vs {paper_j_plus_1['title'][:45]}"
            )

        if not swapped:
            break

    print(f"Completed {comparison_count} pairwise comparisons.")

    result = sorted_papers.reset_index(drop=True)
    # Bubble-sort result is already ordered best->worst.
    result["stage2_rank"] = result.index + 1
    return result


def _tournament_pairwise(
    papers: pd.DataFrame,
    query: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> pd.DataFrame:
    """
    Single-elimination tournament ranking.
    Makes ~n-1 comparisons (fast, but order-dependent).
    """
    n = len(papers)
    remaining = list(range(n))
    comparison_count = 0

    print(f"Starting tournament-style pairwise ranking (n={n})...")

    while len(remaining) > 1:
        next_round = []

        for i in range(0, len(remaining), 2):
            idx_a = remaining[i]
            if i + 1 < len(remaining):
                idx_b = remaining[i + 1]
                
                paper_a = papers.iloc[idx_a]
                paper_b = papers.iloc[idx_b]

                prompt = build_pairwise_prompt(query, paper_a, paper_b)
                raw_output = generate_pairwise_comparison(prompt, tokenizer, model)
                winner, score_a, score_b, reasoning = parse_pairwise_output(raw_output)
                comparison_count += 1

                if winner == "B":
                    next_round.append(idx_b)
                    loser_title = paper_a['title'][:35]
                    winner_title = paper_b['title'][:35]
                else:
                    next_round.append(idx_a)
                    loser_title = paper_b['title'][:35]
                    winner_title = paper_a['title'][:35]

                print(
                    f"  [{comparison_count:2d}] {winner_title} > {loser_title}"
                )
            else:
                # Odd paper, auto-advance
                next_round.append(idx_a)

        remaining = next_round

    print(f"Completed {comparison_count} pairwise comparisons (tournament).")

    # Rank by tournament order (remaining[0] = winner)
    ranking = [0] * n
    for rank, idx in enumerate(remaining + [i for i in range(n) if i not in remaining]):
        ranking[idx] = rank + 1

    result = papers.copy()
    result["stage2_rank"] = [ranking[i] for i in range(n)]
    result = result.sort_values("stage2_rank").reset_index(drop=True)
    result["stage2_rank"] = result.index + 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(top_k_df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Run fine pairwise LLM ranking on the top-k papers.

    Parameters
    ----------
    top_k_df : pd.DataFrame
        Output of stage1_reranker.run_stage1().
    query : str
        Search query.

    Returns
    -------
    pd.DataFrame
        Ranked papers with stage2_rank, stage2_score, stage2_rationale.
        All original columns preserved.
    """
    print("\n" + "="*60)
    print("STAGE 2 — Fine Ranking with Pairwise LLM Comparison")
    print("="*60)
    print(f"Input text mode : {config.STAGE2_INPUT_TEXT}")
    print(f"Pairwise method : {getattr(config, 'LLM_PAIRWISE_METHOD', 'bubble-sort')}")

    tokenizer, model = load_llm()

    # Choose pairwise strategy
    method = getattr(config, 'LLM_PAIRWISE_METHOD', 'bubble-sort')
    if method == "tournament":
        ranked_df = _tournament_pairwise(top_k_df, query, tokenizer, model)
    else:
        # Default: bubble-sort
        ranked_df = _bubble_sort_pairwise(top_k_df, query, tokenizer, model)

    # Safety guard: ensure rank column always exists.
    if "stage2_rank" not in ranked_df.columns:
        ranked_df = ranked_df.reset_index(drop=True)
        ranked_df["stage2_rank"] = ranked_df.index + 1

    # Assign scores based on rank
    ranked_df["stage2_score"] = (
        1.0 - (ranked_df["stage2_rank"] - 1) / len(ranked_df)
    ).round(2)
    ranked_df["stage2_score"] = ranked_df["stage2_score"].clip(lower=0.0)

    ranked_df["stage2_rationale"] = (
        f"Ranked via pairwise LLM comparison ({method} method)."
    )

    _print_results(ranked_df)

    print(f"\n✅ Stage 2 complete — pairwise ranking finished.")
    return ranked_df


def _print_results(df: pd.DataFrame) -> None:
    """Print final Stage 2 pairwise ranking results."""
    print(f"\n{'─'*70}")
    print(f"{'Rank':<6} {'Score':>8} {'Title':<50}")
    print(f"{'─'*70}")
    for _, row in df.iterrows():
        print(
            f"  [{int(row['stage2_rank']):2d}]  "
            f"{row['stage2_score']:>7.2f}  "
            f"{str(row['title'])[:50]}"
        )
    print(f"{'─'*70}")
