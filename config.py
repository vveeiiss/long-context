# ─────────────────────────────────────────────────────────────────────────────
# config.py
# Central configuration for the Scientific IR Pipeline.
# All other modules import from here — edit only this file to change settings.
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from a .env file if present

# ── Google Scholar ────────────────────────────────────────────────────────────
SCHOLAR_PROFILE_URL: str = os.getenv(
    "SCHOLAR_PROFILE_URL",
    "https://scholar.google.com/citations?user=AVDkgFIAAAAJ&hl",
)
N_PAPERS: int = 10  # number of most-recent papers to fetch

# ── Query ─────────────────────────────────────────────────────────────────────
QUERY: str = "revisiting text ranking in deep research"

# ── File paths ────────────────────────────────────────────────────────────────
CSV_RAW_PATH:    str = "data/papers_raw.csv"      # Stage 0 output
CSV_STAGE1_PATH: str = "data/papers_stage1.csv"   # Stage 1 output (top-k)
CSV_STAGE2_PATH: str = "data/papers_stage2.csv"   # Stage 2 output (final)

# ── Stage 1 — Cross-encoder reranker ─────────────────────────────────────────
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

# Stage 1 uses title + abstract only (cross-encoder token budget).
# full_text is preserved in the DataFrame for later stages.
RERANKER_MAX_LENGTH: int = 512

TOP_K: int = 5  # papers to keep after Stage 1

# ── Stage 2 — LLM ────────────────────────────────────────────────────────────
# Swap model name here to change the LLM used in Stage 2.
# Options:
#   "Qwen/Qwen2.5-7B-Instruct"
#   "Qwen/Qwen2.5-3B-Instruct"
#   "meta-llama/Llama-3.2-3B-Instruct"
LLM_MODEL_NAME:     str   = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_NEW_TOKENS: int   = 600   # lower generation length to reduce GPU memory usage
LLM_TEMPERATURE:    float = 0.1    # low = more deterministic ranking output
LLM_LOAD_IN_4BIT:   bool  = True  # set True to save VRAM on smaller GPUs

# Stage 2 input text per paper:
#   "abstract"   → title + abstract only  (~400 tokens/paper, faster)
#   "full_text"  → full paper text        (~7500 tokens/paper, richer context)
# Note: full_text × 10 papers ≈ 75K tokens — fits in 128K context window.
#       Falls back to abstract if full_text is empty for a given paper.
STAGE2_INPUT_TEXT: str = "abstract"   # "abstract" | "full_text"

# ── Semantic Scholar API ──────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API:   str   = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_DELAY: float = 0.5   # seconds between requests (rate limiting)
