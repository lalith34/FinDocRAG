"""Central configuration for the Financial Document Intelligence RAG pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # raw filing HTML
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned plain text
INDEX_DIR = DATA_DIR / "index"        # local chunk snapshots (one subdir per chunking strategy)
EVAL_DIR = ROOT / "eval"
LOGS_DIR = ROOT / "logs"
QUERY_LOG = LOGS_DIR / "queries.jsonl"
FEEDBACK_LOG = LOGS_DIR / "feedback.jsonl"

for _d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Corpus ------------------------------------------------------------------
# Ticker -> human readable company name. CIKs are resolved at ingest time.
COMPANIES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
}

# --- SEC / EDGAR -------------------------------------------------------------
# SEC requires a descriptive User-Agent with contact info.
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "GenAcademy Research your-email@example.com"
)
FILING_FORM = "10-K"

# --- Embeddings (OpenAI) -----------------------------------------------------
# Embeddings stay on OpenAI: Anthropic has no embeddings endpoint, and retrieval
# is independent of the generation provider. text-embedding-3-small is a single
# stable model (OpenAI does not rotate dated snapshots for it), so the name itself
# pins the version. The on-disk embed cache keys on (model, text), so once warm,
# embeddings are fully reproducible.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
EMBED_DIM = 1536

# --- Generation + judge (Anthropic or OpenAI) --------------------------------
# Generation runs on whichever provider's key is configured; the model is chosen
# per query (UI dropdown / CLI flag). The LLM judge runs on a SEPARATE, lighter
# Claude model so it (a) draws from a different per-model rate-limit bucket
# instead of competing with generation, and (b) is a different model from the one
# it grades (cleaner eval methodology).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-opus-4-8")
JUDGE_MODEL = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-sonnet-4-6")
# Cap on generated answer length. Comparison answers (5 companies) are the longest
# real case and fit well under this; staying <16K keeps non-streaming calls under
# the SDK's HTTP-timeout guard.
GEN_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))

# Chat models the UI/CLI can answer with, grouped by the provider that serves
# them. The dropdown is built from these at runtime, filtered to the providers
# whose API key is actually set (see available_chat_models). Override either list
# from the environment (comma-separated) to add/remove models without code edits.
ANTHROPIC_CHAT_MODELS = [
    m.strip() for m in os.getenv(
        "ANTHROPIC_CHAT_MODELS",
        "claude-opus-4-8,claude-opus-4-7,claude-sonnet-4-6,claude-haiku-4-5",
    ).split(",") if m.strip()
]
OPENAI_CHAT_MODELS = [
    m.strip() for m in os.getenv(
        "OPENAI_CHAT_MODELS",
        "gpt-4o,gpt-4o-mini,gpt-4.1,gpt-4.1-mini",
    ).split(",") if m.strip()
]
# Note: Opus 4.8/4.7 reject temperature/top_p/seed (they 400), so the Anthropic
# arm is no longer pinned by a sampling seed; reproducibility there rests on the
# fixed prompt + model id. The OpenAI arm pins temperature=0 for stability.


def model_provider(model: str) -> str:
    """Return 'anthropic' or 'openai' for a chat model id. Falls back to a name
    heuristic so models added via the *_CHAT_MODELS env override still route."""
    if model in ANTHROPIC_CHAT_MODELS:
        return "anthropic"
    if model in OPENAI_CHAT_MODELS:
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise ValueError(f"Unknown chat model {model!r}; set its provider via *_CHAT_MODELS")


def available_chat_models() -> list[str]:
    """Chat models offerable right now, given which provider keys are set. Keeps
    the configured default (CHAT_MODEL) first when its provider is available so
    the dropdown default and the headless default agree."""
    models: list[str] = []
    if ANTHROPIC_API_KEY:
        models += ANTHROPIC_CHAT_MODELS
    if OPENAI_API_KEY:
        models += OPENAI_CHAT_MODELS
    if CHAT_MODEL in models:
        models = [CHAT_MODEL] + [m for m in models if m != CHAT_MODEL]
    return models


# $/1M tokens (input, output) for query-cost estimation in telemetry. Approximate
# — used only for the displayed cost estimate; unknown models fall back to (0, 0).
MODEL_PRICES = {
    # Anthropic
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    # Embeddings
    "text-embedding-3-small": (0.02, 0.0),
}

# --- Pinecone ------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fintech-10k")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_BATCH_SIZE = 100

# --- Reranker ------------------------------------------------------------------
# Local ONNX cross-encoder (fastembed); no torch, works on Intel macOS.
RERANK_ONNX_MODEL = os.getenv("RERANK_ONNX_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# --- Chunking ----------------------------------------------------------------
# Fixed-size strategy (token based). 800 tokens pairs well with a 1536-dim model.
FIXED_CHUNK_TOKENS = 800
FIXED_CHUNK_OVERLAP = 120

# Semantic strategy: group sentences, split where adjacent similarity drops below
# the Nth percentile of all adjacent gaps. Soft cap keeps chunks embeddable.
SEMANTIC_BREAKPOINT_PERCENTILE = 90
SEMANTIC_MAX_TOKENS = 1000
SEMANTIC_MIN_TOKENS = 120

STRATEGIES = ("fixed", "semantic")

# --- Retrieval ---------------------------------------------------------------
TOP_K = 5            # final chunks handed to the generator
RETRIEVE_K = 20      # candidates pulled before reranking
RRF_K = 60           # reciprocal-rank-fusion constant for hybrid

# --- Generation --------------------------------------------------------------
# Phrase the model must emit when retrieval does not support an answer.
REFUSAL_TEXT = (
    "I could not find this in the filings I have indexed."
)
