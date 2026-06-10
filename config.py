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
INDEX_DIR = DATA_DIR / "index"        # built indexes (one subdir per chunking strategy)
EVAL_DIR = ROOT / "eval"

for _d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
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

# --- OpenAI models -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
# Generation uses the stronger model: gpt-4o-mini spuriously refuses when the
# retrieved context mixes the answer with related-but-partial chunks (e.g. a
# cross-company comparison padded with segment-level tables). gpt-4o is robust to
# that noise. Reranking is just relevance scoring, so the cheap model suffices.
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")

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
