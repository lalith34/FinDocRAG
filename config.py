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

# --- OpenAI models -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# text-embedding-3-small is a single stable model (OpenAI does not rotate dated
# snapshots for it), so the name itself pins the version. The on-disk embed cache
# keys on (model, text), so once warm, embeddings are fully reproducible.
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
EMBED_DIM = 1536
# Generation uses the stronger model: gpt-4o-mini spuriously refuses when the
# retrieved context mixes the answer with related-but-partial chunks (e.g. a
# cross-company comparison padded with segment-level tables). gpt-4o is robust to
# that noise.
#
# Pin a DATED snapshot, not the floating "gpt-4o" alias: the alias silently rolls
# to a new underlying model every few months, which changes answers for the same
# query/context. A dated snapshot is the only way to keep generation reproducible
# over time.
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-2024-08-06")
# Best-effort determinism knob for chat completions. Combined with temperature=0
# and a pinned snapshot, the same (seed, model, prompt) returns the same output
# as long as system_fingerprint (logged per query) is unchanged.
GEN_SEED = int(os.getenv("OPENAI_GEN_SEED", "7"))

# $/1M tokens (input, output) for query-cost estimation in telemetry.
MODEL_PRICES = {
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
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

# --- App -----------------------------------------------------------------------
# Shared password for the Streamlit UI. Unset = open access (dev mode).
APP_PASSWORD = os.getenv("APP_PASSWORD")

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
