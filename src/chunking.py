"""Two chunking strategies, compared head-to-head later in the eval:

1. fixed   - token-window chunks with overlap (cheap, deterministic, can split
             mid-thought).
2. semantic- sentence-grouped chunks that break where the topic shifts
             (adjacent-sentence embedding distance crosses a percentile),
             with min/max token guards so chunks stay embeddable.

Both emit a common Chunk record so the rest of the pipeline is strategy-agnostic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import tiktoken

import config

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    company: str
    source_url: str
    strategy: str
    text: str
    token_count: int
    position: int  # ordinal within the document
    meta: dict = field(default_factory=dict)


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# --- Fixed-size --------------------------------------------------------------
def fixed_chunks(
    text: str,
    *,
    size: int = config.FIXED_CHUNK_TOKENS,
    overlap: int = config.FIXED_CHUNK_OVERLAP,
) -> list[str]:
    tokens = _ENC.encode(text)
    if not tokens:
        return []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        out.append(_ENC.decode(window).strip())
        if start + size >= len(tokens):
            break
    return [c for c in out if c]


# --- Semantic ----------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")
# Hard ceiling so no single pseudo-sentence (e.g. a giant flattened table) can
# exceed the embedding model's input limit.
_SENT_MAX_TOKENS = 500


def _hard_split(text: str, max_tokens: int = _SENT_MAX_TOKENS) -> list[str]:
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    return [
        _ENC.decode(toks[i : i + max_tokens]).strip()
        for i in range(0, len(toks), max_tokens)
    ]


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.extend(_hard_split(p))
    return out


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return 1.0 - float(np.dot(a, b) / denom)


def semantic_chunks(
    text: str,
    embed_fn,
    *,
    percentile: int = config.SEMANTIC_BREAKPOINT_PERCENTILE,
    max_tokens: int = config.SEMANTIC_MAX_TOKENS,
    min_tokens: int = config.SEMANTIC_MIN_TOKENS,
) -> list[str]:
    """`embed_fn(list[str]) -> np.ndarray` embeds sentences for breakpoint
    detection."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    vecs = embed_fn(sentences)
    dists = [_cosine_dist(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    threshold = float(np.percentile(dists, percentile)) if dists else 1.0

    chunks: list[str] = []
    cur: list[str] = [sentences[0]]
    cur_tok = n_tokens(sentences[0])
    for i in range(1, len(sentences)):
        sent = sentences[i]
        tok = n_tokens(sent)
        gap = dists[i - 1]
        # Break on a topic shift, but only once the chunk is big enough; force a
        # break if we'd otherwise blow past the max token budget.
        hard_break = cur_tok + tok > max_tokens
        soft_break = gap >= threshold and cur_tok >= min_tokens
        if hard_break or soft_break:
            chunks.append(" ".join(cur).strip())
            cur, cur_tok = [sent], tok
        else:
            cur.append(sent)
            cur_tok += tok
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]


# --- Driver ------------------------------------------------------------------
def chunk_document(
    *,
    ticker: str,
    company: str,
    source_url: str,
    text: str,
    strategy: str,
    embed_fn=None,
) -> list[Chunk]:
    if strategy == "fixed":
        pieces = fixed_chunks(text)
    elif strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires an embed_fn")
        pieces = semantic_chunks(text, embed_fn)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [
        Chunk(
            chunk_id=f"{ticker}-{strategy}-{i:04d}",
            ticker=ticker,
            company=company,
            source_url=source_url,
            strategy=strategy,
            text=piece,
            token_count=n_tokens(piece),
            position=i,
        )
        for i, piece in enumerate(pieces)
    ]
