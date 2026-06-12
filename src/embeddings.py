"""OpenAI embeddings with batching and a small on-disk cache.

The cache keys on (model, text) so re-running the build — or embedding the same
sentence during both semantic chunking and indexing — never pays twice.
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import tiktoken

import config

from .reliability import make_openai_client, openai_retry

_ENC = tiktoken.get_encoding("cl100k_base")
_CACHE_PATH = config.DATA_DIR / "embed_cache.pkl"

# Batch guards: stay well under OpenAI's per-request input + token limits.
_MAX_BATCH = 128
_MAX_BATCH_TOKENS = 250_000

_client = None
_cache: dict[str, np.ndarray] | None = None


def _get_client():
    global _client
    if _client is None:
        _client = make_openai_client()
    return _client


def _load_cache() -> dict[str, np.ndarray]:
    global _cache
    if _cache is None:
        if _CACHE_PATH.exists():
            try:
                _cache = pickle.loads(_CACHE_PATH.read_bytes())
            except Exception:  # noqa: BLE001 - corrupt cache shouldn't be fatal
                _cache = {}
        else:
            _cache = {}
    return _cache


def save_cache() -> None:
    if _cache is not None:
        _CACHE_PATH.write_bytes(pickle.dumps(_cache))


def _key(text: str) -> str:
    h = hashlib.sha1(f"{config.EMBED_MODEL}::{text}".encode("utf-8")).hexdigest()
    return h


@openai_retry
def _embed_batch(texts: list[str]) -> list[np.ndarray]:
    resp = _get_client().embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]


def embed_texts(texts: list[str], *, use_cache: bool = True) -> np.ndarray:
    """Embed a list of strings, returning an (N, dim) float32 matrix."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    cache = _load_cache() if use_cache else {}
    results: list[np.ndarray | None] = [None] * len(texts)
    pending: list[int] = []
    for i, t in enumerate(texts):
        if use_cache and (vec := cache.get(_key(t))) is not None:
            results[i] = vec
        else:
            pending.append(i)

    # Build token-aware batches over the uncached items.
    batch: list[int] = []
    batch_tokens = 0
    flushed = 0

    def flush():
        nonlocal batch, batch_tokens, flushed
        if not batch:
            return
        vecs = _embed_batch([texts[j] for j in batch])
        for j, vec in zip(batch, vecs):
            results[j] = vec
            if use_cache:
                cache[_key(texts[j])] = vec
        flushed += len(batch)
        print(f"    embedded {flushed}/{len(pending)} new", end="\r", flush=True)
        batch, batch_tokens = [], 0

    for i in pending:
        t_tok = len(_ENC.encode(texts[i]))
        if batch and (len(batch) >= _MAX_BATCH or batch_tokens + t_tok > _MAX_BATCH_TOKENS):
            flush()
        batch.append(i)
        batch_tokens += t_tok
    flush()
    if pending:
        print()  # newline after progress
    if use_cache:
        save_cache()

    return np.vstack([r for r in results])


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
