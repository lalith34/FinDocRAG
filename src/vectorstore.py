"""A lightweight, dependency-free vector store with dense, sparse, and hybrid
retrieval.

Dense  : cosine similarity over OpenAI embeddings (vectors stored L2-normalized,
         so a dot product is the cosine).
Sparse : BM25 over whitespace-tokenized chunk text (catches exact matches that
         dense retrieval misses — ticker symbols, line-item names, error codes).
Hybrid : Reciprocal Rank Fusion of the two ranked lists.

One store is built per chunking strategy and persisted under data/index/<strategy>/.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict

import numpy as np
from rank_bm25 import BM25Okapi

import config
from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return mat / norms


class VectorStore:
    def __init__(self, strategy: str, chunks: list[Chunk], embeddings: np.ndarray):
        self.strategy = strategy
        self.chunks = chunks
        self.embeddings = _normalize(embeddings.astype(np.float32))
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    # --- persistence ---------------------------------------------------------
    @property
    def _dir(self):
        d = config.INDEX_DIR / self.strategy
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self) -> None:
        np.save(self._dir / "embeddings.npy", self.embeddings)
        (self._dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, strategy: str) -> "VectorStore":
        d = config.INDEX_DIR / strategy
        emb = np.load(d / "embeddings.npy")
        raw = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw]
        obj = cls.__new__(cls)
        obj.strategy = strategy
        obj.chunks = chunks
        obj.embeddings = emb  # already normalized at build time
        obj._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        return obj

    # --- retrieval -----------------------------------------------------------
    def dense(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = query_vec.astype(np.float32)
        q = q / (np.linalg.norm(q) or 1e-9)
        scores = self.embeddings @ q
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def hybrid(
        self, query: str, query_vec: np.ndarray, k: int, *, rrf_k: int = config.RRF_K
    ) -> list[tuple[int, float]]:
        pool = max(k, config.RETRIEVE_K)
        dense_rank = [i for i, _ in self.dense(query_vec, pool)]
        sparse_rank = [i for i, _ in self.sparse(query, pool)]
        fused: dict[int, float] = {}
        for rank, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, i in enumerate(sparse_rank):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(int(i), float(s)) for i, s in ordered]

    def get(self, idx: int) -> Chunk:
        return self.chunks[idx]
