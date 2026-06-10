"""Hybrid retrieval over Pinecone (dense) + local BM25 (sparse).

Dense  : OpenAI embeddings stored in a Pinecone serverless index (cosine), one
         namespace per chunking strategy.
Sparse : BM25 over the local chunks.json snapshot (catches exact matches dense
         retrieval misses — ticker symbols, line-item names).
Hybrid : Reciprocal Rank Fusion of the two ranked lists, keyed on chunk_id.

The per-strategy chunks.json under data/index/<strategy>/ remains the local
source of truth for BM25 and evaluation; vectors live only in Pinecone.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict

import numpy as np
from rank_bm25 import BM25Okapi

import config
from .chunking import Chunk
from .reliability import pinecone_retry

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(
    ranked_id_lists: list[list[str]],
    k: int,
    *,
    rrf_k: int = config.RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of chunk_ids via Reciprocal Rank Fusion.

    `weights` scales each list's contribution (defaults to equal weight). The
    router uses this to upweight BM25 for lexical queries or dense for semantic
    ones without dropping the other signal entirely.
    """
    if weights is None:
        weights = [1.0] * len(ranked_id_lists)
    fused: dict[str, float] = {}
    for w, ranked in zip(weights, ranked_id_lists):
        for rank, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + w / (rrf_k + rank + 1)
    # Tie-break on chunk_id so equal fused scores produce a stable order
    # regardless of dict insertion order or upstream ranking jitter.
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


_pc = None


def _get_pinecone():
    global _pc
    if _pc is None:
        if not config.PINECONE_API_KEY:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from pinecone import Pinecone

        _pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc


def _get_index():
    pc = _get_pinecone()
    name = config.PINECONE_INDEX_NAME
    if not pc.has_index(name):
        from pinecone import ServerlessSpec

        print(f"[pinecone] creating index '{name}' ...")
        pc.create_index(
            name=name,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return pc.Index(name)


class PineconeStore:
    """One store per chunking strategy; namespace == strategy."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index = _get_index()
        self.chunks = self._load_chunks()
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._bm25 = (
            BM25Okapi([_tokenize(c.text) for c in self.chunks]) if self.chunks else None
        )

    # --- local snapshot (BM25 + eval source of truth) -------------------------
    @property
    def _chunks_path(self):
        d = config.INDEX_DIR / self.strategy
        d.mkdir(parents=True, exist_ok=True)
        return d / "chunks.json"

    def _load_chunks(self) -> list[Chunk]:
        if not self._chunks_path.exists():
            return []
        raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        return [Chunk(**c) for c in raw]

    def save_chunks(self, chunks: list[Chunk]) -> None:
        chunks = sorted(chunks, key=lambda c: c.chunk_id)
        self._chunks_path.write_text(
            json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8"
        )
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    # --- write path ------------------------------------------------------------
    @pinecone_retry
    def _upsert_batch(self, vectors: list[dict]) -> None:
        self.index.upsert(vectors=vectors, namespace=self.strategy)

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        vectors = [
            {
                "id": c.chunk_id,
                "values": embeddings[i].astype(np.float32).tolist(),
                "metadata": {
                    "text": c.text,
                    "ticker": c.ticker,
                    "company": c.company,
                    "source_url": c.source_url,
                    "strategy": c.strategy,
                    "position": c.position,
                    "token_count": c.token_count,
                },
            }
            for i, c in enumerate(chunks)
        ]
        for start in range(0, len(vectors), config.PINECONE_BATCH_SIZE):
            self._upsert_batch(vectors[start : start + config.PINECONE_BATCH_SIZE])
            print(
                f"    upserted {min(start + config.PINECONE_BATCH_SIZE, len(vectors))}"
                f"/{len(vectors)}",
                end="\r",
                flush=True,
            )
        if vectors:
            print()

    @pinecone_retry
    def delete_ticker(self, ticker: str) -> None:
        # Serverless indexes have no delete-by-metadata-filter; chunk_ids are
        # "<TICKER>-<strategy>-NNNN", so id-prefix listing covers it.
        for ids in self.index.list(prefix=f"{ticker}-", namespace=self.strategy):
            # Newer pinecone clients yield ListItem objects, not bare strings;
            # the delete endpoint needs JSON-serializable string ids.
            ids = [getattr(x, "id", x) for x in ids]
            if ids:
                self.index.delete(ids=ids, namespace=self.strategy)

    # --- retrieval ---------------------------------------------------------------
    def _chunk_from_match(self, match) -> Chunk:
        cached = self._by_id.get(match.id)
        if cached is not None:
            return cached
        m = match.metadata or {}
        return Chunk(
            chunk_id=match.id,
            ticker=m.get("ticker", ""),
            company=m.get("company", ""),
            source_url=m.get("source_url", ""),
            strategy=m.get("strategy", self.strategy),
            text=m.get("text", ""),
            token_count=int(m.get("token_count", 0)),
            position=int(m.get("position", 0)),
        )

    @pinecone_retry
    def dense(
        self, query_vec: np.ndarray, k: int, *, ticker: str | None = None
    ) -> list[tuple[Chunk, float]]:
        res = self.index.query(
            namespace=self.strategy,
            vector=query_vec.astype(np.float32).tolist(),
            top_k=k,
            include_metadata=True,
            filter={"ticker": {"$eq": ticker}} if ticker else None,
        )
        # Pinecone is an approximate index and does not guarantee a stable order
        # for equal scores; re-sort by (score desc, id) for a reproducible ranking.
        matches = sorted(res.matches, key=lambda m: (-float(m.score), m.id))
        return [(self._chunk_from_match(m), float(m.score)) for m in matches]

    def sparse(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # Stable sort so equal BM25 scores keep chunk order; self.chunks is stored
        # sorted by chunk_id, so ties break deterministically by chunk_id.
        idx = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[int(i)], float(scores[i])) for i in idx]

    def hybrid(
        self,
        query: str,
        query_vec: np.ndarray,
        k: int,
        *,
        ticker: str | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[tuple[Chunk, float]]:
        pool = max(k, config.RETRIEVE_K)
        dense_hits = self.dense(query_vec, pool, ticker=ticker)
        sparse_hits = self.sparse(query, pool)
        by_id = {c.chunk_id: c for c, _ in dense_hits + sparse_hits}
        fused = rrf_fuse(
            [[c.chunk_id for c, _ in dense_hits], [c.chunk_id for c, _ in sparse_hits]],
            k,
            weights=[dense_weight, sparse_weight],
        )
        return [(by_id[cid], score) for cid, score in fused]
