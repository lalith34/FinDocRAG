"""Offline smoke test: exercises every code path that does NOT require an API
key — ingestion/cleaning, both chunkers (semantic uses a deterministic fake
embedder), BM25 sparse retrieval, and RRF fusion. Pinecone dense retrieval is
covered by the online eval, not here.

Run after ingestion:  python -m scripts.smoke_test
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ingest  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.vectorstore import rrf_fuse, _tokenize  # noqa: E402

DIM = 64


def fake_embed(texts):
    """Deterministic pseudo-embeddings from text hashes — no network."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        out.append(rng.standard_normal(DIM).astype(np.float32))
    return np.vstack(out)


def main():
    docs = ingest.load_processed()
    assert docs, "No processed docs. Run: python -m src.ingest"
    print(f"Loaded {len(docs)} processed filings: {', '.join(docs)}")

    all_chunks = []
    for ticker, payload in docs.items():
        meta = payload["meta"]
        text = payload["text"]
        fixed = chunk_document(
            ticker=ticker, company=meta["company"], source_url=meta["source_url"],
            text=text, strategy="fixed",
        )
        semantic = chunk_document(
            ticker=ticker, company=meta["company"], source_url=meta["source_url"],
            text=text, strategy="semantic", embed_fn=fake_embed,
        )
        print(f"  {ticker}: fixed={len(fixed):4d} chunks  semantic={len(semantic):4d} chunks  "
              f"(text {len(text):,} chars)")
        assert fixed and semantic, f"{ticker} produced no chunks"
        all_chunks.extend(fixed)

    # BM25 sparse retrieval over the chunks, no network.
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokenize(c.text) for c in all_chunks])
    q = "What were total net sales and revenue?"
    scores = bm25.get_scores(_tokenize(q))
    sparse_ids = [all_chunks[int(i)].chunk_id for i in np.argsort(-scores)[:5]]
    assert len(sparse_ids) == 5

    # RRF fusion is a pure function — fuse sparse with a shuffled copy.
    fused = rrf_fuse([sparse_ids, list(reversed(sparse_ids))], 5)
    assert len(fused) == 5 and fused[0][1] > 0
    print(f"\nRetrieval OK — BM25 returned 5 hits over {len(all_chunks)} chunks; "
          f"RRF fused cleanly.")
    print("Top sparse hit:", sparse_ids[0])
    print("\nSMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()
