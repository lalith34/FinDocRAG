"""Offline smoke test: exercises every code path that does NOT require an API
key — ingestion/cleaning, both chunkers (semantic uses a deterministic fake
embedder), and dense/sparse/hybrid retrieval over the vector store.

Run after ingestion:  python -m scripts.smoke_test
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import ingest  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

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

    # Build a store with fake embeddings and test all three retrieval modes.
    vecs = fake_embed([c.text for c in all_chunks])
    store = VectorStore("fixed", all_chunks, vecs)
    q = "What were total net sales and revenue?"
    qv = fake_embed([q])[0]
    dense = store.dense(qv, 5)
    sparse = store.sparse(q, 5)
    hybrid = store.hybrid(q, qv, 5)
    assert len(dense) == 5 and len(sparse) == 5 and len(hybrid) == 5
    print(f"\nRetrieval OK — dense/sparse/hybrid each returned 5 hits over "
          f"{len(all_chunks)} chunks.")
    print("Top hybrid hit:", store.get(hybrid[0][0]).chunk_id)
    print("\nSMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()
