"""Retrieval-relevance labelling shared by the eval harness and unit tests.

A retrieved chunk counts as relevant if it comes from an expected ticker AND
contains at least one expected keyword.
"""
from __future__ import annotations

from .chunking import Chunk


def is_relevant(chunk: Chunk, query: dict) -> bool:
    if chunk.ticker not in query["expected_tickers"]:
        return False
    text = chunk.text.lower()
    return any(kw.lower() in text for kw in query["must_contain"])


def rank_metrics(chunks: list[Chunk], query: dict, k: int) -> dict:
    top = chunks[:k]
    rels = [is_relevant(c, query) for c in top]
    hit = 1.0 if any(rels) else 0.0
    mrr = 0.0
    for rank, r in enumerate(rels):
        if r:
            mrr = 1.0 / (rank + 1)
            break
    precision = sum(rels) / k
    return {"hit": hit, "mrr": mrr, "precision": precision}
