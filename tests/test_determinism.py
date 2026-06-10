"""Determinism guards.

These pin the reproducibility properties of the pipeline that are cheap to test
without network/model access: deterministic tie-breaking in fusion and reranking,
and the config knobs (pinned model snapshot + fixed generation seed) that keep
generation reproducible over time. The one stage we cannot make bit-deterministic
in a unit test is the OpenAI generation call itself; those guarantees are
documented and audited via the logged `system_fingerprint`.
"""
import random
import re

import config
from src.chunking import Chunk
from src.vectorstore import rrf_fuse


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        ticker="AAPL",
        company="Apple Inc.",
        source_url="http://example/10-k",
        strategy="fixed",
        text=f"text for {cid}",
        token_count=10,
        position=0,
    )


# --- RRF fusion --------------------------------------------------------------
def test_rrf_ties_break_by_chunk_id():
    # Two single-item lists with disjoint ids => both get identical fused scores
    # (rank 0 in their own list). Tie must resolve by chunk_id ascending.
    fused = rrf_fuse([["b"], ["a"]], 2)
    assert [cid for cid, _ in fused] == ["a", "b"]


def test_rrf_is_order_independent_for_equal_scores():
    # Every id appears once at rank 0 across distinct lists => all tied. The
    # output order must not depend on the order the lists are presented in.
    ids = ["d", "a", "c", "b"]
    runs = []
    for _ in range(5):
        shuffled = ids[:]
        random.shuffle(shuffled)
        fused = rrf_fuse([[cid] for cid in shuffled], len(ids))
        runs.append([cid for cid, _ in fused])
    assert all(r == sorted(ids) for r in runs)


def test_rrf_score_still_dominates_tiebreak():
    # "a" is top of both lists (higher fused score) and must outrank "z" even
    # though "z" sorts earlier than nothing — score beats the id tie-break.
    fused = rrf_fuse([["a", "z"], ["a", "z"]], 2)
    assert fused[0][0] == "a"


# --- Reranker tie-break (no real model needed) -------------------------------
def test_rerank_ties_break_by_chunk_id(monkeypatch):
    from src import rerank

    class _ConstModel:
        def rerank(self, query, texts):
            return [1.0] * len(texts)  # every candidate scores identically

    monkeypatch.setattr(rerank, "_get_model", lambda: _ConstModel())
    candidates = [_chunk(c) for c in ["c2", "c0", "c1"]]
    out = rerank.rerank("q", candidates, top_k=3)
    assert [c.chunk_id for c, _ in out] == ["c0", "c1", "c2"]


# --- Config knobs that keep generation reproducible over time ----------------
def test_chat_model_is_a_pinned_dated_snapshot():
    # A floating alias (e.g. "gpt-4o") silently rolls to a new model; require a
    # dated snapshot so the same query keeps returning the same answer.
    assert re.search(r"-\d{4}-\d{2}-\d{2}$", config.CHAT_MODEL), (
        f"CHAT_MODEL={config.CHAT_MODEL!r} is not a dated snapshot"
    )


def test_generation_seed_is_set():
    assert isinstance(config.GEN_SEED, int)
