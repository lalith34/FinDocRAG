"""Determinism guards.

These pin the reproducibility properties of the pipeline that are cheap to test
without network/model access: deterministic tie-breaking in fusion and reranking,
and the config invariants for the generation/judge models. Generation runs on
Claude (Opus 4.8), which removed the `seed`/`temperature` sampling knobs the old
OpenAI path leaned on — so reproducibility now rests on the fixed prompt + model
id, and these tests guard that the now-rejected sampling params are not sent.
"""
import inspect
import random

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


# --- Config invariants for the generation/judge models -----------------------
def test_chat_model_matches_default_and_is_known():
    # The headless default (CHAT_MODEL) must equal the UI dropdown default
    # (CHAT_MODELS[0]) so both paths answer with the same model, and it must be a
    # model we actually offer.
    assert config.CHAT_MODEL == config.CHAT_MODELS[0]
    assert config.CHAT_MODEL in set(config.CHAT_MODELS)


def test_judge_model_differs_from_generation_model():
    # The judge runs on a different Claude than generation: a separate per-model
    # rate-limit bucket (no contention) and a judge that isn't grading itself.
    assert config.JUDGE_MODEL != config.CHAT_MODEL


def test_generation_sends_no_rejected_sampling_params():
    # Opus 4.8 returns 400 if temperature/top_p/top_k/seed are sent; guard against
    # any of them creeping back into the generation call.
    from src import generate

    # Strip comments so the explanatory note ("rejects temperature/seed") doesn't
    # trip the check; we only care about actual keyword arguments (`param=`).
    src = inspect.getsource(generate._chat)
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for banned in ("temperature=", "top_p=", "top_k=", "seed="):
        assert banned not in code, f"generate._chat must not send {banned} (400 on Opus 4.8)"
