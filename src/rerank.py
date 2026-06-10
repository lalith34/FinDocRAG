"""Reranking step: reorder the hybrid-retrieval candidate pool by true relevance
to the query using a local ONNX cross-encoder (fastembed, no torch needed).

Runs fully offline after a one-time ~90MB model download. Falls back to the
original retrieval order if the model cannot be loaded.
"""
from __future__ import annotations

import logging

import config
from .chunking import Chunk

log = logging.getLogger("rerank")

_model = None
_model_failed = False


def _get_model():
    global _model, _model_failed
    if _model is None and not _model_failed:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            _model = TextCrossEncoder(model_name=config.RERANK_ONNX_MODEL)
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            log.warning("cross-encoder load failed (%s); reranking disabled", e)
            _model_failed = True
    return _model


def rerank(
    query: str, candidates: list[Chunk], top_k: int = config.TOP_K
) -> list[tuple[Chunk, float]]:
    if not candidates:
        return []

    model = _get_model()
    if model is None:
        return [(c, float(len(candidates) - i)) for i, c in enumerate(candidates)][:top_k]

    scores = list(model.rerank(query, [c.text for c in candidates]))
    # Sort by score desc, tie-broken by chunk_id, so equal cross-encoder scores
    # produce a stable, reproducible order independent of candidate input order.
    ranked = sorted(
        range(len(candidates)), key=lambda i: (-scores[i], candidates[i].chunk_id)
    )
    return [(candidates[i], float(scores[i])) for i in ranked[:top_k]]
