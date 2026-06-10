"""Reranking step: take the candidate pool from hybrid retrieval and reorder it
by true relevance to the query.

We use an LLM listwise reranker (one call per query): the model scores every
candidate 0-10 and we sort by that score. This is provider-native (no extra
model download) and the eval measures the lift it gives over raw retrieval.
"""
from __future__ import annotations

import json

import config
from .chunking import Chunk

# Show the reranker (almost) the whole chunk: financial figures often sit deep
# inside a flattened income-statement table, so a short snippet hides the very
# number the query is asking about and the chunk gets wrongly demoted.
_SNIPPET_CHARS = 3500

_SYSTEM = (
    "You are a relevance judge for a financial-filings search engine. "
    "Given a user query and numbered candidate passages from SEC 10-K filings, "
    "score how well each passage helps answer the query on a 0-10 scale "
    "(10 = directly answers it, 0 = irrelevant). "
    'Respond ONLY with JSON: {"scores": [{"id": <int>, "score": <number>}, ...]} '
    "covering every candidate id."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def rerank(
    query: str, candidates: list[Chunk], top_k: int = config.TOP_K
) -> list[tuple[Chunk, float]]:
    if not candidates:
        return []

    listing = "\n\n".join(
        f"[{i}] ({c.ticker}) {c.text[:_SNIPPET_CHARS]}"
        for i, c in enumerate(candidates)
    )
    user = f"Query: {query}\n\nCandidates:\n{listing}"

    try:
        resp = _get_client().chat.completions.create(
            model=config.RERANK_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        scores = {int(s["id"]): float(s["score"]) for s in data.get("scores", [])}
    except Exception as e:  # noqa: BLE001 - degrade gracefully to retrieval order
        print(f"[rerank] LLM rerank failed ({e}); falling back to retrieval order")
        scores = {i: float(len(candidates) - i) for i in range(len(candidates))}

    ranked = sorted(
        range(len(candidates)),
        key=lambda i: scores.get(i, -1.0),
        reverse=True,
    )
    return [(candidates[i], scores.get(i, 0.0)) for i in ranked[:top_k]]
