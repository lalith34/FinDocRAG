"""LangGraph orchestration of the query path.

The per-query flow is a compiled LangGraph StateGraph instead of a hand-rolled
if/else chain. Each pipeline stage is a node; the routing decisions that used to
be Python control flow are conditional edges, so the topology *is* the
documentation:

                          ┌─ blocked ──────────────────────────► END
    input_guard ─► route ─┼─ smalltalk/corpus ─► canned_reply ──► END
                          ├─ comparison ─► retrieve_comparison ─┐
                          ├─ pageindex  ─► retrieve_pageindex ──┼─► generate
                          └─ default ─► retrieve ─► rerank ─────┘      │
                                            └──(rerank off)────────────┤
                                                                       ▼
                                                          audit_output ─► END

The nodes close over a RAGPipeline instance for retrieval (Pinecone hybrid /
per-ticker quota / PageIndex) and call the same domain functions as before —
guardrails, router, rerank, generate. State is a plain TypedDict; every node
returns only the keys it produced, and LangGraph merges them.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

import config
from . import guardrails, router
from .generate import Answer, generate
from .rerank import rerank


class QueryState(TypedDict, total=False):
    # inputs
    query: str
    use_rerank: bool
    top_k: int
    model: str
    retriever: str            # requested arm: "vector" | "pageindex"
    # input guard
    guardrail: str            # guard that fired ("" if none)
    # router
    route_kind: str
    tickers: list[str]
    dense_weight: float
    sparse_weight: float
    # retrieval
    top: list[Any]            # list[Chunk] handed to the generator
    reranked: bool
    arm: str                  # arm actually used: "vector" | "pageindex"
    nav_path: str
    # generation + audit
    answer: Any               # Answer
    audit: Any                # CitationAudit | None
    # timings (perf_counter marks; the pipeline turns them into ms)
    t0: float
    t1: float
    t2: float
    t3: float


def build_query_graph(pipe, *, smalltalk_reply, corpus_reply):
    """Compile the query StateGraph around a RAGPipeline (for its retrievers) and
    the canned-reply builders (passed in to avoid a circular import)."""

    def input_guard(state: QueryState) -> QueryState:
        guard = guardrails.check_input(state["query"])
        if not guard.allowed:
            return {
                "guardrail": guard.reason,
                "answer": Answer(text=guard.reply, sources=[], refused=False),
            }
        return {"guardrail": ""}

    def route(state: QueryState) -> QueryState:
        r = router.route(state["query"])
        return {
            "route_kind": r.kind,
            "tickers": list(r.tickers),
            "dense_weight": r.dense_weight,
            "sparse_weight": r.sparse_weight,
            "t0": time.perf_counter(),
        }

    def canned_reply(state: QueryState) -> QueryState:
        text = (
            corpus_reply() if state["route_kind"] == router.CORPUS else smalltalk_reply()
        )
        return {"answer": Answer(text=text, sources=[], refused=False)}

    def retrieve(state: QueryState) -> QueryState:
        # LEXICAL / SEMANTIC / HYBRID all run weighted hybrid retrieval; the
        # router only shifts the dense/BM25 fusion weights. A single named
        # company scopes retrieval to that ticker.
        tickers = state["tickers"]
        candidates = pipe.retrieve(
            state["query"],
            config.RETRIEVE_K,
            ticker=tickers[0] if len(tickers) == 1 else None,
            dense_weight=state["dense_weight"],
            sparse_weight=state["sparse_weight"],
        )
        return {"top": candidates, "arm": "vector", "t1": time.perf_counter()}

    def retrieve_comparison(state: QueryState) -> QueryState:
        # Per-company quota so one company can't monopolise the context; the
        # cross-encoder is skipped (see RAGPipeline.retrieve_per_ticker). Floor
        # of 5 per company: an income-statement table can sit at rank ~4 behind
        # residual XBRL noise.
        tickers = state["tickers"]
        per_company = max(5, -(-state["top_k"] // len(tickers)))
        top = pipe.retrieve_per_ticker(
            state["query"], list(tickers), per_company=per_company
        )
        now = time.perf_counter()
        return {"top": top, "reranked": False, "arm": "vector", "t1": now, "t2": now}

    def retrieve_pageindex(state: QueryState) -> QueryState:
        top, nav_path, _reasoning = pipe.pageindex().retrieve(
            state["query"], state["tickers"][0], state["top_k"]
        )
        now = time.perf_counter()
        return {
            "top": top,
            "reranked": False,
            "arm": "pageindex",
            "nav_path": nav_path,
            "t1": now,
            "t2": now,
        }

    def rerank_top(state: QueryState) -> QueryState:
        top = [c for c, _ in rerank(state["query"], state["top"], state["top_k"])]
        return {"top": top, "reranked": True, "t2": time.perf_counter()}

    def truncate_top(state: QueryState) -> QueryState:
        # Rerank toggled off: keep the retrieval order, cut to top-k.
        return {
            "top": state["top"][: state["top_k"]],
            "reranked": False,
            "t2": time.perf_counter(),
        }

    def generate_answer(state: QueryState) -> QueryState:
        # Comparison context is grouped per company, not globally ranked, so the
        # lost-in-the-middle reorder is skipped there.
        ans = generate(
            state["query"],
            state["top"],
            model=state["model"],
            reorder=state["route_kind"] != router.COMPARISON,
        )
        return {"answer": ans, "t3": time.perf_counter()}

    def audit_output(state: QueryState) -> QueryState:
        ans = state["answer"]
        audit = guardrails.audit_citations(
            ans.text, len(ans.sources), refused=ans.refused
        )
        if not ans.refused and ans.sources:
            ans.text = guardrails.with_disclaimer(ans.text)
        return {"answer": ans, "audit": audit}

    # --- edges -----------------------------------------------------------------
    def after_guard(state: QueryState) -> str:
        return "blocked" if state["guardrail"] else "ok"

    def after_route(state: QueryState) -> str:
        kind = state["route_kind"]
        if kind in (router.SMALLTALK, router.CORPUS):
            return "canned"
        if kind == router.COMPARISON:
            return "comparison"
        # PageIndex is per-document: only a single-company, non-comparison query
        # can use it; everything else falls back to the vector arm.
        if state["retriever"] == "pageindex" and len(state["tickers"]) == 1:
            return "pageindex"
        return "vector"

    def after_retrieve(state: QueryState) -> str:
        return "rerank" if state["use_rerank"] else "truncate"

    g = StateGraph(QueryState)
    g.add_node("input_guard", input_guard)
    g.add_node("route", route)
    g.add_node("canned_reply", canned_reply)
    g.add_node("retrieve", retrieve)
    g.add_node("retrieve_comparison", retrieve_comparison)
    g.add_node("retrieve_pageindex", retrieve_pageindex)
    g.add_node("rerank_top", rerank_top)
    g.add_node("truncate_top", truncate_top)
    g.add_node("generate_answer", generate_answer)
    g.add_node("audit_output", audit_output)

    g.set_entry_point("input_guard")
    g.add_conditional_edges("input_guard", after_guard, {"blocked": END, "ok": "route"})
    g.add_conditional_edges(
        "route",
        after_route,
        {
            "canned": "canned_reply",
            "comparison": "retrieve_comparison",
            "pageindex": "retrieve_pageindex",
            "vector": "retrieve",
        },
    )
    g.add_edge("canned_reply", END)
    g.add_conditional_edges(
        "retrieve", after_retrieve, {"rerank": "rerank_top", "truncate": "truncate_top"}
    )
    g.add_edge("rerank_top", "generate_answer")
    g.add_edge("truncate_top", "generate_answer")
    g.add_edge("retrieve_comparison", "generate_answer")
    g.add_edge("retrieve_pageindex", "generate_answer")
    g.add_edge("generate_answer", "audit_output")
    g.add_edge("audit_output", END)
    return g.compile()
