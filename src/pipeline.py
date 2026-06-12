"""Orchestration: build the indexes and answer queries end-to-end.

Build:  ingest (incremental by EDGAR accession) -> chunk changed tickers ->
        embed -> upsert to Pinecone -> refresh local chunks.json snapshot.
Query:  a compiled LangGraph StateGraph (src/graph.py): input guard -> route ->
        hybrid retrieve (Pinecone dense + local BM25) | per-company quota |
        PageIndex -> cross-encoder rerank -> cited generation -> citation audit,
        with a per-query telemetry trace assembled from the final graph state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from . import embeddings, ingest, pageindex, telemetry
from .chunking import Chunk, chunk_document
from .generate import Answer
from .graph import build_query_graph
from .router import CORPUS, SMALLTALK, is_smalltalk, mentioned_tickers
from .vectorstore import PineconeStore

__all__ = ["RAGPipeline", "RAGResult", "build_indexes", "is_smalltalk", "mentioned_tickers"]


def _smalltalk_reply() -> str:
    names = ", ".join(config.COMPANIES)
    return (
        "Hi! I'm a financial-filings assistant. I answer questions grounded in "
        f"the latest SEC 10-K filings for **{names}**, with citations.\n\n"
        "Try asking things like:\n"
        "- *What were Apple's total net sales last year?*\n"
        "- *How much did Microsoft spend on research and development?*\n"
        "- *What does NVIDIA report about its Data Center business?*\n"
        "- *What risk factors does Amazon disclose?*"
    )


def _corpus_reply() -> str:
    """Deterministic listing of every indexed filing, built from the corpus
    metadata rather than retrieval — so a "what is in the corpus?" question
    always reports all companies, not just the few nearest to the query."""
    docs = ingest.load_processed()
    if docs:
        rows = [
            f"- **{docs[t]['meta']['company']} ({t})** — "
            f"{docs[t]['meta']['form']} filed {docs[t]['meta']['filing_date']}"
            for t in sorted(docs)
        ]
        n = len(docs)
    else:  # index not built yet; fall back to the configured corpus
        rows = [f"- **{name} ({tk})**" for tk, name in config.COMPANIES.items()]
        n = len(config.COMPANIES)
    body = "\n".join(rows)
    return (
        f"I have the latest SEC 10-K filings for these {n} companies indexed:\n\n"
        f"{body}\n\n"
        "Ask me anything about their financials, segments, risks, or disclosures."
    )


# --- Build -------------------------------------------------------------------
def build_indexes(
    strategies=config.STRATEGIES,
    *,
    do_ingest: bool = True,
    force: bool = False,
    refresh_raw: bool = False,
    tickers: list[str] | None = None,
) -> dict:
    """Build/refresh the indexes. When `tickers` is given, ingest and rebuild only
    those companies (the rest of the corpus is left in place) — this is the fast
    path for adding a single company at runtime."""
    if tickers is not None:
        tickers = [t.upper() for t in tickers]
    if do_ingest:
        subset = {t: config.COMPANIES[t] for t in tickers} if tickers else None
        ingest_results = ingest.ingest_all(
            companies=subset, force=force, refresh_raw=refresh_raw
        )
        changed_tickers = {t for t, (_, changed) in ingest_results.items() if changed}
    else:
        changed_tickers = set(tickers) if tickers else set(config.COMPANIES)

    docs = ingest.load_processed()
    if not docs:
        raise RuntimeError("No processed documents found. Run ingestion first.")

    summary = {}
    for strategy in strategies:
        print(f"\n=== Building '{strategy}' index ===")
        store = PineconeStore(strategy)

        # First build (no local snapshot) or --force: rebuild everything.
        rebuild = set(changed_tickers)
        if force or not store.chunks:
            rebuild = set(docs)

        kept = [c for c in store.chunks if c.ticker not in rebuild]
        new_chunks: list[Chunk] = []
        for ticker in sorted(rebuild):
            payload = docs[ticker]
            meta = payload["meta"]
            chunks = chunk_document(
                ticker=ticker,
                company=meta["company"],
                source_url=meta["source_url"],
                text=payload["text"],
                strategy=strategy,
                embed_fn=embeddings.embed_texts if strategy == "semantic" else None,
            )
            print(f"  {ticker}: {len(chunks)} chunks")
            new_chunks.extend(chunks)

        if new_chunks:
            print(f"  embedding {len(new_chunks)} chunks ...")
            vecs = embeddings.embed_texts([c.text for c in new_chunks])
            for ticker in sorted(rebuild):
                store.delete_ticker(ticker)
            store.upsert(new_chunks, vecs)
        else:
            print("  no tickers changed; index up to date")

        all_chunks = kept + new_chunks
        store.save_chunks(all_chunks)
        summary[strategy] = {
            "chunks": len(all_chunks),
            "rebuilt_tickers": sorted(rebuild) if new_chunks else [],
            "avg_tokens": round(
                sum(c.token_count for c in all_chunks) / max(1, len(all_chunks)), 1
            ),
        }
        print(f"  snapshot -> data/index/{strategy}/chunks.json")
    return summary


# --- Query -------------------------------------------------------------------
@dataclass
class RAGResult:
    query: str
    answer: Answer
    retrieved: list[Chunk]
    strategy: str
    reranked: bool
    route: str = ""
    retriever: str = "vector"
    nav_path: str = ""  # PageIndex reasoning path, when that retriever was used
    guardrail: str = ""  # input guard that fired ("" if none): injection|advice|length
    dangling_citations: list[int] = field(default_factory=list)
    trace: telemetry.QueryTrace | None = field(default=None)


class RAGPipeline:
    def __init__(self, strategy: str = "semantic"):
        self.strategy = strategy
        self.store = PineconeStore(strategy)
        self._pageindex: pageindex.PageIndexRetriever | None = None
        self._graph = None  # compiled LangGraph, built lazily

    def graph(self):
        """Lazily compile the query StateGraph (src/graph.py) over this
        pipeline's retrievers and the canned-reply builders."""
        if self._graph is None:
            self._graph = build_query_graph(
                self, smalltalk_reply=_smalltalk_reply, corpus_reply=_corpus_reply
            )
        return self._graph

    def pageindex(self) -> pageindex.PageIndexRetriever:
        """Lazily construct the tree-navigation retriever (builds a ticker's tree
        on first use and memoises it for the process)."""
        if self._pageindex is None:
            self._pageindex = pageindex.PageIndexRetriever()
        return self._pageindex

    def retrieve(
        self,
        query: str,
        k: int = config.RETRIEVE_K,
        *,
        ticker: str | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[Chunk]:
        qvec = embeddings.embed_query(query)
        return [
            c
            for c, _ in self.store.hybrid(
                query,
                qvec,
                k,
                ticker=ticker,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        ]

    def retrieve_per_ticker(
        self, query: str, tickers: list[str], *, per_company: int
    ) -> list[Chunk]:
        """Retrieve a per-company quota so every named company is represented in
        the final context. A flat top-k lets the best-matching company
        monopolise the slots, which is why cross-company comparisons used to
        drop companies and refuse.

        Uses hybrid (dense + BM25) retrieval per company and deliberately skips
        the cross-encoder: on a multi-company query ("AAPL NVDA MSFT ... sales")
        the reranker scores every chunk as irrelevant and buries each company's
        income-statement table below XBRL-tag noise. The BM25 arm is what lifts
        each company's income statement to rank ~1 even when its line-item label
        differs from the query wording (Apple reports "Total net sales", Alphabet
        "Total revenues" against a "total revenue" question); dense-only
        similarity missed those rows and was silently dropping AAPL and GOOG from
        5-ticker rankings."""
        qvec = embeddings.embed_query(query)
        out: list[Chunk] = []
        for tk in tickers:
            out.extend(
                c for c, _ in self.store.hybrid(query, qvec, per_company, ticker=tk)
            )
        return out

    def answer(
        self,
        query: str,
        *,
        use_rerank: bool = True,
        top_k: int = config.TOP_K,
        model: str | None = None,
        retriever: str = "vector",
    ) -> RAGResult:
        model = model or config.CHAT_MODEL

        # Run the compiled LangGraph: input guard -> route -> retrieve (vector /
        # comparison quota / PageIndex) -> rerank -> generate -> citation audit.
        # The graph returns its final state; everything below just assembles the
        # RAGResult + telemetry trace from it.
        state = self.graph().invoke(
            {
                "query": query,
                "use_rerank": use_rerank,
                "top_k": top_k,
                "model": model,
                "retriever": retriever,
            }
        )

        # Input guard fired: safe completion, logged with the guard that tripped,
        # no retrieval or model call happened.
        if state["guardrail"]:
            route = f"guardrail/{state['guardrail']}"
            trace = telemetry.QueryTrace(
                query=query, strategy=self.strategy, reranked=False, refused=False,
                route=route, guardrail=state["guardrail"], model=model,
            )
            telemetry.log_query(trace)
            return RAGResult(
                query=query,
                answer=state["answer"],
                retrieved=[],
                strategy=self.strategy,
                reranked=False,
                route=route,
                retriever=retriever,
                guardrail=state["guardrail"],
                trace=trace,
            )

        # Smalltalk / corpus meta-question: canned deterministic reply, no trace.
        if state["route_kind"] in (SMALLTALK, CORPUS):
            return RAGResult(
                query=query,
                answer=state["answer"],
                retrieved=[],
                strategy=self.strategy,
                reranked=False,
                route=state["route_kind"],
                retriever=retriever,
            )

        ans: Answer = state["answer"]
        audit = state["audit"]
        top = state["top"]
        t0, t1, t2, t3 = state["t0"], state["t1"], state["t2"], state["t3"]
        trace = telemetry.QueryTrace(
            query=query,
            strategy=self.strategy,
            reranked=state["reranked"],
            refused=ans.refused,
            route=f"{state['route_kind']}/{state['arm']}",
            dangling_citations=audit.dangling,
            candidates=[{"chunk_id": c.chunk_id} for c in top],
            retrieval_ms=round((t1 - t0) * 1000, 1),
            rerank_ms=round((t2 - t1) * 1000, 1),
            generation_ms=round((t3 - t2) * 1000, 1),
            total_ms=round((t3 - t0) * 1000, 1),
            prompt_tokens=ans.usage.get("prompt_tokens", 0),
            completion_tokens=ans.usage.get("completion_tokens", 0),
            est_cost_usd=round(
                telemetry.estimate_cost(
                    model,
                    ans.usage.get("prompt_tokens", 0),
                    ans.usage.get("completion_tokens", 0),
                ),
                6,
            ),
            model=model,
            system_fingerprint=ans.usage.get("system_fingerprint"),
        )
        telemetry.log_query(trace)

        return RAGResult(
            query=query,
            answer=ans,
            retrieved=top,
            strategy=self.strategy,
            reranked=state["reranked"],
            route=state["route_kind"],
            retriever=state["arm"],
            nav_path=state.get("nav_path", ""),
            dangling_citations=audit.dangling,
            trace=trace,
        )
