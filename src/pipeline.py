"""Orchestration: build the indexes and answer queries end-to-end.

Build:  ingest (incremental by EDGAR accession) -> chunk changed tickers ->
        embed -> upsert to Pinecone -> refresh local chunks.json snapshot.
Query:  hybrid retrieve (Pinecone dense + local BM25) -> cross-encoder rerank ->
        cited generation, with a per-query telemetry trace.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import config
from . import embeddings, ingest, telemetry
from .chunking import Chunk, chunk_document
from .generate import Answer, generate
from .rerank import rerank
from .vectorstore import PineconeStore

# Conversational inputs that should be answered directly instead of triggering a
# retrieval (otherwise "hi" pulls random chunks and the generator refuses).
_GREETING_RE = re.compile(
    r"^(hi|hey+|hello|yo|sup|howdy|gm|good\s+(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|thx|ty|ok(ay)?|cool|nice|great|bye|goodbye|"
    r"who\s+are\s+you|what\s+can\s+you\s+do|help|test)[\s!.?]*$",
    re.IGNORECASE,
)


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


def is_smalltalk(query: str) -> bool:
    q = query.strip()
    # Greetings/thanks, or a very short fragment with no question intent.
    return bool(_GREETING_RE.match(q)) or len(q) < 3


# --- Build -------------------------------------------------------------------
def build_indexes(
    strategies=config.STRATEGIES, *, do_ingest: bool = True, force: bool = False
) -> dict:
    if do_ingest:
        ingest_results = ingest.ingest_all(force=force)
        changed_tickers = {t for t, (_, changed) in ingest_results.items() if changed}
    else:
        changed_tickers = set(config.COMPANIES)

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
    trace: telemetry.QueryTrace | None = field(default=None)


class RAGPipeline:
    def __init__(self, strategy: str = "semantic"):
        self.strategy = strategy
        self.store = PineconeStore(strategy)

    def retrieve(self, query: str, k: int = config.RETRIEVE_K) -> list[Chunk]:
        qvec = embeddings.embed_query(query)
        return [c for c, _ in self.store.hybrid(query, qvec, k)]

    def answer(
        self,
        query: str,
        *,
        use_rerank: bool = True,
        top_k: int = config.TOP_K,
    ) -> RAGResult:
        if is_smalltalk(query):
            return RAGResult(
                query=query,
                answer=Answer(text=_smalltalk_reply(), sources=[], refused=False),
                retrieved=[],
                strategy=self.strategy,
                reranked=False,
            )

        t0 = time.perf_counter()
        candidates = self.retrieve(query, config.RETRIEVE_K)
        t1 = time.perf_counter()
        if use_rerank:
            reranked_pairs = rerank(query, candidates, top_k)
            top = [c for c, _ in reranked_pairs]
        else:
            top = candidates[:top_k]
        t2 = time.perf_counter()
        ans = generate(query, top)
        t3 = time.perf_counter()

        trace = telemetry.QueryTrace(
            query=query,
            strategy=self.strategy,
            reranked=use_rerank,
            refused=ans.refused,
            candidates=[{"chunk_id": c.chunk_id} for c in top],
            retrieval_ms=round((t1 - t0) * 1000, 1),
            rerank_ms=round((t2 - t1) * 1000, 1),
            generation_ms=round((t3 - t2) * 1000, 1),
            total_ms=round((t3 - t0) * 1000, 1),
            prompt_tokens=ans.usage.get("prompt_tokens", 0),
            completion_tokens=ans.usage.get("completion_tokens", 0),
            est_cost_usd=round(
                telemetry.estimate_cost(
                    config.CHAT_MODEL,
                    ans.usage.get("prompt_tokens", 0),
                    ans.usage.get("completion_tokens", 0),
                ),
                6,
            ),
        )
        telemetry.log_query(trace)

        return RAGResult(
            query=query,
            answer=ans,
            retrieved=top,
            strategy=self.strategy,
            reranked=use_rerank,
            trace=trace,
        )
