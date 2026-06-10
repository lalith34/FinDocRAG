"""Orchestration: build the indexes and answer queries end-to-end.

Build:  ingest -> chunk (per strategy) -> embed -> persist vector store.
Query:  hybrid retrieve -> (optional) rerank -> cited generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config
from . import embeddings, ingest
from .chunking import Chunk, chunk_document
from .generate import Answer, generate
from .rerank import rerank
from .vectorstore import VectorStore

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
def build_indexes(strategies=config.STRATEGIES, *, do_ingest: bool = True) -> dict:
    if do_ingest:
        ingest.ingest_all()
    docs = ingest.load_processed()
    if not docs:
        raise RuntimeError("No processed documents found. Run ingestion first.")

    summary = {}
    for strategy in strategies:
        print(f"\n=== Building '{strategy}' index ===")
        all_chunks: list[Chunk] = []
        for ticker, payload in docs.items():
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
            all_chunks.extend(chunks)

        print(f"  embedding {len(all_chunks)} chunks ...")
        vecs = embeddings.embed_texts([c.text for c in all_chunks])
        store = VectorStore(strategy, all_chunks, vecs)
        store.save()
        summary[strategy] = {
            "chunks": len(all_chunks),
            "avg_tokens": round(
                sum(c.token_count for c in all_chunks) / max(1, len(all_chunks)), 1
            ),
        }
        print(f"  saved -> data/index/{strategy}/")
    return summary


# --- Query -------------------------------------------------------------------
@dataclass
class RAGResult:
    query: str
    answer: Answer
    retrieved: list[Chunk]
    strategy: str
    reranked: bool


class RAGPipeline:
    def __init__(self, strategy: str = "semantic"):
        self.strategy = strategy
        self.store = VectorStore.load(strategy)

    def retrieve(self, query: str, k: int = config.RETRIEVE_K) -> list[Chunk]:
        qvec = embeddings.embed_query(query)
        hits = self.store.hybrid(query, qvec, k)
        return [self.store.get(i) for i, _ in hits]

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

        candidates = self.retrieve(query, config.RETRIEVE_K)
        if use_rerank:
            top = [c for c, _ in rerank(query, candidates, top_k)]
        else:
            top = candidates[:top_k]
        ans = generate(query, top)
        return RAGResult(
            query=query,
            answer=ans,
            retrieved=top,
            strategy=self.strategy,
            reranked=use_rerank,
        )
