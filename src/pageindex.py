"""PageIndex-style "vectorless" retrieval: navigate a reasoning tree of one
filing instead of doing nearest-neighbour search over embeddings.

The idea (scoped to a single document, which is where it pays off): a 10-K is a
deeply structured document — its numbered Items form a natural tree. We build that
tree once (Items as branches, chunks as leaves, each leaf summarised to one line
by a cheap LLM), cache it, and at query time let a reasoning model walk the tree
to the leaves that answer the question. Retrieval is then an interpretable *path*
("Item 1A — Risk Factors → MSFT-fixed-0019…0023") rather than an opaque set of
nearest vectors — the property that makes this attractive for auditable, financial
RAG.

Trade-offs vs the Pinecone hybrid+rerank path (measured in eval/pageindex_*_report):
  + selection comes with a traceable reasoning path; tends to rank the single best
    leaf higher (MRR) and return fewer, more precise leaves.
  - retrieval is a full LLM call (higher latency/cost than an ANN lookup) and does
    not fan out across many documents — it is a *per-document* retriever, so the
    pipeline only routes single-company questions here and falls back otherwise.

The tree is built over the `fixed` chunking snapshot (clean token windows make
tidy leaves) and loaded straight from the local chunks.json, so navigation needs
no Pinecone connection.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from .chunking import Chunk
from .reliability import anthropic_retry, make_anthropic_client

TREE_DIR = config.DATA_DIR / "pageindex"
TREE_STRATEGY = "fixed"             # chunk snapshot the leaves come from
SUMMARY_MODEL = "claude-haiku-4-5"  # cheap; only used once per leaf to build the tree
NAV_MODEL = config.CHAT_MODEL       # the reasoning model that walks the tree
TOP_K = config.TOP_K

_client = None


def _cli():
    global _client
    if _client is None:
        _client = make_anthropic_client()
    return _client


def _tree_path(ticker: str) -> Path:
    return TREE_DIR / f"{ticker}_tree.json"


def load_chunks(ticker: str) -> list[Chunk]:
    """Leaves for a ticker, read from the local fixed-strategy snapshot (no
    Pinecone needed — this is the same file BM25 and the eval read)."""
    snap = config.INDEX_DIR / TREE_STRATEGY / "chunks.json"
    if not snap.exists():
        raise RuntimeError(
            f"No chunk snapshot at {snap}. Run `python -m scripts.build_index` first."
        )
    raw = json.loads(snap.read_text(encoding="utf-8"))
    return [Chunk.from_dict(c) for c in raw if c["ticker"] == ticker]


# --- tree build --------------------------------------------------------------
_SUMMARY_SYS = (
    "Summarise this excerpt from a 10-K filing in ONE short line (max 20 words) "
    "describing what facts/figures it contains, so a router can decide whether it "
    "answers a question. No preamble."
)


@anthropic_retry
def _summarise(text: str) -> str:
    resp = _cli().messages.create(
        model=SUMMARY_MODEL,
        max_tokens=64,
        system=_SUMMARY_SYS,
        messages=[{"role": "user", "content": text[:4000]}],
    )
    out = "".join(b.text for b in resp.content if b.type == "text")
    return out.strip().replace("\n", " ")


def build_tree(ticker: str, chunks: list[Chunk] | None = None, *, rebuild: bool = False,
               verbose: bool = False) -> dict:
    """Tree = {section_label: [{id, summary}, ...]} in document order. Leaf
    summaries are LLM-generated once and cached to disk, so navigation is the only
    per-query cost thereafter."""
    path = _tree_path(ticker)
    if path.exists() and not rebuild:
        return json.loads(path.read_text())

    chunks = chunks if chunks is not None else load_chunks(ticker)
    if not chunks:
        raise RuntimeError(f"No chunks found for {ticker}; cannot build a tree.")

    TREE_DIR.mkdir(parents=True, exist_ok=True)
    tree: dict[str, list[dict]] = {}
    for i, c in enumerate(chunks):
        summary = _summarise(c.text)
        tree.setdefault(c.section or "Front Matter", []).append(
            {"id": c.chunk_id, "summary": summary}
        )
        if verbose:
            print(f"  [tree:{ticker}] {c.chunk_id} ({i + 1}/{len(chunks)}) "
                  f"{summary[:60]}", flush=True)
    path.write_text(json.dumps(tree, indent=2))
    if verbose:
        print(f"  tree -> {path}")
    return tree


def render_tree(tree: dict) -> str:
    """The tree as the navigator sees it: Item headers with their numbered leaves."""
    lines = []
    for section, leaves in tree.items():
        lines.append(f"\n## {section}")
        for leaf in leaves:
            lines.append(f"- {leaf['id']}: {leaf['summary']}")
    return "\n".join(lines)


# --- navigation (the vectorless retrieval step) ------------------------------
_NAV_SYS = (
    "You are a retrieval router for a single company's 10-K filing. You are given "
    "the filing's TABLE OF CONTENTS as a tree: 10-K Items, each with numbered leaf "
    "nodes summarised in one line. Given a QUESTION, REASON about which Item(s) "
    "would contain the answer, then select the specific leaf node ids that answer "
    "it.\n"
    "Respond ONLY as JSON: "
    '{"section": "<the Item you navigated to>", '
    '"reasoning": "<one sentence: why that Item, then why those leaves>", '
    '"node_ids": ["<id>", ...]}. '
    f"Select at most {TOP_K} node ids, best first. Pick only nodes whose summary "
    "plausibly contains the answer; fewer precise nodes beat many vague ones."
)


@anthropic_retry
def _nav_call(question: str, rendered: str) -> dict:
    resp = _cli().messages.create(
        model=NAV_MODEL,
        max_tokens=512,
        system=_NAV_SYS + '\n\nRespond ONLY with a JSON object, no other text.',
        messages=[
            {"role": "user", "content": f"TREE:\n{rendered}\n\nQUESTION: {question}"},
        ],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start : end + 1])


class PageIndexRetriever:
    """Per-document tree-navigation retriever. Trees (and their rendered form and
    id->chunk maps) are built lazily on first use per ticker and memoised, so a
    long-lived process (the Streamlit app) pays the build cost at most once."""

    def __init__(self):
        self._cache: dict[str, tuple[dict, str, dict[str, Chunk]]] = {}

    def _load(self, ticker: str):
        if ticker not in self._cache:
            chunks = load_chunks(ticker)
            tree = build_tree(ticker, chunks)
            self._cache[ticker] = (tree, render_tree(tree),
                                   {c.chunk_id: c for c in chunks})
        return self._cache[ticker]

    def retrieve(self, question: str, ticker: str, k: int = TOP_K):
        """Navigate the ticker's tree. Returns (chunks, path_str, reasoning)."""
        _tree, rendered, by_id = self._load(ticker)
        nav = _nav_call(question, rendered)
        ids = [i for i in nav.get("node_ids", []) if i in by_id][:k]
        chunks = [by_id[i] for i in ids]
        path = f"{nav.get('section', '?')} → {', '.join(ids) or '(none)'}"
        return chunks, path, nav.get("reasoning", "")
