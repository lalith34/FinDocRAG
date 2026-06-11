"""Cited answer generation with a designed refusal path.

The generator only sees retrieved context. It must cite every claim with [n]
markers tied to the numbered sources, and if the context does not support an
answer it must return the configured refusal text rather than guess. The refusal
path is the point of this stage: a RAG app that invents numbers from a 10-K is
worse than one that admits it could not find them.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from .chunking import Chunk
from .reliability import make_openai_client, openai_retry

_SYSTEM = f"""You are a financial analyst assistant. You answer questions ONLY \
using the SOURCES provided, which are excerpts from companies' SEC 10-K (annual) \
filings. Each source is given as an <source id="N" ...> element; cite a fact by \
that id as [N].

IMPORTANT — these are ANNUAL filings. They contain full-fiscal-year figures \
(usually the latest 2-3 years side by side in a table), and NEVER quarterly \
figures. Therefore:
- NEVER invent, infer, split, or label any figure as a quarter (Q1/Q2/Q3/Q4) or \
  as "three months ended". The SOURCES do not contain quarterly net income, \
  revenue, or profit. Do not present annual or prior-year column values as if \
  they were quarters.
- If the user asks for quarterly data (e.g. "across all quarters", "Q3", "last \
  quarter"), do NOT refuse and do NOT fabricate a per-quarter breakdown. Instead, \
  state plainly that 10-K filings report annual results only, then give the \
  relevant ANNUAL figure(s).
- Only attach a time-period label (e.g. a fiscal year) to a number if that exact \
  label is shown next to the number in the SOURCES. Income-statement tables list \
  multiple years; map each number to its column using the "Years ended ..." header \
  row. If no such header is present, report the figures as the most recent reported \
  years without guessing the mapping, rather than assigning years (or quarters) you \
  are unsure of.
- When the question asks for the "latest", "last", "most recent", or "this year" \
  figure and the table shows several years, use the MOST RECENT fiscal year — the \
  one with the latest period-end date in the header (the first/leftmost data \
  column), not a prior year.

Some SOURCES may be unrelated to the question, or contain only partial or \
segment-level figures. Ignore those and answer from whichever SOURCES are \
relevant — you do not need to use every source. The presence of irrelevant or \
partial SOURCES is NEVER a reason to refuse, as long as at least one source \
contains a figure or statement that addresses the question.

Rules:
- Use only information present in the SOURCES. Do not use outside knowledge.
- Cite every factual claim with bracketed markers like [1] or [2][3] that refer \
to the source numbers.
- Quote figures exactly as they appear; do not estimate, convert, or compute new \
numbers (you may state a difference only if it is shown in the SOURCES).
- For comparison questions, give each company's figure with its citation, then a \
one-line comparison.
- Refuse ONLY when the SOURCES contain no figure or statement relevant to the \
subject of the question (e.g. the metric or company simply is not there). A \
period mismatch (quarter vs year) is NOT grounds for refusal. When you do \
refuse, reply with exactly:
  "{config.REFUSAL_TEXT}"
- Be concise and specific. Name the company when relevant.
"""


@dataclass
class Source:
    n: int
    ticker: str
    company: str
    source_url: str
    chunk_id: str
    text: str
    section: str = ""


@dataclass
class Answer:
    text: str
    sources: list[Source]
    refused: bool
    usage: dict = None  # {"prompt_tokens": int, "completion_tokens": int}

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = make_openai_client()
    return _client


def _order_for_context(chunks: list[Chunk]) -> list[Chunk]:
    """Lay out relevance-ranked chunks to fight "lost in the middle" (deck S2 §9):
    long-context models attend most to the start and end of the prompt and least
    to the middle. Input is assumed best-first (the reranker's output); we put the
    strongest chunk first, the second-strongest last, and bury weaker ones in the
    middle. For [r1,r2,r3,r4,r5] this yields [r1,r3,r5,r4,r2]."""
    front = chunks[0::2]          # ranks 1, 3, 5, ... at the front, in order
    back = chunks[1::2][::-1]     # ranks 2, 4, ... at the back, reversed
    return front + back


def _build_sources(chunks: list[Chunk]) -> list[Source]:
    return [
        Source(
            n=i + 1,
            ticker=c.ticker,
            company=c.company,
            source_url=c.source_url,
            chunk_id=c.chunk_id,
            text=c.text,
            section=c.section,
        )
        for i, c in enumerate(chunks)
    ]


@openai_retry
def _chat(messages: list[dict], model: str):
    return _get_client().chat.completions.create(
        model=model,
        temperature=0,
        seed=config.GEN_SEED,
        messages=messages,
    )


def generate(
    query: str, chunks: list[Chunk], *, model: str | None = None, reorder: bool = True
) -> Answer:
    model = model or config.CHAT_MODEL
    # Reorder relevance-ranked chunks into the lost-in-the-middle layout. Skipped
    # (reorder=False) for comparison queries, whose chunks are grouped per company
    # rather than globally ranked, so interleaving them would scramble the groups.
    if reorder and len(chunks) > 2:
        chunks = _order_for_context(chunks)
    sources = _build_sources(chunks)
    if not sources:
        return Answer(text=config.REFUSAL_TEXT, sources=[], refused=True)

    # Structured <source> elements (deck S2 §9: "structured context > plain prose").
    # The id is the citation number; the section gives the model the 10-K location.
    context = "\n".join(
        f'<source id="{s.n}" company="{s.company}" ticker="{s.ticker}"'
        f'{f" section={s.section!r}" if s.section else ""}>\n{s.text}\n</source>'
        for s in sources
    )
    user = (
        f"<context>\n{context}\n</context>\n\n"
        f"<question>{query}</question>\n\n"
        "Answer with citations:"
    )

    resp = _chat(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        model,
    )
    text = resp.choices[0].message.content.strip()
    refused = config.REFUSAL_TEXT.lower() in text.lower()
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        # The backend build that served this completion. With temperature=0 and a
        # fixed seed, output is reproducible only while this fingerprint is stable;
        # a change here is OpenAI's signal that determinism may have shifted.
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
    }
    return Answer(text=text, sources=sources, refused=refused, usage=usage)
