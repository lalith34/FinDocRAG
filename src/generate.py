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

_SYSTEM = f"""You are a financial analyst assistant. You answer questions ONLY \
using the numbered SOURCES provided, which are excerpts from companies' SEC \
10-K (annual) filings.

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


@dataclass
class Answer:
    text: str
    sources: list[Source]
    refused: bool


_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _build_sources(chunks: list[Chunk]) -> list[Source]:
    return [
        Source(
            n=i + 1,
            ticker=c.ticker,
            company=c.company,
            source_url=c.source_url,
            chunk_id=c.chunk_id,
            text=c.text,
        )
        for i, c in enumerate(chunks)
    ]


def generate(query: str, chunks: list[Chunk]) -> Answer:
    sources = _build_sources(chunks)
    if not sources:
        return Answer(text=config.REFUSAL_TEXT, sources=[], refused=True)

    context = "\n\n".join(
        f"[{s.n}] {s.company} ({s.ticker}) 10-K:\n{s.text}" for s in sources
    )
    user = f"SOURCES:\n{context}\n\nQUESTION: {query}\n\nAnswer with citations:"

    resp = _get_client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    refused = config.REFUSAL_TEXT.lower() in text.lower()
    return Answer(text=text, sources=sources, refused=refused)
