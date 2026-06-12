"""Cited answer generation with a designed refusal path.

The generator only sees retrieved context. It must cite every claim with [n]
markers tied to the numbered sources, and if the context does not support an
answer it must return the configured refusal text rather than guess. The refusal
path is the point of this stage: a RAG app that invents numbers from a 10-K is
worse than one that admits it could not find them.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

import config
from .chunking import Chunk
from .reliability import ANTHROPIC_TIMEOUT, OPENAI_TIMEOUT, anthropic_retry, openai_retry

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


# One cached LangChain chat model per (provider, model id), built lazily so a
# missing key only errors when that provider is actually used (the other arms
# still work). max_retries=0 everywhere: tenacity owns the retry policy.
_chat_models: dict[tuple[str, str], object] = {}


def _make_chat_model(provider: str, model: str):
    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env "
                "(see .env.example) to answer with Claude models."
            )
        from langchain_anthropic import ChatAnthropic

        # No temperature/seed: Opus 4.8/4.7 reject them (they 400), so
        # determinism on this arm rests on the fixed prompt + pinned model id.
        return ChatAnthropic(
            model=model,
            api_key=config.ANTHROPIC_API_KEY,
            max_tokens=config.GEN_MAX_TOKENS,
            timeout=ANTHROPIC_TIMEOUT,
            max_retries=0,
        )

    from langchain_openai import ChatOpenAI

    if provider == "nebius":
        # Nebius Token Factory is OpenAI-compatible: same wrapper, different
        # base_url. vLLM-backed, so temperature + seed are honored.
        if not config.NEBIUS_API_KEY:
            raise RuntimeError(
                "NEBIUS_API_KEY is not set. Add it to your .env (see .env.example) "
                "to answer with Nebius Token Factory models."
            )
        return ChatOpenAI(
            model=model,
            api_key=config.NEBIUS_API_KEY,
            base_url=config.NEBIUS_BASE_URL,
            max_tokens=config.GEN_MAX_TOKENS,
            temperature=0,
            seed=0,
            timeout=OPENAI_TIMEOUT,
            max_retries=0,
        )

    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if model.startswith(("o1", "o3", "o4")):
        # Reasoning models reject temperature/seed and use max_completion_tokens.
        return ChatOpenAI(
            model=model,
            api_key=config.OPENAI_API_KEY,
            model_kwargs={"max_completion_tokens": config.GEN_MAX_TOKENS},
            timeout=OPENAI_TIMEOUT,
            max_retries=0,
        )
    return ChatOpenAI(
        model=model,
        api_key=config.OPENAI_API_KEY,
        max_tokens=config.GEN_MAX_TOKENS,
        temperature=0,
        seed=0,  # best-effort determinism on the OpenAI arm
        timeout=OPENAI_TIMEOUT,
        max_retries=0,
    )


def _get_chat_model(provider: str, model: str):
    key = (provider, model)
    if key not in _chat_models:
        _chat_models[key] = _make_chat_model(provider, model)
    return _chat_models[key]


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


# All arms return a normalized (text, usage) tuple so generate() is provider-blind.
# Every provider goes through a LangChain chat model (ChatAnthropic / ChatOpenAI);
# Nebius Token Factory is OpenAI-compatible, so it shares the ChatOpenAI wrapper
# with its own base_url. The tenacity decorators still own retries — LangChain
# wrappers re-raise the underlying SDK's exceptions, which is what they catch.
def _chat(system: str, user: str, model: str) -> tuple[str, dict]:
    provider = config.model_provider(model)
    if provider == "anthropic":
        return _invoke_anthropic(system, user, model)
    return _invoke_openai_compatible(system, user, model, provider)


def _normalize_response(msg, provider: str) -> tuple[str, dict]:
    """Map a LangChain AIMessage onto the (text, usage) contract telemetry and
    cost estimation expect, regardless of provider."""
    content = msg.content
    if isinstance(content, list):  # Anthropic can return content blocks
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    meta = getattr(msg, "response_metadata", {}) or {}
    tokens = getattr(msg, "usage_metadata", None) or {}
    usage = {
        "prompt_tokens": tokens.get("input_tokens", 0),
        "completion_tokens": tokens.get("output_tokens", 0),
        # OpenAI exposes a system_fingerprint; Anthropic/Nebius record the
        # resolved model id instead (reproducibility provenance).
        "system_fingerprint": meta.get("system_fingerprint")
        or meta.get("model")
        or meta.get("model_name"),
    }
    return content.strip(), usage


@anthropic_retry
def _invoke_anthropic(system: str, user: str, model: str) -> tuple[str, dict]:
    msg = _get_chat_model("anthropic", model).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return _normalize_response(msg, "anthropic")


@openai_retry
def _invoke_openai_compatible(
    system: str, user: str, model: str, provider: str
) -> tuple[str, dict]:
    msg = _get_chat_model(provider, model).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return _normalize_response(msg, provider)


def _normalize_citations(text: str) -> str:
    """Rewrite fullwidth citation brackets (【2】) to ASCII ([2]). Some open-weight
    models (e.g. gpt-oss via Nebius) emit the CJK form, which the citation audit
    and the UI's [n] markers would otherwise miss."""
    return text.replace("【", "[").replace("】", "]")


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

    text, usage = _chat(_SYSTEM, user, model)
    text = _normalize_citations(text)
    refused = config.REFUSAL_TEXT.lower() in text.lower()
    return Answer(text=text, sources=sources, refused=refused, usage=usage)
