"""Deterministic guardrails — input safety, financial-domain safety, and output
integrity — wrapped around the RAG pipeline.

Design: rule-based and pure, for the same reasons the router is (see router.py).
An LLM-based guard would add a model call's latency/cost to every query and
reintroduce nondeterminism; these checks are pure functions of the text, so they
are fast, free, reproducible, and unit-testable. They are a first line of defence,
not a replacement for the grounded-generation + refusal prompt in generate.py.

Three layers:
  1. Input guard (check_input)  — runs BEFORE retrieval/generation:
       - length cap                     (prompt-stuffing / cost abuse)
       - prompt-injection / extraction  (instruction override, system-prompt leak,
                                         context-tag breakout)
       - investment-advice solicitation (a financial assistant must not tell a user
                                         to buy/sell; it deflects to the disclosures)
  2. Output guard (audit_citations) — runs AFTER generation:
       - citation integrity: every [n] marker must point at a real source; a
         substantive answer with no citation at all is flagged as ungrounded.
  3. Disclaimer (with_disclaimer)   — a standing "from filings / not advice" footer
       appended to substantive answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Hard cap on query length. Real analyst questions are short; anything longer is
# prompt-stuffing or an attempt to bury an injection in a wall of text.
MAX_QUERY_CHARS = 2000

REFUSAL_DISCLAIMER = (
    "_Grounded in the cited SEC 10-K filings. Figures may reflect retrieval limits — "
    "verify against the linked filing. This is not investment advice._"
)


@dataclass(frozen=True)
class InputGuard:
    allowed: bool
    reply: str = ""      # safe completion to return instead, when not allowed
    reason: str = ""      # telemetry tag: "" | length | injection | advice


@dataclass
class CitationAudit:
    n_citations: int = 0
    unique_cited: list[int] = field(default_factory=list)
    dangling: list[int] = field(default_factory=list)  # cited ids with no such source
    has_citation: bool = False
    ungrounded: bool = False  # substantive answer that cites nothing


# --- 1. Prompt injection / instruction override ------------------------------
# Tight set, anchored on override/extraction verbs and context-tag breakout, to
# catch the common attacks without flagging ordinary finance questions. We avoid
# loose triggers like "act as" (legitimately appears: "entities that act as ...").
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|above|prior|earlier|"
               r"these|your)\s+(?:instructions?|prompts?|rules?|directions?)", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+|the\s+|your\s+)?(?:previous\s+|above\s+)?"
               r"(?:instructions?|rules?|prompts?|context)", re.I),
    re.compile(r"\bforget\s+(?:everything|all|your|the)\b.*\b(?:instructions?|rules?|"
               r"said|above)\b", re.I),
    re.compile(r"\b(?:reveal|show|print|repeat|output|tell\s+me)\b.{0,30}\b"
               r"(?:system\s+prompt|your\s+instructions?|the\s+prompt|your\s+rules?)", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\byou\s+are\s+now\b|\bnew\s+instructions?\s*:|\bdeveloper\s+mode\b|"
               r"\bjailbreak\b|\bpretend\s+(?:you|to\s+be)\b", re.I),
    # Context-tag breakout: the user trying to close/open the structured tags the
    # generator uses (<context>, <source>, <system>, <question>).
    re.compile(r"</?\s*(?:system|context|source|sources|question|instructions?)\s*>", re.I),
]

# --- 2. Investment-advice solicitation ---------------------------------------
# A 10-K assistant reports what filings disclose; it must not make buy/sell calls
# or predictions. These are deflected (not refused) toward the factual disclosures.
_ADVICE_PATTERNS = [
    re.compile(r"\bshould\s+i\s+(?:buy|sell|invest(?:\s+in)?|hold|short|own|purchase|dump)\b", re.I),
    re.compile(r"\bis\s+(?:it|this|now|\w+)\s+(?:a\s+)?(?:good|bad|smart|wise)\s+"
               r"(?:time\s+to\s+(?:buy|sell|invest)|buy|investment|stock\s+to\s+buy)", re.I),
    re.compile(r"\b(?:buy|sell|hold)\s+recommendation\b|\bprice\s+target\b", re.I),
    re.compile(r"\bwill\s+(?:the\s+)?(?:stock|share|price|shares?)\b.{0,30}\b"
               r"(?:go\s+up|go\s+down|rise|fall|drop|increase|decline|crash|moon|soar)", re.I),
    re.compile(r"\bwhat\s+stock(?:s)?\s+should\s+i\b|\bis\s+it\s+worth\s+(?:buying|investing)\b", re.I),
    re.compile(r"\b(?:recommend|suggest)\s+(?:me\s+)?(?:a\s+|some\s+)?(?:stock|share|investment)s?\b", re.I),
]

_INJECTION_REPLY = (
    "I can only answer questions grounded in the SEC 10-K filings I have indexed, "
    "and I can't change or reveal those instructions. Ask me about the companies' "
    "financials, segments, risk factors, or other disclosures."
)
_ADVICE_REPLY = (
    "I don't provide investment advice, recommendations, or price predictions. "
    "I can tell you what the filings actually disclose — for example revenue, "
    "margins, R&D spend, segment results, or risk factors. Try asking a factual "
    "question about the 10-Ks, e.g. *\"What were the company's total revenues and "
    "operating margin last year?\"*"
)


def check_input(query: str) -> InputGuard:
    """Screen a query before it reaches retrieval/generation. Returns an InputGuard;
    when allowed is False, `reply` is the safe completion to return to the user."""
    q = (query or "").strip()
    if len(q) > MAX_QUERY_CHARS:
        return InputGuard(
            False,
            f"That question is too long for me to process safely (limit "
            f"{MAX_QUERY_CHARS:,} characters). Please shorten it to the specific "
            f"thing you want to know from the filings.",
            reason="length",
        )
    if any(p.search(q) for p in _INJECTION_PATTERNS):
        return InputGuard(False, _INJECTION_REPLY, reason="injection")
    if any(p.search(q) for p in _ADVICE_PATTERNS):
        return InputGuard(False, _ADVICE_REPLY, reason="advice")
    return InputGuard(True)


# --- output integrity --------------------------------------------------------
_CITE_RE = re.compile(r"\[(\d+)\]")


def audit_citations(text: str, n_sources: int, *, refused: bool = False) -> CitationAudit:
    """Verify the generated answer's [n] markers against the sources it was given.

    A dangling citation (an [n] with no matching source) means the model invented a
    reference — exactly what a grounded-RAG system must catch. A substantive answer
    (non-refusal, sources were available) that cites nothing is flagged ungrounded."""
    ids = [int(m) for m in _CITE_RE.findall(text or "")]
    unique = sorted(set(ids))
    dangling = sorted({i for i in unique if i < 1 or i > n_sources})
    has_citation = bool(ids)
    ungrounded = (not refused) and n_sources > 0 and not has_citation
    return CitationAudit(
        n_citations=len(ids),
        unique_cited=unique,
        dangling=dangling,
        has_citation=has_citation,
        ungrounded=ungrounded,
    )


def with_disclaimer(text: str) -> str:
    """Append the standing not-advice / verify-against-source footer once."""
    if not text or REFUSAL_DISCLAIMER in text:
        return text
    return f"{text}\n\n{REFUSAL_DISCLAIMER}"
