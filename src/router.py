"""Deterministic query router.

Decides *how* to retrieve for a given question instead of always running a fixed
hybrid blend. The router is rule-based on purpose: an LLM classifier would
reintroduce the nondeterminism this pipeline was deliberately engineered to
remove (pinned snapshot + seed), and would add a call's worth of latency/cost to
every query. Every decision here is a pure function of the query string, so it is
reproducible and unit-testable.

Robustness principle: the router tunes the dense/BM25 fusion *weights* rather than
hard-switching to a single retriever. A query it reads wrong still gets the other
signal's votes, and an ambiguous query falls back to a balanced HYBRID. The only
hard switches are for paths that genuinely need different control flow (skip
retrieval for smalltalk; per-company quota for comparisons).

Routes:
  SMALLTALK   greeting / trivially short input        -> skip retrieval
  COMPARISON  two or more companies named             -> per-company dense
  LEXICAL     exact-match signals dominate            -> upweight BM25
  SEMANTIC    conceptual natural-language question     -> upweight dense
  HYBRID      balanced default when nothing dominates  -> equal weights
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config

# --- Route kinds -------------------------------------------------------------
SMALLTALK = "smalltalk"
COMPARISON = "comparison"
LEXICAL = "lexical"
SEMANTIC = "semantic"
HYBRID = "hybrid"


@dataclass(frozen=True)
class Route:
    kind: str
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    tickers: tuple[str, ...] = ()
    reason: str = ""


# --- Smalltalk ---------------------------------------------------------------
# Conversational inputs that should be answered directly instead of triggering a
# retrieval (otherwise "hi" pulls random chunks and the generator refuses).
_GREETING_RE = re.compile(
    r"^(hi|hey+|hello|yo|sup|howdy|gm|good\s+(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|thx|ty|ok(ay)?|cool|nice|great|bye|goodbye|"
    r"who\s+are\s+you|what\s+can\s+you\s+do|help|test)[\s!.?]*$",
    re.IGNORECASE,
)


def is_smalltalk(query: str) -> bool:
    q = query.strip()
    # Greetings/thanks, or a very short fragment with no question intent.
    return bool(_GREETING_RE.match(q)) or len(q) < 3


# --- Company detection -------------------------------------------------------
# Name aliases beyond the ticker itself, so "compare Apple and Google sales" is
# recognised as a multi-company query the same way "AAPL GOOGL" is. 10-K bodies
# use company names, never tickers, so detection must cover both.
_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple",),
    "NVDA": ("nvidia",),
    "MSFT": ("microsoft",),
    "GOOGL": ("alphabet", "google"),
    "AMZN": ("amazon",),
}


def mentioned_tickers(query: str) -> list[str]:
    """Tickers explicitly named in the query (by symbol or company name)."""
    q = query.lower()
    found = []
    for tk in config.COMPANIES:
        aliases = (tk.lower(),) + _NAME_ALIASES.get(tk, ())
        if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases):
            found.append(tk)
    return found


# --- Lexical vs semantic signals ---------------------------------------------
# Multi-word financial line items: when a user names one, they almost always
# want the exact figure/row, which BM25 nails and dense paraphrase can blur.
_LINE_ITEM_TERMS = (
    "stock-based compensation", "deferred revenue", "cost of revenue",
    "cost of sales", "net sales", "net income", "net revenue", "gross margin",
    "gross profit", "operating margin", "operating income", "operating expenses",
    "research and development", "free cash flow", "earnings per share",
    "diluted eps", "basic eps", "total assets", "total liabilities",
    "shareholders' equity", "stockholders' equity", "capital expenditures",
    "effective tax rate", "dividends per share", "share repurchase",
    "accounts receivable", "retained earnings", "deferred tax", "goodwill",
)

# Short uppercase acronyms analysts type verbatim (exact-match territory).
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z&]{1,5}\b")
# Company names/tickers are subjects, not exact-match metrics: an all-caps name
# like "NVIDIA" must not be counted as a lexical acronym (that would wrongly pull
# a conceptual question about the company toward BM25).
_COMPANY_TOKENS = {tk.upper() for tk in config.COMPANIES} | {
    alias.upper() for aliases in _NAME_ALIASES.values() for alias in aliases
}
# A specific figure being asked for: $, %, or a multi-digit number.
_NUMBER_RE = re.compile(r"\$\s?\d|\d+\s?%|\b\d{2,}\b")
_QUOTED_RE = re.compile(r"\"[^\"]+\"|'[^']+'")

# Conceptual cues that favour dense/semantic retrieval.
_SEMANTIC_CUES = (
    "why", "how does", "how do", "how is", "how are", "describe", "explain",
    "summarize", "summary", "overview", "discuss", "tell me about",
    "what does", "what are the", "strategy", "approach", "outlook", "guidance",
    "competition", "competitive", "business model", "philosophy", "narrative",
    "trend", "reason", "rationale", "concern", "challenge", "opportunit",
    "risk factor", "compare the", "relationship between",
)

_WORD_RE = re.compile(r"[A-Za-z0-9$%&'.-]+")
_QUESTION_WORDS = ("what", "which", "who", "where", "when", "list", "name")

# Margin a signal tally must clear to flip away from balanced hybrid. >1 means a
# single weak cue cannot tip the decision, which keeps borderline queries on the
# robust hybrid path.
_DECISION_MARGIN = 2


def _lexical_score(q: str, lower: str, words: list[str]) -> int:
    score = 0
    if _QUOTED_RE.search(q):
        score += 2
    if any(term in lower for term in _LINE_ITEM_TERMS):
        score += 2
    # Acronyms and explicit numbers each contribute, capped so a figure-heavy
    # sentence cannot runaway-dominate. Company names are excluded (see above).
    acronyms = [a for a in _ACRONYM_RE.findall(q) if a.upper() not in _COMPANY_TOKENS]
    score += min(2, len(acronyms))
    if _NUMBER_RE.search(q):
        score += 1
    # A short keyword-style query with no question framing reads as a lookup.
    if len(words) <= 4 and not any(w in _QUESTION_WORDS for w in (lower.split() or [""])):
        score += 1
    return score


def _semantic_score(lower: str, words: list[str]) -> int:
    score = min(3, sum(1 for cue in _SEMANTIC_CUES if cue in lower))
    # Long, sentence-shaped questions lean conceptual.
    if len(words) >= 12:
        score += 1
    return score


def route(query: str) -> Route:
    """Classify a query into a retrieval Route. Pure and deterministic."""
    q = query.strip()
    if is_smalltalk(q):
        return Route(SMALLTALK, 0.0, 0.0, reason="greeting/short input")

    tickers = mentioned_tickers(q)
    if len(tickers) >= 2:
        return Route(
            COMPARISON,
            tickers=tuple(tickers),
            reason=f"{len(tickers)} companies named",
        )

    lower = q.lower()
    words = _WORD_RE.findall(q)
    lex = _lexical_score(q, lower, words)
    sem = _semantic_score(lower, words)
    margin = lex - sem

    if margin >= _DECISION_MARGIN:
        return Route(LEXICAL, dense_weight=1.0, sparse_weight=2.0,
                     reason=f"lexical signals dominate (lex={lex}, sem={sem})")
    if -margin >= _DECISION_MARGIN:
        return Route(SEMANTIC, dense_weight=2.0, sparse_weight=1.0,
                     reason=f"semantic signals dominate (lex={lex}, sem={sem})")
    return Route(HYBRID, reason=f"balanced (lex={lex}, sem={sem})")
