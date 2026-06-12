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
  CORPUS      meta-question about what is indexed      -> skip retrieval
  COMPARISON  two or more companies named             -> per-company dense
  LEXICAL     exact-match signals dominate            -> upweight BM25
  SEMANTIC    conceptual natural-language question     -> upweight dense
  HYBRID      balanced default when nothing dominates  -> equal weights

When exactly one company is named, the route also carries that ticker so the
pipeline can constrain retrieval to that company's filing. Without it a question
like "who is the CEO of AAPL?" competes against every other company's near-
identical signature block, and the cross-encoder can seat a wrong-company chunk
in the final context. A single named ticker is an unambiguous filter; scoping to
it costs nothing and removes the cross-company contamination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config

# --- Route kinds -------------------------------------------------------------
SMALLTALK = "smalltalk"
CORPUS = "corpus"
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


# --- Corpus / meta questions -------------------------------------------------
# Questions about *what the assistant has indexed* (the corpus itself), not
# questions answered *from* the filings. These must skip retrieval: "what is in
# the corpus?" matches no 10-K chunk, so hybrid retrieval returns an arbitrary
# 2-4 companies' chunks and the listing comes out incomplete. Answering directly
# from the corpus metadata guarantees all companies are always reported.
_CORPUS_DIRECT = ("corpus", "knowledge base", "knowledgebase")
_CORPUS_SUBJECTS = (
    "companies", "company", "filings", "filing", "documents", "document",
    "tickers", "ticker", "10-k", "10-ks", "10k", "10ks", "index",
)
# A meta cue distinguishes "which companies do you have?" (corpus) from
# "what companies have the highest revenue?" (a content question).
_CORPUS_META_CUES = (
    "do you have", "do you cover", "do you support", "can i ask", "can i query",
    "are available", "is available", "are there", "are indexed", "is indexed",
    "are loaded", "are covered", "you have", "you cover", "available", "indexed",
    "what is in", "what's in", "in the corpus", "in your", "list",
)


def is_corpus_query(query: str) -> bool:
    """True for meta-questions about the indexed corpus itself. Caller should
    only treat a query as CORPUS when no specific company is named, so that
    "what filings does Apple have?" still scopes to Apple's content instead."""
    q = query.lower()
    if any(t in q for t in _CORPUS_DIRECT):
        return True
    return any(s in q for s in _CORPUS_SUBJECTS) and any(m in q for m in _CORPUS_META_CUES)


# --- Corpus-wide content questions -------------------------------------------
# "compare cloud spending across all the tickers in the corpus" names no company
# but spans every one. It must NOT be read as a corpus meta-question just because
# it says "corpus"/"tickers": it asks for data *from* the filings, so it has to
# fan out to a per-company COMPARISON, not skip retrieval. The trap this avoids:
# the word "corpus" alone matched is_corpus_query and returned the canned "here's
# what I have indexed" listing instead of an actual cross-company answer.
_ALL_CORPUS_RE = re.compile(
    r"\b(all|each|every|across)\b.{0,30}\b(compan(?:y|ies)|tickers?|filings?|"
    r"firms?|corpus|them)\b|"
    r"\b(?:all|each)\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\b|"
    r"\b(?:whole|entire)\s+corpus\b",
    re.IGNORECASE,
)
# Content/analytical intent that distinguishes a corpus-wide *question* ("compare
# revenue across all companies") from a bare *listing* ("list all the companies
# in the corpus"). Only the former should expand to a full-corpus comparison.
_AGGREGATE_INTENT = (
    "compare", "comparison", "versus", " vs", "rank", "highest", "lowest",
    "biggest", "largest", "smallest", "most", "least", "greatest", "average",
    "difference", "differ", "spend", "spending", "breakdown", "trend",
    "which company", "which has", "who has", "how much", "how many",
)


def _is_all_corpus_content(query: str, lower: str) -> bool:
    """True when the query spans the whole corpus *and* asks something analytical
    about it — i.e. a cross-company content question, not a corpus meta-listing."""
    if not _ALL_CORPUS_RE.search(query):
        return False
    return (
        any(cue in lower for cue in _AGGREGATE_INTENT)
        or any(term in lower for term in _LINE_ITEM_TERMS)
        or any(cue in lower for cue in _SEMANTIC_CUES)
        or bool(_NUMBER_RE.search(query))
    )


# --- Company detection -------------------------------------------------------
# Name aliases beyond the ticker itself, so "compare Apple and Google sales" is
# recognised as a multi-company query the same way "AAPL GOOGL" is. 10-K bodies
# use company names, never tickers, so detection must cover both. Aliases come
# from the live registry (config.COMPANY_ALIASES), so a ticker added at runtime is
# detected by name without a code change.
def mentioned_tickers(query: str) -> list[str]:
    """Tickers explicitly named in the query (by symbol or company name)."""
    q = query.lower()
    found = []
    for tk in config.COMPANIES:
        aliases = (tk.lower(),) + tuple(config.COMPANY_ALIASES.get(tk, ()))
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
# like "NVIDIA" (or a newly added ticker like "TSLA") must not be counted as a
# lexical acronym (that would wrongly pull a conceptual question about the company
# toward BM25). Read live from the registry so runtime-added tickers are covered.
def _company_tokens() -> set[str]:
    toks = {tk.upper() for tk in config.COMPANIES}
    for aliases in config.COMPANY_ALIASES.values():
        toks |= {a.upper() for a in aliases}
    return toks
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
    company_tokens = _company_tokens()
    acronyms = [a for a in _ACRONYM_RE.findall(q) if a.upper() not in company_tokens]
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
        return Route(SMALLTALK, 0.0, 0.0)

    tickers = mentioned_tickers(q)
    lower = q.lower()
    # "compare X across all the tickers/companies in the corpus": no company is
    # named but every one is meant. Expand to the full corpus so it routes to a
    # per-company COMPARISON instead of being mistaken for a corpus meta-question.
    if not tickers and _is_all_corpus_content(q, lower):
        tickers = list(config.COMPANIES)
    # A corpus/meta question with no specific company named is answered directly
    # from the corpus metadata, not via retrieval (see is_corpus_query).
    if not tickers and is_corpus_query(q):
        return Route(CORPUS, 0.0, 0.0)
    if len(tickers) >= 2:
        return Route(COMPARISON, tickers=tuple(tickers))

    # A single named company scopes retrieval to that filing (see module docstring).
    scope = tuple(tickers)  # () or one ticker

    words = _WORD_RE.findall(q)
    lex = _lexical_score(q, lower, words)
    sem = _semantic_score(lower, words)
    margin = lex - sem

    if margin >= _DECISION_MARGIN:
        return Route(LEXICAL, dense_weight=1.0, sparse_weight=2.0, tickers=scope)
    if -margin >= _DECISION_MARGIN:
        return Route(SEMANTIC, dense_weight=2.0, sparse_weight=1.0, tickers=scope)
    return Route(HYBRID, tickers=scope)
