"""Router behaviour + robustness.

The router is a pure function of the query string, so it is fully unit-testable
without network/model access. These tests pin the route each query shape takes,
that weights are sane, and that the router never raises on adversarial input.
"""
import pytest

import config
from src import router
from src.router import (
    COMPARISON,
    CORPUS,
    HYBRID,
    LEXICAL,
    SEMANTIC,
    SMALLTALK,
    route,
)


# --- Smalltalk ---------------------------------------------------------------
@pytest.mark.parametrize("q", ["hi", "hello!", "thanks", "who are you", "  ", "ok"])
def test_smalltalk_skips_retrieval(q):
    r = route(q)
    assert r.kind == SMALLTALK
    assert r.dense_weight == 0.0 and r.sparse_weight == 0.0


# --- Corpus / meta -----------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        "What is in the corpus?",
        "what's in your corpus",
        "What companies can I ask about?",
        "which companies do you have?",
        "What filings do you have?",
        "what documents are indexed?",
        "list the companies you cover",
        "how many 10-Ks do you have?",
    ],
)
def test_corpus_questions_skip_retrieval(q):
    r = route(q)
    assert r.kind == CORPUS
    assert r.dense_weight == 0.0 and r.sparse_weight == 0.0


@pytest.mark.parametrize(
    "q",
    [
        "What companies have the highest revenue?",  # content question, no meta cue
        "What were Apple's total net sales",          # names a company
        "what does the company say about its segments",
    ],
)
def test_content_questions_are_not_corpus(q):
    assert route(q).kind != CORPUS


# --- Comparison --------------------------------------------------------------
@pytest.mark.parametrize(
    "q,n",
    [
        ("compare Apple and Microsoft net sales", 2),
        ("AAPL vs NVDA vs GOOGL revenue", 3),
        ("how do amazon, google and microsoft describe AI?", 3),
    ],
)
def test_two_or_more_companies_routes_comparison(q, n):
    r = route(q)
    assert r.kind == COMPARISON
    assert len(r.tickers) == n


def test_single_company_is_not_comparison():
    assert route("what were Apple's total net sales").kind != COMPARISON


@pytest.mark.parametrize(
    "q",
    [
        # Regression: "corpus"/"tickers" as a *scope* word for a content question
        # must not be mistaken for a corpus meta-listing. These name no company but
        # span every one, so they fan out to a full-corpus per-company comparison.
        "Compare Cloud Based Spending across all the Tickers in the Corpus?",
        "rank all companies by revenue",
        "average R&D spend across all five companies",
        "which company has the highest operating margin across the corpus",
    ],
)
def test_corpus_wide_content_question_routes_comparison(q):
    r = route(q)
    assert r.kind == COMPARISON
    assert len(r.tickers) == len(config.COMPANIES)


@pytest.mark.parametrize(
    "q",
    [
        # A bare listing that spans the corpus is still a meta-question, not a
        # content comparison — no analytical intent, so it stays CORPUS.
        "list all the companies in the corpus",
        "show me all the tickers you have",
    ],
)
def test_corpus_wide_listing_stays_corpus(q):
    assert route(q).kind == CORPUS


@pytest.mark.parametrize(
    "q,expected",
    [
        # "GOOG" (not just GOOGL/Alphabet/Google) must resolve to the corpus
        # ticker, else a 5-ticker ranking silently drops Alphabet.
        ("Rank MSFT, AMZN, GOOG, AAPL, and NVDA by revenue", "GOOGL"),
        ("GOOG cloud revenue", "GOOGL"),
    ],
)
def test_goog_alias_resolves_to_googl(q, expected):
    assert expected in router.mentioned_tickers(q)


# --- Single-company scoping --------------------------------------------------
@pytest.mark.parametrize(
    "q,tk",
    [
        ("Who is the Chief Executive Officer of AAPL?", "AAPL"),
        ("what were Apple's total net sales", "AAPL"),
        ("explain Microsoft's cloud strategy and outlook", "MSFT"),  # semantic route
        ("nvidia net sales figure", "NVDA"),                         # lexical route
    ],
)
def test_one_named_company_scopes_route_to_that_ticker(q, tk):
    r = route(q)
    assert r.kind != COMPARISON
    assert r.tickers == (tk,)


def test_no_company_named_leaves_route_unscoped():
    r = route("what were total net sales last year")
    assert r.tickers == ()


# --- Lexical -----------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        'find the exact phrase "stock-based compensation"',
        "net sales figure",
        "EPS GAAP COGS",
        "total assets and total liabilities",
        "deferred revenue 2023",
    ],
)
def test_lexical_queries_upweight_bm25(q):
    r = route(q)
    assert r.kind == LEXICAL
    assert r.sparse_weight > r.dense_weight


# --- Semantic ----------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        "why does the company believe its competitive strategy will succeed",
        "how does Microsoft describe its overall business approach and outlook",
        "explain the rationale behind the company's cloud strategy and philosophy",
    ],
)
def test_semantic_queries_upweight_dense(q):
    r = route(q)
    assert r.kind == SEMANTIC
    assert r.dense_weight > r.sparse_weight


# --- Hybrid fallback ---------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        "tell me about Amazon's revenue",  # one weak semantic cue, no exact-match cue
        "Apple cloud services",
    ],
)
def test_ambiguous_queries_fall_back_to_balanced_hybrid(q):
    r = route(q)
    assert r.kind == HYBRID
    assert r.dense_weight == r.sparse_weight == 1.0


# --- Determinism + robustness ------------------------------------------------
def test_route_is_deterministic():
    q = "explain the rationale behind the cloud strategy"
    first = route(q)
    assert all(route(q) == first for _ in range(5))


@pytest.mark.parametrize(
    "q",
    ["", "   ", "?????", "$$$ %%% &&&", "🚀🚀🚀", "a" * 5000, "SELECT * FROM x; --"],
)
def test_router_never_raises_on_adversarial_input(q):
    r = route(q)
    assert r.kind in {SMALLTALK, CORPUS, COMPARISON, LEXICAL, SEMANTIC, HYBRID}
    # Weights are always non-negative and finite.
    assert r.dense_weight >= 0 and r.sparse_weight >= 0


def test_route_kinds_are_distinct_constants():
    kinds = {SMALLTALK, CORPUS, COMPARISON, LEXICAL, SEMANTIC, HYBRID}
    assert len(kinds) == 6
