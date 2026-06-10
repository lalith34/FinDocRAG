"""Router behaviour + robustness.

The router is a pure function of the query string, so it is fully unit-testable
without network/model access. These tests pin the route each query shape takes,
that weights are sane, and that the router never raises on adversarial input.
"""
import pytest

from src import router
from src.router import (
    COMPARISON,
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
    assert r.kind in {SMALLTALK, COMPARISON, LEXICAL, SEMANTIC, HYBRID}
    # Weights are always non-negative and finite.
    assert r.dense_weight >= 0 and r.sparse_weight >= 0


def test_route_kinds_are_distinct_constants():
    kinds = {SMALLTALK, COMPARISON, LEXICAL, SEMANTIC, HYBRID}
    assert len(kinds) == 5
