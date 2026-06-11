"""Registry-backed corpus: alias derivation, persistence, and that a ticker added
at runtime is immediately recognised by the router — all without network access.

These pin the contract that makes adding a company a data change, not a code change.
"""
import pytest

import config
from src import router


@pytest.fixture
def temp_registry(tmp_path, monkeypatch):
    """Point the registry at a throwaway file; restore the real corpus after."""
    monkeypatch.setattr(config, "COMPANIES_FILE", tmp_path / "companies.json")
    config.reload_registry()  # seed (file absent yet)
    yield tmp_path / "companies.json"
    monkeypatch.undo()
    config.reload_registry()  # back to the real on-disk/seed corpus


# --- alias derivation --------------------------------------------------------
@pytest.mark.parametrize(
    "name,alias",
    [
        ("Tesla, Inc.", "tesla"),
        ("Meta Platforms, Inc.", "meta"),
        ("Apple Inc.", "apple"),
        ("NVIDIA Corporation", "nvidia"),
        ("The Coca-Cola Company", "coca-cola"),
        ("Berkshire Hathaway Inc.", "berkshire"),
    ],
)
def test_derive_aliases_strips_corporate_suffixes(name, alias):
    assert config.derive_aliases(name) == [alias]


def test_derive_aliases_empty_is_safe():
    assert config.derive_aliases("") == []


# --- persistence + corrupt-file resilience -----------------------------------
def test_registry_roundtrips_to_disk(temp_registry):
    reg = config.load_registry()
    reg["TSLA"] = {"name": "Tesla, Inc.", "aliases": config.derive_aliases("Tesla, Inc.")}
    config.save_registry(reg)
    assert config.load_registry()["TSLA"]["name"] == "Tesla, Inc."


def test_corrupt_registry_falls_back_to_seed(temp_registry):
    temp_registry.write_text("{ not valid json", encoding="utf-8")
    reg = config.load_registry()
    assert "AAPL" in reg  # seed, not a crash


# --- the payoff: a runtime add is router-visible -----------------------------
def test_added_ticker_is_detected_by_router(temp_registry):
    reg = config.load_registry()
    reg["TSLA"] = {"name": "Tesla, Inc.", "aliases": ["tesla"]}
    config.save_registry(reg)
    config.reload_registry()

    assert "TSLA" in config.COMPANIES
    assert "TSLA" in router.mentioned_tickers("how much revenue did Tesla make?")
    assert "TSLA" in router.mentioned_tickers("TSLA gross margin")
    # Two named companies -> comparison route, using the freshly added one.
    assert router.route("compare Tesla and Apple revenue").kind == router.COMPARISON


def test_removed_ticker_is_no_longer_detected(temp_registry):
    reg = config.load_registry()
    reg.pop("AMZN", None)
    config.save_registry(reg)
    config.reload_registry()

    assert "AMZN" not in config.COMPANIES
    assert router.mentioned_tickers("what are Amazon's net sales?") == []


def test_new_ticker_symbol_not_miscounted_as_lexical_acronym(temp_registry):
    """A 4-letter ticker like TSLA must be treated as a company subject, not an
    all-caps exact-match acronym that drags the route toward BM25."""
    reg = config.load_registry()
    reg["TSLA"] = {"name": "Tesla, Inc.", "aliases": ["tesla"]}
    config.save_registry(reg)
    config.reload_registry()

    # Conceptual question about the company should not be forced to LEXICAL.
    assert router.route("explain Tesla's overall business strategy and outlook").kind == (
        router.SEMANTIC
    )
