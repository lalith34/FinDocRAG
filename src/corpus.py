"""Runtime corpus management: add or remove a company end-to-end.

`add_company(ticker)` is the robust, demo-safe entry point:
    validate the ticker against SEC EDGAR -> confirm a 10-K exists -> register it
    (official name + auto-derived alias) -> ingest + chunk + embed + upsert *only*
    that ticker (the other companies are untouched).
Every failure mode (unknown symbol, foreign filer with no 10-K, network/indexing
error) returns a structured CorpusResult instead of raising, so a live demo never
crashes — the caller shows res.message and moves on.

`remove_company(ticker)` is the clean inverse: delete the company's vectors from
every Pinecone namespace, prune the local BM25 snapshots, drop its processed/raw/
tree files, and unregister it — no ghost data left behind.
"""
from __future__ import annotations

from dataclasses import dataclass

import config

from . import ingest, pageindex
from .pipeline import build_indexes
from .vectorstore import PineconeStore


@dataclass
class CorpusResult:
    ticker: str
    ok: bool
    message: str
    name: str = ""
    filing_date: str = ""
    chunks: dict[str, int] | None = None


def _has_local_data(ticker: str) -> bool:
    return (config.PROCESSED_DIR / f"{ticker}.txt").exists()


def _validate(ticker: str) -> tuple[int, str, str]:
    """Return (cik, official_name, filing_date) or raise ValueError with a
    human-readable reason the ticker can't be added."""
    try:
        cik, name = ingest.resolve_company(ticker)
    except KeyError:
        raise ValueError(
            f"'{ticker}' is not a recognised SEC ticker — check the symbol "
            f"(US-listed common stock)."
        )
    try:
        _acc, _doc, date = ingest.latest_filing(cik)
    except LookupError:
        raise ValueError(
            f"{ticker} has no 10-K on EDGAR (foreign private issuers file 20-F, "
            f"funds file N-series). This corpus indexes 10-Ks only."
        )
    return cik, name, date


def add_company(ticker: str, *, rebuild: bool = False) -> CorpusResult:
    """Validate, register, and index a single ticker. Idempotent and self-healing:
    re-adding an existing company just rebuilds it; a half-finished previous add is
    completed. Returns a CorpusResult; never raises for the expected failure modes."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return CorpusResult(ticker, False, "Please enter a ticker symbol.")

    registry = config.load_registry()
    already = ticker in registry

    try:
        _cik, name, date = _validate(ticker)
    except ValueError as e:
        return CorpusResult(ticker, False, str(e))

    # Register first so the router/pipeline (which read config.COMPANIES) recognise
    # the ticker during the build. Preserve any curated aliases already on file.
    registry[ticker] = {
        "name": name,
        "aliases": registry.get(ticker, {}).get("aliases") or config.derive_aliases(name),
    }
    config.save_registry(registry)
    config.reload_registry()

    try:
        summary = build_indexes(tickers=[ticker], force=rebuild)
    except Exception as e:  # pragma: no cover - network/index failure path
        # Roll back the registry entry for a brand-new ticker so a failed add does
        # not leave an unindexed company the router would then try to retrieve.
        if not already:
            registry.pop(ticker, None)
            config.save_registry(registry)
            config.reload_registry()
        return CorpusResult(ticker, False, f"Indexing failed: {e}", name=name)

    chunks = {s: stats["chunks"] for s, stats in summary.items()}
    verb = "Rebuilt" if already else "Added"
    return CorpusResult(
        ticker, True,
        f"{verb} {name} ({ticker}) — latest 10-K filed {date}.",
        name=name, filing_date=date, chunks=chunks,
    )


def remove_company(ticker: str) -> CorpusResult:
    """Remove a company and all of its data from the corpus."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return CorpusResult(ticker, False, "Please enter a ticker symbol.")

    registry = config.load_registry()
    if ticker not in registry and not _has_local_data(ticker):
        return CorpusResult(ticker, False, f"{ticker} is not in the corpus.")
    name = registry.get(ticker, {}).get("name", ticker)

    # Vectors (Pinecone) + local BM25 snapshot, per chunking strategy.
    for strategy in config.STRATEGIES:
        try:
            store = PineconeStore(strategy)
            store.delete_ticker(ticker)
            store.save_chunks([c for c in store.chunks if c.ticker != ticker])
        except Exception as e:  # pragma: no cover - network/index failure path
            return CorpusResult(ticker, False, f"Failed clearing '{strategy}' index: {e}")

    # On-disk artefacts: cleaned text + meta, raw HTML, PageIndex tree.
    for p in (
        config.PROCESSED_DIR / f"{ticker}.txt",
        config.PROCESSED_DIR / f"{ticker}.meta.json",
        config.RAW_DIR / f"{ticker}.html",
        pageindex.TREE_DIR / f"{ticker}_tree.json",
    ):
        p.unlink(missing_ok=True)

    registry.pop(ticker, None)
    config.save_registry(registry)
    config.reload_registry()
    return CorpusResult(ticker, True, f"Removed {name} ({ticker}) from the corpus.")
