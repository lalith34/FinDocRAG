"""Reproducibility guarantee: re-cleaning a filing must be deterministic.

_clean_html is pure, but the EDGAR download is not — the same immutable
accession can return slightly different bytes between fetches, which made
re-ingest churn the processed char counts. The cached data/raw HTML is the
source of truth: a forced rebuild must re-clean from it (no re-download) and
produce byte-identical output."""
import json

import pytest

import config
import src.ingest as ingest


def _cached_tickers():
    """Tickers that have both a committed meta.json and a cached raw HTML."""
    out = []
    for meta_path in sorted(config.PROCESSED_DIR.glob("*.meta.json")):
        tk = meta_path.name.split(".")[0]
        if (config.RAW_DIR / f"{tk}.html").exists():
            out.append(tk)
    return out


def test_clean_html_is_deterministic():
    tickers = _cached_tickers()
    if not tickers:
        pytest.skip("no cached raw filings to clean")
    for tk in tickers:
        html = (config.RAW_DIR / f"{tk}.html").read_text(encoding="utf-8")
        assert ingest._clean_html(html) == ingest._clean_html(html)


def test_force_reclean_uses_cache_and_does_not_drift(monkeypatch, tmp_path):
    """force=True must reuse cached raw HTML (no document download) and reproduce
    the committed char_count exactly."""
    tickers = _cached_tickers()
    if not tickers:
        pytest.skip("no cached raw filings to clean")

    # Snapshot the committed metas before redirecting writes.
    committed = {
        tk: json.loads(
            (config.PROCESSED_DIR / f"{tk}.meta.json").read_text(encoding="utf-8")
        )
        for tk in tickers
    }

    # Any attempt to fetch from EDGAR during a cached reclean is a bug.
    def _no_network(url, *, as_json=False):
        raise AssertionError(f"unexpected network fetch during reclean: {url}")

    monkeypatch.setattr(ingest, "_get", _no_network)
    # Write outputs to a temp processed dir so the real corpus is untouched;
    # raw HTML is still read from the real config.RAW_DIR.
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)

    for tk in tickers:
        # Seed the temp dir with the committed meta so the cache path has the
        # accession/source_url it needs.
        (tmp_path / f"{tk}.meta.json").write_text(
            json.dumps(committed[tk]), encoding="utf-8"
        )
        meta, changed = ingest.ingest_company(
            tk, committed[tk]["company"], committed[tk]["cik"], force=True
        )
        assert changed is True
        assert meta.char_count == committed[tk]["char_count"], (
            f"{tk} drifted: {meta.char_count} != committed {committed[tk]['char_count']}"
        )
