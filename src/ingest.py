"""Ingestion + cleaning: pull the latest 10-K for each company from SEC EDGAR
and turn the filing HTML into clean, table-aware plain text.

The cleaning step matters as much as the model: we strip script/style/markup,
decode entities, drop boilerplate whitespace, and — crucially for financial
documents — flatten HTML tables into pipe-delimited rows so dollar figures keep
their row/column context instead of collapsing into a number soup.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import asdict, dataclass

import requests
from bs4 import BeautifulSoup

try:  # some filings are inline-XBRL/XML; we parse them as HTML on purpose
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

import config

from .reliability import HTTP_TIMEOUT, http_retry

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# --- iXBRL/XBRL artifact stripping -------------------------------------------
# SEC filings are inline-XBRL: the visible 10-K is wrapped in a "hidden" block of
# machine-readable facts (tag contexts, CIK, period dates, units, taxonomy URLs).
# BeautifulSoup's get_text() dumps that block into the plain text as a long run of
# single-token lines, e.g.
#     0001045810
#     us-gaap:RetainedEarningsMember
#     2025-01-27
#     iso4217:USD
#     P1Y
# These tokens are pure tagging noise: they carry no prose and crowd out real
# tables in dense retrieval. We drop a line only when, stripped, it matches one of
# these *unambiguous* XBRL-only shapes. Real financial tables survive because the
# table flattener emits pipe-delimited rows ("Revenue | $ | 215,938 | ...") that
# never match — a qualified name, lone CIK, lone ISO date, ISO duration, or
# taxonomy URL is always a single whitespace-free token, while table rows and
# prose contain spaces and/or pipes.
#
# We deliberately do NOT strip bare years (e.g. "2002"), "FY", or "true"/"false":
# those also appear as legitimate content (exhibit dates, headings) and chunking
# can push a real year to a line start, so removing them risks deleting real text.
_XBRL_LINE_PATTERNS = (
    # Qualified XBRL names: us-gaap:/dei:/srt:/<ticker>:/iso4217:/xbrli:..., incl.
    # the "...Member" dimension members. Prefix may contain a hyphen (us-gaap).
    re.compile(r"^[A-Za-z][\w-]*:[\w.\-]+$"),
    re.compile(r"^\d{10}$"),                 # bare CIK, e.g. 0001045810
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),      # lone ISO context date
    re.compile(r"^P\d+[YMWD]$"),             # ISO-8601 duration, e.g. P1Y / P3Y
    re.compile(r"^https?://\S+$"),           # taxonomy URLs (fasb.org/...)
)


def _is_xbrl_artifact(line: str) -> bool:
    return any(p.match(line) for p in _XBRL_LINE_PATTERNS)


def strip_xbrl_artifacts(text: str) -> str:
    """Remove inline-XBRL tagging lines, keeping prose and flattened tables."""
    kept = [ln for ln in text.splitlines() if not _is_xbrl_artifact(ln.strip())]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


@dataclass
class FilingMeta:
    ticker: str
    company: str
    cik: int
    form: str
    filing_date: str
    accession: str
    source_url: str
    char_count: int


@http_retry
def _get(url: str, *, as_json: bool = False):
    resp = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    time.sleep(0.2)  # stay under SEC's ~10 req/s guidance
    return resp.json() if as_json else resp.text


def resolve_company(ticker: str) -> tuple[int, str]:
    """Return (cik, official company name) for a ticker from SEC's table. Raises
    KeyError if the symbol is not a recognised SEC filer."""
    data = _get(_TICKERS_URL, as_json=True)
    t = ticker.upper()
    for v in data.values():
        if v["ticker"].upper() == t:
            return int(v["cik_str"]), str(v.get("title") or ticker)
    raise KeyError(f"Ticker {ticker} not found in SEC ticker table")


def resolve_ciks(tickers: list[str]) -> dict[str, int]:
    data = _get(_TICKERS_URL, as_json=True)
    table = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    out = {}
    for t in tickers:
        if t.upper() not in table:
            raise KeyError(f"Ticker {t} not found in SEC ticker table")
        out[t] = table[t.upper()]
    return out


def latest_filing(cik: int, form: str = config.FILING_FORM) -> tuple[str, str, str]:
    """Return (accession_no_dashes, primary_document, filing_date) for the most
    recent filing of the given form type."""
    data = _get(_SUBMISSIONS_URL.format(cik=cik), as_json=True)
    recent = data["filings"]["recent"]
    for form_t, acc, doc, date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
    ):
        if form_t == form:
            return acc.replace("-", ""), doc, date
    raise LookupError(f"No {form} filing found for CIK {cik}")


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()

    # Flatten tables into pipe-delimited rows so numbers keep their labels.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", c.get_text(" ", strip=True))
                for c in tr.find_all(["td", "th"])
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n" if rows else "")

    text = soup.get_text("\n")
    # Collapse whitespace and entity artifacts.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln).strip()
    return strip_xbrl_artifacts(text)


def ingest_company(
    ticker: str, company: str, cik: int, *, force: bool = False, refresh_raw: bool = False
) -> tuple[FilingMeta, bool]:
    """Fetch/clean the latest filing. Returns (meta, changed).

    The raw filing HTML cached under data/raw is the source of truth. Re-cleaning
    (e.g. after changing the cleaner) reuses that cached HTML, so the processed
    text is reproducible byte-for-byte — _clean_html is deterministic, the
    download is not (EDGAR can return slightly different bytes for the same
    immutable accession between fetches). The document is fetched from EDGAR only
    when it is missing, when a newer accession is published, or when
    refresh_raw=True forces a re-download.

    When the latest EDGAR accession matches what we already processed (and we are
    not forcing or refreshing), skip the work and return existing meta,
    changed=False."""
    meta_path = config.PROCESSED_DIR / f"{ticker}.meta.json"
    text_path = config.PROCESSED_DIR / f"{ticker}.txt"
    raw_path = config.RAW_DIR / f"{ticker}.html"
    cached_meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    )

    # Deterministic re-clean: reuse cached raw HTML and its accession, no network
    # for the document. This is the rebuild-after-cleaner-change path.
    reclean_from_cache = (
        force and not refresh_raw and raw_path.exists() and cached_meta is not None
    )

    if reclean_from_cache:
        acc = cached_meta["accession"]
        date = cached_meta["filing_date"]
        url = cached_meta["source_url"]
        html = raw_path.read_text(encoding="utf-8")
    else:
        acc, doc, date = latest_filing(cik)
        url = _ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc)
        # Incremental skip: same accession already processed, nothing forced.
        if (
            not force
            and not refresh_raw
            and cached_meta is not None
            and text_path.exists()
            and raw_path.exists()
            and cached_meta.get("accession") == acc
        ):
            return FilingMeta(**cached_meta), False
        html = _get(url)
        raw_path.write_text(html, encoding="utf-8")

    text = _clean_html(html)
    text_path.write_text(text, encoding="utf-8")

    meta = FilingMeta(
        ticker=ticker,
        company=company,
        cik=cik,
        form=config.FILING_FORM,
        filing_date=date,
        accession=acc,
        source_url=url,
        char_count=len(text),
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta, True


def ingest_all(
    companies: dict[str, str] | None = None,
    *,
    force: bool = False,
    refresh_raw: bool = False,
) -> dict[str, tuple[FilingMeta, bool]]:
    """Returns {ticker: (meta, changed)} so callers can rebuild only what moved."""
    companies = companies or config.COMPANIES
    ciks = resolve_ciks(list(companies))
    out = {}
    for ticker, company in companies.items():
        print(f"[ingest] {ticker} ({company}) CIK={ciks[ticker]} ...", flush=True)
        meta, changed = ingest_company(
            ticker, company, ciks[ticker], force=force, refresh_raw=refresh_raw
        )
        status = "updated" if changed else "unchanged (skipped download)"
        print(
            f"         {meta.form} {meta.filing_date}  "
            f"{meta.char_count:,} chars  [{status}]",
            flush=True,
        )
        out[ticker] = (meta, changed)
    return out


def load_processed() -> dict[str, dict]:
    """Return {ticker: {"text": ..., "meta": {...}}} for everything ingested."""
    out = {}
    for meta_path in sorted(config.PROCESSED_DIR.glob("*.meta.json")):
        ticker = meta_path.name.split(".")[0]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = (config.PROCESSED_DIR / f"{ticker}.txt").read_text(encoding="utf-8")
        out[ticker] = {"text": text, "meta": meta}
    return out


if __name__ == "__main__":
    ingest_all()
