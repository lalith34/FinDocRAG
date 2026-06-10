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

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


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


def _get(url: str, *, as_json: bool = False, retries: int = 3):
    """GET with SEC-friendly throttling and basic retry."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=60)
            resp.raise_for_status()
            time.sleep(0.2)  # stay under SEC's ~10 req/s guidance
            return resp.json() if as_json else resp.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


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
    return "\n".join(ln for ln in lines if ln).strip()


def ingest_company(ticker: str, company: str, cik: int) -> FilingMeta:
    acc, doc, date = latest_filing(cik)
    url = _ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc)
    html = _get(url)

    (config.RAW_DIR / f"{ticker}.html").write_text(html, encoding="utf-8")
    text = _clean_html(html)
    (config.PROCESSED_DIR / f"{ticker}.txt").write_text(text, encoding="utf-8")

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
    (config.PROCESSED_DIR / f"{ticker}.meta.json").write_text(
        json.dumps(asdict(meta), indent=2), encoding="utf-8"
    )
    return meta


def ingest_all(companies: dict[str, str] | None = None) -> list[FilingMeta]:
    companies = companies or config.COMPANIES
    ciks = resolve_ciks(list(companies))
    metas = []
    for ticker, company in companies.items():
        print(f"[ingest] {ticker} ({company}) CIK={ciks[ticker]} ...", flush=True)
        meta = ingest_company(ticker, company, ciks[ticker])
        print(
            f"         {meta.form} {meta.filing_date}  "
            f"{meta.char_count:,} chars  -> data/processed/{ticker}.txt",
            flush=True,
        )
        metas.append(meta)
    return metas


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
