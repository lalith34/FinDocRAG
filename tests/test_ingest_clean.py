"""Cleaning guarantees: processed text and the chunk snapshots must not carry
inline-XBRL tagging soup (tag contexts, lone CIK/dates, taxonomy URLs), while
real financial tables (pipe-delimited income statements) survive intact."""
import json
import re

import config
from src.ingest import strip_xbrl_artifacts

# A line is XBRL tagging noise if, stripped, it is a single whitespace-free token
# of one of these unambiguous shapes. Mirrors src.ingest._XBRL_LINE_PATTERNS.
_XBRL_PATTERNS = [
    re.compile(r"^[A-Za-z][\w-]*:[\w.\-]+$"),   # us-gaap:/dei:/srt:/<ticker>:/...Member
    re.compile(r"^\d{10}$"),                     # bare CIK
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),          # lone ISO context date
    re.compile(r"^P\d+[YMWD]$"),                 # ISO-8601 duration
    re.compile(r"^https?://\S+$"),               # taxonomy URL
]


def _xbrl_lines(text: str) -> list[str]:
    return [
        ln for ln in text.splitlines()
        if any(p.match(ln.strip()) for p in _XBRL_PATTERNS)
    ]


def test_strip_removes_xbrl_keeps_table():
    raw = "\n".join([
        "0001045810",
        "us-gaap:RetainedEarningsMember",
        "nvda:JohnO.DabiriMember",
        "2025-01-27",
        "iso4217:USD",
        "xbrli:shares",
        "P1Y",
        "http://fasb.org/us-gaap/2025#AccruedLiabilitiesCurrent",
        "Revenue | $ | 215,938 | $ | 130,497 | $ | 60,922",  # income statement row
        "(In millions, except per share data)",
        "We generated more than 70% of total revenues from advertising.",
    ])
    out = strip_xbrl_artifacts(raw)

    assert _xbrl_lines(out) == []
    # None of the unambiguous XBRL tokens survive.
    for tok in ("0001045810", "us-gaap:RetainedEarningsMember", "2025-01-27",
                "iso4217:USD", "P1Y", "http://fasb.org"):
        assert tok not in out
    # Real financial content must survive verbatim.
    assert "Revenue | $ | 215,938 | $ | 130,497 | $ | 60,922" in out
    assert "(In millions, except per share data)" in out
    assert "total revenues from advertising" in out


def test_strip_keeps_legit_year_and_date_in_table_rows():
    # A pipe-delimited row containing a year/date is table data, not a lone tag.
    raw = "Fiscal year | 2025 | 2024\nPeriod ended | 2025-01-27 | 2024-01-28"
    out = strip_xbrl_artifacts(raw)
    assert "2025" in out and "2025-01-27" in out
    assert out.count("\n") == 1  # both rows retained


def test_processed_files_have_no_xbrl_noise():
    files = sorted(config.PROCESSED_DIR.glob("*.txt"))
    assert files, "no processed filings found; run ingestion first"
    for f in files:
        leftover = _xbrl_lines(f.read_text(encoding="utf-8"))
        assert not leftover, f"{f.name} still has XBRL lines: {leftover[:5]}"


def test_chunk_snapshots_have_no_xbrl_noise():
    for strategy in config.STRATEGIES:
        snap = config.INDEX_DIR / strategy / "chunks.json"
        if not snap.exists():
            continue
        chunks = json.loads(snap.read_text(encoding="utf-8"))
        for c in chunks:
            leftover = _xbrl_lines(c["text"])
            assert not leftover, (
                f"{strategy} chunk {c.get('chunk_id')} has XBRL lines: {leftover[:5]}"
            )
