"""Unit tests for the PageIndex retriever's pure logic — tree rendering and the
navigation post-processing (id validation + top-k cap + path formatting). The LLM
navigation call is monkeypatched, so these run offline and deterministically."""
from src import pageindex
from src.chunking import Chunk


def _chunk(cid: str, section: str, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=cid, ticker="MSFT", company="Microsoft Corporation",
        source_url="u", strategy="fixed", text=text, token_count=1,
        position=0, section=section,
    )


def test_render_tree_lists_items_and_leaves():
    tree = {
        "Item 1 — Business": [{"id": "MSFT-fixed-0001", "summary": "segments"}],
        "Item 1A — Risk Factors": [{"id": "MSFT-fixed-0019", "summary": "cyber risk"}],
    }
    rendered = pageindex.render_tree(tree)
    assert "## Item 1 — Business" in rendered
    assert "## Item 1A — Risk Factors" in rendered
    assert "- MSFT-fixed-0001: segments" in rendered
    assert "- MSFT-fixed-0019: cyber risk" in rendered


def _patch_retriever(monkeypatch, nav_return):
    """A retriever whose tree/by_id is synthetic and whose nav call is stubbed."""
    retr = pageindex.PageIndexRetriever()
    by_id = {f"MSFT-fixed-000{i}": _chunk(f"MSFT-fixed-000{i}", "Item 1 — Business")
             for i in range(4)}
    retr._cache["MSFT"] = ({}, "rendered tree", by_id)
    monkeypatch.setattr(pageindex, "_nav_call", lambda q, r: nav_return)
    return retr


def test_navigate_drops_hallucinated_ids(monkeypatch):
    retr = _patch_retriever(monkeypatch, {
        "section": "Item 1 — Business",
        "reasoning": "because",
        "node_ids": ["MSFT-fixed-0001", "NOT-A-REAL-ID", "MSFT-fixed-0002"],
    })
    chunks, path, reasoning = retr.retrieve("q", "MSFT")
    ids = [c.chunk_id for c in chunks]
    assert ids == ["MSFT-fixed-0001", "MSFT-fixed-0002"]  # bogus id filtered out
    assert "NOT-A-REAL-ID" not in path
    assert reasoning == "because"


def test_navigate_caps_at_k(monkeypatch):
    retr = _patch_retriever(monkeypatch, {
        "section": "Item 1 — Business",
        "node_ids": [f"MSFT-fixed-000{i}" for i in range(4)],
    })
    chunks, _path, _r = retr.retrieve("q", "MSFT", k=2)
    assert len(chunks) == 2


def test_navigate_path_shows_section_and_ids(monkeypatch):
    retr = _patch_retriever(monkeypatch, {
        "section": "Item 1 — Business",
        "node_ids": ["MSFT-fixed-0001"],
    })
    _chunks, path, _r = retr.retrieve("q", "MSFT")
    assert path == "Item 1 — Business → MSFT-fixed-0001"


def test_navigate_empty_selection_yields_none_path(monkeypatch):
    retr = _patch_retriever(monkeypatch, {"section": "Item 1 — Business", "node_ids": []})
    chunks, path, _r = retr.retrieve("q", "MSFT")
    assert chunks == []
    assert path.endswith("(none)")
