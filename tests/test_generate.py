from src.chunking import Chunk
from src.generate import _normalize_citations, _order_for_context
from src.guardrails import audit_citations


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"AAPL-fixed-{i:04d}",
        ticker="AAPL",
        company="Apple Inc.",
        source_url="https://example.com",
        strategy="fixed",
        text=f"chunk {i}",
        token_count=2,
        position=i,
    )


def test_order_for_context_bookends_strongest_chunks():
    """Best chunk first, second-best last, weaker ones buried in the middle."""
    ranked = [_chunk(i) for i in range(5)]  # 0 = most relevant ... 4 = least
    out = _order_for_context(ranked)
    ids = [c.position for c in out]
    assert ids == [0, 2, 4, 3, 1]
    assert ids[0] == 0   # strongest at the very start
    assert ids[-1] == 1  # second-strongest at the very end


def test_order_for_context_is_a_permutation():
    ranked = [_chunk(i) for i in range(7)]
    out = _order_for_context(ranked)
    assert sorted(c.position for c in out) == list(range(7))


def test_fullwidth_citations_are_normalized_and_audit_clean():
    # Some open-weight models (gpt-oss via Nebius) cite as 【2】; after
    # normalization the audit must see a grounded answer, not flag it ungrounded.
    raw = "Revenue was $281,724 million【2】, up from prior year【1】【3】."
    normalized = _normalize_citations(raw)
    assert "[2]" in normalized and "【" not in normalized
    audit = audit_citations(normalized, n_sources=5)
    assert audit.unique_cited == [1, 2, 3]
    assert not audit.ungrounded and not audit.dangling
