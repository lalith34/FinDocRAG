from src.chunking import chunk_document, detect_section, fixed_chunks, n_tokens


def test_fixed_chunks_window_and_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = fixed_chunks(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(n_tokens(c) <= 100 for c in chunks)


def test_fixed_chunks_empty():
    assert fixed_chunks("") == []


def test_fixed_chunks_short_text_single_chunk():
    chunks = fixed_chunks("hello world", size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_document_ids_and_metadata():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_document(
        ticker="AAPL",
        company="Apple Inc.",
        source_url="https://example.com",
        text=text,
        strategy="fixed",
    )
    assert chunks
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"AAPL-fixed-{i:04d}"
        assert c.ticker == "AAPL"
        assert c.strategy == "fixed"
        assert c.position == i
        assert c.token_count == n_tokens(c.text)


def test_detect_section_named_and_fallback():
    assert detect_section("Item 1A. Risk Factors\nWe face competition...") == (
        "Item 1A — Risk Factors"
    )
    assert detect_section("ITEM 7. MANAGEMENT'S DISCUSSION") == (
        "Item 7 — Management's Discussion and Analysis"
    )
    # An item number with no canonical name still yields a usable label.
    assert detect_section("Item 16. Form 10-K Summary") == "Item 16"
    # "item" mid-sentence is not a header (must be at a line start).
    assert detect_section("Each line item 1a was reviewed.") is None
    assert detect_section("Just some prose with no header.") is None


def test_chunk_document_carries_section_forward():
    # Header line, then enough prose to spill into a second chunk with no header
    # of its own — it must inherit Risk Factors.
    body = " ".join(f"word{i}" for i in range(400))
    text = f"Item 1A. Risk Factors\n{body}"
    chunks = chunk_document(
        ticker="AAPL",
        company="Apple Inc.",
        source_url="https://example.com",
        text=text,
        strategy="fixed",
    )
    assert len(chunks) > 1
    assert all(c.section == "Item 1A — Risk Factors" for c in chunks)


def test_chunk_document_front_matter_before_first_item():
    text = "Cover page and table of contents.\n" + " ".join(
        f"word{i}" for i in range(400)
    )
    chunks = chunk_document(
        ticker="AAPL",
        company="Apple Inc.",
        source_url="https://example.com",
        text=text,
        strategy="fixed",
    )
    assert chunks[0].section == "Front Matter"
