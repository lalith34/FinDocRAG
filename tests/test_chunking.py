from src.chunking import chunk_document, fixed_chunks, n_tokens


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
