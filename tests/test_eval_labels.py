from src.chunking import Chunk
from src.eval_utils import is_relevant, rank_metrics


def make_chunk(ticker: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{ticker}-fixed-0000",
        ticker=ticker,
        company=ticker,
        source_url="https://example.com",
        strategy="fixed",
        text=text,
        token_count=10,
        position=0,
    )


QUERY = {
    "expected_tickers": ["AAPL"],
    "must_contain": ["net sales"],
}


def test_relevant_requires_ticker_and_keyword():
    assert is_relevant(make_chunk("AAPL", "Total net sales were $400B"), QUERY)
    assert not is_relevant(make_chunk("MSFT", "Total net sales were $200B"), QUERY)
    assert not is_relevant(make_chunk("AAPL", "Risk factors include..."), QUERY)


def test_keyword_match_is_case_insensitive():
    assert is_relevant(make_chunk("AAPL", "TOTAL NET SALES"), QUERY)


def test_rank_metrics_hit_at_first_rank():
    chunks = [
        make_chunk("AAPL", "net sales table"),
        make_chunk("MSFT", "unrelated"),
    ]
    m = rank_metrics(chunks, QUERY, k=2)
    assert m["hit"] == 1.0
    assert m["mrr"] == 1.0
    assert m["precision"] == 0.5


def test_rank_metrics_hit_at_second_rank():
    chunks = [
        make_chunk("MSFT", "unrelated"),
        make_chunk("AAPL", "net sales table"),
    ]
    m = rank_metrics(chunks, QUERY, k=2)
    assert m["hit"] == 1.0
    assert m["mrr"] == 0.5


def test_rank_metrics_no_hit():
    chunks = [make_chunk("MSFT", "unrelated")] * 3
    m = rank_metrics(chunks, QUERY, k=3)
    assert m == {"hit": 0.0, "mrr": 0.0, "precision": 0.0, "ndcg": 0.0}


def test_ndcg_rewards_higher_ranking():
    """Same relevant chunk earns a higher NDCG when it ranks first vs second."""
    first = rank_metrics(
        [make_chunk("AAPL", "net sales"), make_chunk("MSFT", "x")], QUERY, k=2
    )
    second = rank_metrics(
        [make_chunk("MSFT", "x"), make_chunk("AAPL", "net sales")], QUERY, k=2
    )
    assert first["ndcg"] == 1.0          # relevant chunk at rank 1 is ideal
    assert second["ndcg"] < first["ndcg"]  # demoting it costs NDCG
