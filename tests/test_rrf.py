from src.vectorstore import rrf_fuse


def test_rrf_top_of_both_lists_wins():
    fused = rrf_fuse([["a", "b", "c"], ["a", "c", "b"]], 3)
    assert fused[0][0] == "a"
    assert len(fused) == 3


def test_rrf_dedup():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], 10)
    ids = [cid for cid, _ in fused]
    assert sorted(ids) == ["a", "b"]
    assert len(ids) == len(set(ids))


def test_rrf_k_truncates():
    fused = rrf_fuse([["a", "b", "c", "d"]], 2)
    assert len(fused) == 2
    assert fused[0][0] == "a"


def test_rrf_single_list_preserves_order():
    fused = rrf_fuse([["x", "y", "z"]], 3)
    assert [cid for cid, _ in fused] == ["x", "y", "z"]


def test_rrf_scores_decrease_with_rank():
    fused = rrf_fuse([["a", "b", "c"]], 3)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_item_in_both_lists_beats_single_list_item():
    fused = rrf_fuse([["a", "b"], ["b", "c"]], 3)
    assert fused[0][0] == "b"
