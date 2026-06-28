import pandas as pd
import pytest
from student_utils import feature_search as fs


def test_format_candidates_pulls_tokens_from_catalog():
    catalog = [fs.LayerLens(t=[["x"], ["agree", "yes"], ["z"]], b=[["no"], ["disagree"], ["w"]])]
    selected = pd.DataFrame([{"layer": 0, "feature_id": 1, "score": 88, "sign": -1}])
    out = fs._format_candidates(selected, catalog, "contrastive")
    row = out.iloc[0]
    assert list(out.columns) == fs.CANDIDATE_COLUMNS
    assert row["top_tokens"] == ["agree", "yes"]
    assert row["source_method"] == "contrastive"


def test_format_candidates_without_catalog_is_empty_tokens():
    selected = pd.DataFrame([{"layer": 2, "feature_id": 7, "score": 1, "sign": -1}])
    out = fs._format_candidates(selected, None, "token")
    assert out.iloc[0]["top_tokens"] == []


def test_format_candidates_carries_frac_firing_when_present():
    catalog = [fs.LayerLens(t=[["x"], ["agree", "yes"], ["z"]], b=[["no"], ["disagree"], ["w"]])]
    selected = pd.DataFrame([
        {"layer": 0, "feature_id": 1, "score": 88, "sign": -1,
         "frac_firing_target": 0.5, "frac_firing_control": 0.1},
    ])
    out = fs._format_candidates(selected, catalog, "contrastive")
    row = out.iloc[0]
    assert "frac_firing_target" in out.columns and "frac_firing_control" in out.columns
    assert row["frac_firing_target"] == 0.5 and row["frac_firing_control"] == 0.1


def test_format_candidates_no_firing_columns_when_absent():
    selected = pd.DataFrame([{"layer": 2, "feature_id": 7, "score": 1, "sign": -1}])
    out = fs._format_candidates(selected, None, "token")
    assert list(out.columns) == fs.CANDIDATE_COLUMNS
    assert "frac_firing_target" not in out.columns


def test_format_candidates_handles_nonrange_index():
    """Regression: a custom select_fn that does NOT reset_index returns rows with
    an arbitrary index. _format_candidates must still work (previously raised
    KeyError: 0 -- the cell-14 crash)."""
    catalog = [fs.LayerLens(t=[["x"], ["agree"], ["z"]], b=[["no"], ["dis"], ["w"]])]
    selected = pd.DataFrame(
        [{"layer": 0, "feature_id": 1, "score": 88, "sign": -1,
          "frac_firing_target": 0.5, "frac_firing_control": 0.1},
         {"layer": 0, "feature_id": 2, "score": 10, "sign": -1,
          "frac_firing_target": 0.3, "frac_firing_control": 0.2}],
        index=[5, 9],   # non-RangeIndex, as a contrastive select_fn would return
    )
    out = fs._format_candidates(selected, catalog, "contrastive")
    assert len(out) == 2
    assert list(out["frac_firing_target"]) == [0.5, 0.3]
    assert list(out["frac_firing_control"]) == [0.1, 0.2]


def test_matched_tokens_uses_full_lists_not_truncated_display():
    """matched_tokens must reflect the full top OR bottom lists, even when the hit
    is on the bottom side and beyond the truncated display window."""
    catalog = [fs.LayerLens(
        t=[["a", "b", "c", "TOP_HIT"]],     # TOP_HIT beyond a small display cut
        b=[["x", "BOTTOM_HIT"]],
    )]
    selected = pd.DataFrame([{"layer": 0, "feature_id": 0, "sign": -1, "score": 1}])
    out = fs._format_candidates(selected, catalog, "token",
                                tokens=["TOP_HIT", "BOTTOM_HIT", "nope"], top_k_tokens=2)
    row = out.iloc[0]
    assert set(row["matched_tokens"]) == {"TOP_HIT", "BOTTOM_HIT"}
    assert row["top_tokens"] == ["a", "b"]   # display truncated to 2


def test_find_contrastive_features_requires_select_fn():
    with pytest.raises(ValueError, match="select_fn"):
        fs.find_contrastive_features(["t"], ["c"], model=None, select_fn=None)
