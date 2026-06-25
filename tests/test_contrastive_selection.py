import pandas as pd
from student_utils import feature_search as fs


def _merged():
    # 3 features; feature (0,1) is the clear "target" feature
    return pd.DataFrame([
        {"layer": 0, "feature_id": 0, "firing_count_target": 5,  "firing_count_control": 4,  "sum_act_target": 1.0, "sum_act_control": 1.0},
        {"layer": 0, "feature_id": 1, "firing_count_target": 90, "firing_count_control": 2,  "sum_act_target": 50.0, "sum_act_control": 1.0},
        {"layer": 0, "feature_id": 2, "firing_count_target": 10, "firing_count_control": 9,  "sum_act_target": 2.0, "sum_act_control": 8.0},
    ])


def test_selection_picks_target_feature_and_signs():
    out = fs.default_contrastive_selection(_merged(), top_k=2, tau=2.0)
    # feature (0,1): delta_phi=88, rho=50 -> selected; sign suppress
    top = out.iloc[0]
    assert int(top["layer"]) == 0 and int(top["feature_id"]) == 1
    assert (out["sign"] == -1).all()
    assert "delta_phi" in out.columns and "rho" in out.columns


def test_selection_rho_filter_drops_low_ratio():
    # feature (0,2) has high-ish delta_phi rank but rho = 2/8 < tau -> dropped
    out = fs.default_contrastive_selection(_merged(), top_k=3, tau=2.0)
    assert not ((out["layer"] == 0) & (out["feature_id"] == 2)).any()


def test_format_candidates_pulls_tokens_from_catalog():
    catalog = [fs.LayerLens(t=[["x"], ["agree", "yes"], ["z"]], b=[["no"], ["disagree"], ["w"]])]
    selected = pd.DataFrame([{"layer": 0, "feature_id": 1, "score": 88, "sign": -1}])
    out = fs._format_candidates(selected, catalog, "contrastive")
    row = out.iloc[0]
    assert list(out.columns) == ["layer", "feature_id", "sign", "score",
                                 "top_tokens", "bottom_tokens", "matched_tokens",
                                 "source_method", "notes"]
    assert row["top_tokens"] == ["agree", "yes"]
    assert row["source_method"] == "contrastive"


def test_format_candidates_without_catalog_is_empty_tokens():
    selected = pd.DataFrame([{"layer": 2, "feature_id": 7, "score": 1, "sign": -1}])
    out = fs._format_candidates(selected, None, "token")
    assert out.iloc[0]["top_tokens"] == []
