import pytest
from student_utils import pisces_adapter as pa


def _fs(features):
    return {"name": "t", "description": "d", "features": features}


def test_validate_empty_placeholder_ok():
    pa.validate_feature_set(_fs([]))  # placeholder with no features must validate


def test_validate_good_feature_ok():
    pa.validate_feature_set(_fs([{"layer": 12, "feature_id": 3456, "sign": -1, "why": "agreement"}]))


def test_validate_missing_layer_raises():
    with pytest.raises(ValueError, match="layer"):
        pa.validate_feature_set(_fs([{"feature_id": 1, "sign": -1}]))


def test_validate_bad_sign_raises():
    with pytest.raises(ValueError, match="sign"):
        pa.validate_feature_set(_fs([{"layer": 1, "feature_id": 1, "sign": 0}]))


def test_validate_requires_features_list():
    with pytest.raises(ValueError, match="features"):
        pa.validate_feature_set({"name": "t", "description": "d"})


def test_sign_to_neg_mapping():
    assert pa.feature_dict_to_args({"layer": 1, "feature_id": 2, "sign": -1}) == (1, 2, True)
    assert pa.feature_dict_to_args({"layer": 3, "feature_id": 4, "sign": 1}) == (3, 4, False)


def test_random_feature_set_like_is_deterministic_and_in_range():
    fs = _fs([{"layer": 5, "feature_id": 10, "sign": -1, "why": "x"},
              {"layer": 9, "feature_id": 20, "sign": 1, "why": "y"}])
    a = pa.make_random_feature_set_like(fs, seed=0, n_sae_features=100)
    b = pa.make_random_feature_set_like(fs, seed=0, n_sae_features=100)
    assert a == b
    assert [f["layer"] for f in a["features"]] == [5, 9]          # same layers
    assert [f["sign"] for f in a["features"]] == [-1, 1]           # same signs
    assert all(0 <= f["feature_id"] < 100 for f in a["features"])  # ids in range
    assert a["name"].endswith("_random")
