from pathlib import Path
from student_utils import datasets as ds
from student_utils import pisces_adapter as pa

ROOT = Path(__file__).resolve().parents[1]


def test_seed_datasets_validate():
    for name in ["gaia_sycophancy_seed.jsonl", "itay_reliability_seed.jsonl",
                 "general_behavior_controls_seed.jsonl"]:
        df = ds.load_eval_dataset(ROOT / "data" / "student_evals" / name)
        assert len(df) >= 8


def test_placeholder_feature_sets_validate():
    for name in ["gaia_example_features.json", "itay_example_features.json"]:
        fs = pa.load_feature_set(ROOT / "features" / "student_feature_sets" / name)
        assert fs["features"] == []
