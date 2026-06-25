import json
import pandas as pd
from student_utils import reporting as rep


def test_make_run_dir_autoincrements(tmp_path):
    base = tmp_path / "runs"
    d1 = rep.make_run_dir(base_dir=base)
    d2 = rep.make_run_dir(base_dir=base)
    assert d1.name == "run_001" and d2.name == "run_002"
    assert d1.is_dir() and d2.is_dir()


def test_make_run_dir_named(tmp_path):
    d = rep.make_run_dir(base_dir=tmp_path / "runs", run_name="gaia_v1")
    assert d.name == "gaia_v1" and d.is_dir()


def test_save_run_metadata(tmp_path):
    d = rep.make_run_dir(base_dir=tmp_path / "runs", run_name="r")
    rep.save_run_metadata(d, {"tau": 0.9, "mu": 8.0})
    assert json.loads((d / "metadata.json").read_text())["tau"] == 0.9


def test_save_tables(tmp_path):
    d = rep.make_run_dir(base_dir=tmp_path / "runs", run_name="r")
    rep.save_before_after_table(d, pd.DataFrame([{"id": "1", "baseline_response": "a", "edited_response": "b"}]))
    rep.save_score_summary(d, pd.DataFrame([{"kind": "target", "target_bad_behavior": 0.5}]))
    assert (d / "before_after.csv").exists()
    assert (d / "score_summary.csv").exists()
