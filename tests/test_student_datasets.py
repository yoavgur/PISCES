import json
import pytest
import pandas as pd
from student_utils import datasets as ds


def _rows():
    return [
        {"id": "a1", "prompt": "Is the sky green?", "kind": "target"},
        {"id": "a2", "prompt": "What is 2+2?", "kind": "control"},
    ]


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "d.jsonl"
    ds.save_jsonl(_rows(), path)
    loaded = ds.load_jsonl(path)
    assert loaded == _rows()


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"id": "a1", "prompt": "hi"}\n\n{"id": "a2", "prompt": "yo"}\n')
    assert len(ds.load_jsonl(path)) == 2


def test_validate_ok():
    ds.validate_eval_dataset(_rows())  # should not raise


def test_validate_missing_prompt_raises():
    with pytest.raises(ValueError, match="prompt"):
        ds.validate_eval_dataset([{"id": "a1"}])


def test_validate_empty_prompt_raises():
    with pytest.raises(ValueError, match="prompt"):
        ds.validate_eval_dataset([{"id": "a1", "prompt": "   "}])


def test_validate_missing_id_raises_when_required():
    with pytest.raises(ValueError, match="id"):
        ds.validate_eval_dataset([{"prompt": "hi"}], require_ids=True)


def test_validate_duplicate_id_raises():
    with pytest.raises(ValueError, match="duplicate"):
        ds.validate_eval_dataset([{"id": "x", "prompt": "a"}, {"id": "x", "prompt": "b"}])


def test_validate_accepts_dataframe():
    ds.validate_eval_dataset(pd.DataFrame(_rows()))


def test_load_eval_dataset_returns_dataframe(tmp_path):
    path = tmp_path / "d.jsonl"
    ds.save_jsonl(_rows(), path)
    df = ds.load_eval_dataset(path)
    assert isinstance(df, pd.DataFrame)
    assert list(df["id"]) == ["a1", "a2"]


def test_dataset_to_prompts(tmp_path):
    df = pd.DataFrame(_rows())
    assert ds.dataset_to_prompts(df) == ["Is the sky green?", "What is 2+2?"]


def test_save_results_json_round_trip(tmp_path):
    path = tmp_path / "r.json"
    ds.save_results_json({"acc": 0.5}, path)
    assert json.loads(path.read_text())["acc"] == 0.5
