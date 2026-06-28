import pytest
import pandas as pd
from student_utils import datasets as ds


def _rows():
    return [
        {"id": "a1", "prompt": "Is the sky green?", "kind": "target"},
        {"id": "a2", "prompt": "What is 2+2?", "kind": "control"},
    ]


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


def test_dataset_to_prompts():
    df = pd.DataFrame(_rows())
    assert ds.dataset_to_prompts(df) == ["Is the sky green?", "What is 2+2?"]


def test_make_eval_dataframe_basic():
    df = ds.make_eval_dataframe(["p1", "p2"], kinds=["target", "control"])
    assert list(df.columns) == ["id", "prompt", "kind"]
    assert list(df["id"]) == ["0", "1"]
    assert list(df["kind"]) == ["target", "control"]


def test_make_eval_dataframe_no_kinds():
    df = ds.make_eval_dataframe(["p1"])
    assert "kind" not in df.columns


def test_make_eval_dataframe_length_mismatch_raises():
    with pytest.raises(ValueError, match="kinds"):
        ds.make_eval_dataframe(["p1", "p2"], kinds=["target"])
