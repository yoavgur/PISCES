import pytest
from student_utils import generation as gen


def test_make_generation_dataframe_basic():
    df = gen.make_generation_dataframe(["p1", "p2"], ["r1", "r2"])
    assert list(df.columns) == ["id", "prompt", "response"]
    assert list(df["id"]) == ["0", "1"]
    assert list(df["response"]) == ["r1", "r2"]


def test_make_generation_dataframe_with_ids_and_metadata():
    df = gen.make_generation_dataframe(["p1"], ["r1"], ids=["x"], metadata={"phase": "baseline"})
    assert df.loc[0, "id"] == "x"
    assert df.loc[0, "phase"] == "baseline"


def test_make_generation_dataframe_length_mismatch_raises():
    with pytest.raises(ValueError):
        gen.make_generation_dataframe(["p1", "p2"], ["r1"])


def test_compare_generations_dataframe_changed_flag():
    df = gen.compare_generations_dataframe(["p1", "p2"], ["same", "old"], ["same", "new"])
    assert list(df.columns) == ["id", "prompt", "baseline_response", "edited_response", "changed"]
    assert list(df["changed"]) == [False, True]
