import pandas as pd
import pytest
from student_utils import scoring


def _toy_scorer(prompt, response):
    """A tiny stand-in scorer to exercise the harness (NOT shipped to students)."""
    agrees = "you are right" in (response or "").lower()
    return {"agrees": agrees, "target_bad_behavior": 1.0 if agrees else 0.0, "notes": ""}


def test_apply_scorer_adds_columns():
    df = pd.DataFrame([
        {"id": "1", "prompt": "p", "response": "Yes, you are right!", "kind": "target"},
        {"id": "2", "prompt": "p", "response": "It is 4.", "kind": "control"},
    ])
    out = scoring.apply_scorer(df, _toy_scorer)
    assert {"agrees", "target_bad_behavior", "notes"}.issubset(set(out.columns))
    assert len(out) == 2


def test_apply_scorer_requires_prompt_and_response():
    with pytest.raises(ValueError, match="response"):
        scoring.apply_scorer(pd.DataFrame([{"prompt": "p"}]), _toy_scorer)


def test_summarize_scores_groups_by_kind_and_casts_bools():
    df = pd.DataFrame([
        {"id": "1", "prompt": "p", "response": "Yes, you are right!", "kind": "target"},
        {"id": "2", "prompt": "p", "response": "no", "kind": "control"},
    ])
    scored = scoring.apply_scorer(df, _toy_scorer)
    summary = scoring.summarize_scores(scored)
    assert "kind" in summary.columns
    assert set(summary["kind"]) == {"target", "control"}
    tb = dict(zip(summary["kind"], summary["target_bad_behavior"]))
    assert tb["target"] == 1.0 and tb["control"] == 0.0


def test_summarize_scores_without_kind_single_row():
    df = pd.DataFrame([{"id": "1", "prompt": "p", "response": "you are right"}])
    scored = scoring.apply_scorer(df, _toy_scorer)
    summary = scoring.summarize_scores(scored)
    assert len(summary) == 1
    assert "target_bad_behavior" in summary.columns
