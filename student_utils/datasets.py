"""Small helpers for building and validating an eval DataFrame in-notebook.

Model-free: imports only stdlib + pandas. There is no file loading here on
purpose -- you build your probe set inline (make_eval_dataframe) or load a real
dataset yourself (e.g. with `datasets.load_dataset(...)`) and turn it into a
DataFrame with a 'prompt' column.

Convention: an eval DataFrame has one row per example with at least a 'prompt'
column, usually an 'id', and often a 'kind' column ('target' vs 'control') so
summaries can separate the behaviour you target from the cases you must NOT break.
"""
import pandas as pd

REQUIRED_FIELDS = ("id", "prompt")
KNOWN_FIELDS = ("id", "prompt", "split", "kind", "category", "ideal_behavior", "notes")


def _as_rows(rows_or_df):
    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df.to_dict("records")
    return list(rows_or_df)


def validate_eval_dataset(rows_or_df, require_ids: bool = True) -> None:
    """Validate eval rows (a list of dicts or a DataFrame). Raises ValueError.

    Checks: non-empty; every row has a non-empty 'prompt'; if require_ids, every
    row has a non-empty 'id' and ids are unique.
    """
    rows = _as_rows(rows_or_df)
    if not rows:
        raise ValueError("eval dataset is empty")

    seen_ids = []
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"row {i}: missing or empty 'prompt' field (got {prompt!r})")
        if require_ids:
            rid = row.get("id")
            if rid is None or (isinstance(rid, str) and not rid.strip()):
                raise ValueError(f"row {i}: missing or empty 'id' field (set require_ids=False to skip)")
            seen_ids.append(rid)

    if require_ids and len(set(seen_ids)) != len(seen_ids):
        dupes = sorted({x for x in seen_ids if seen_ids.count(x) > 1})
        raise ValueError(f"duplicate id(s) in dataset: {dupes}")


def make_eval_dataframe(prompts, kinds=None, ids=None) -> pd.DataFrame:
    """Build a validated eval DataFrame from inline lists.

    prompts: list of prompt strings.
    kinds:   optional list (same length) labelling each row, e.g. 'target' or
             'control'. If given, a 'kind' column is added.
    ids:     optional list of ids; defaults to '0', '1', ... .

    Returns a DataFrame with columns id, prompt (and kind if provided).
    """
    prompts = list(prompts)
    ids = list(ids) if ids is not None else [str(i) for i in range(len(prompts))]
    if len(ids) != len(prompts):
        raise ValueError(f"ids ({len(ids)}) and prompts ({len(prompts)}) length mismatch")
    data = {"id": ids, "prompt": prompts}
    if kinds is not None:
        kinds = list(kinds)
        if len(kinds) != len(prompts):
            raise ValueError(f"kinds ({len(kinds)}) and prompts ({len(prompts)}) length mismatch")
        data["kind"] = kinds
    df = pd.DataFrame(data)
    validate_eval_dataset(df, require_ids=True)
    return df


def dataset_to_prompts(df) -> list:
    """Return the list of prompts from an eval DataFrame, in row order."""
    if "prompt" not in df.columns:
        raise ValueError("DataFrame has no 'prompt' column")
    return df["prompt"].tolist()
