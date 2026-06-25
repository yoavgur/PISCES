"""JSONL eval-dataset loading, validation, and small DataFrame helpers.

Model-free: imports only stdlib + pandas.
"""
import json
from pathlib import Path

import pandas as pd

REQUIRED_FIELDS = ("id", "prompt")
KNOWN_FIELDS = ("id", "prompt", "split", "kind", "category", "ideal_behavior", "notes")


def load_jsonl(path) -> list:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON on line {i}: {e}") from e
    return rows


def save_jsonl(rows, path) -> None:
    """Write a list of dicts to a JSONL file (one compact JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_rows(rows_or_df):
    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df.to_dict("records")
    return list(rows_or_df)


def validate_eval_dataset(rows_or_df, require_ids: bool = True) -> None:
    """Validate eval-dataset rows. Raises ValueError with a helpful message.

    Checks: non-empty; every row has a non-empty `prompt`; if require_ids, every
    row has a non-empty `id` and ids are unique.
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


def load_eval_dataset(path) -> pd.DataFrame:
    """Load + validate a JSONL eval dataset into a DataFrame."""
    rows = load_jsonl(path)
    validate_eval_dataset(rows, require_ids=True)
    return pd.DataFrame(rows)


def dataset_to_prompts(df) -> list:
    """Return the list of prompts from an eval DataFrame, in row order."""
    if "prompt" not in df.columns:
        raise ValueError("DataFrame has no 'prompt' column")
    return df["prompt"].tolist()


def save_generations_csv(df, path) -> None:
    """Save a generations/results DataFrame to CSV (utf-8, no index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def save_results_json(results, path) -> None:
    """Save a results dict/list to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
