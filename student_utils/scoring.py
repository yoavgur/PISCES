"""Scoring harness: apply a scorer over an eval DataFrame and summarise results.

Model-free: imports only stdlib + pandas.

The harness is provided; the SCORERS are yours to write. A scorer is any function
`scorer_fn(prompt, response) -> dict` that returns a flat dict of metrics for one
generation (booleans/numbers, optionally a 'notes' string). Define exactly what
counts as the target behaviour for your task -- that is core research work.
Include a numeric column you care about (e.g. 'target_bad_behavior' in {0.0, 1.0})
so summarize_scores can average it across target vs control rows.
"""
import pandas as pd


def apply_scorer(df, scorer_fn):
    """Run `scorer_fn(prompt, response)` over each row; add its dict keys as columns.

    `df` must have 'prompt' and 'response' columns. Returns a new DataFrame: the
    original columns plus one column per key returned by the scorer.
    """
    for col in ("prompt", "response"):
        if col not in df.columns:
            raise ValueError(f"apply_scorer needs a '{col}' column; got {list(df.columns)}")
    scores = [scorer_fn(row["prompt"], row["response"]) for _, row in df.iterrows()]
    scored = pd.DataFrame(scores, index=df.index)
    return pd.concat([df, scored], axis=1)


def summarize_scores(scored_df):
    """Mean of the numeric/bool score columns, grouped by 'kind' if that column exists.

    Booleans are cast to floats so they average. Non-score columns (id, prompt,
    response, ...) are ignored. With a 'kind' column you get one row per kind
    (e.g. target vs control); otherwise a single summary row.
    """
    ignore = {"id", "prompt", "response", "split", "category", "ideal_behavior", "notes"}
    num = scored_df.copy()
    # cast bools to float so they aggregate
    for c in list(num.columns):
        if c not in ignore and num[c].dtype == bool:
            num[c] = num[c].astype(float)
    score_cols = [c for c in num.columns
                  if c not in ignore and pd.api.types.is_numeric_dtype(num[c])]
    if "kind" in num.columns:
        return num.groupby("kind", as_index=False)[score_cols].mean()
    return num[score_cols].mean().to_frame().T
