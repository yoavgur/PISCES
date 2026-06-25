"""Run-directory management, saving, display, and simple plots.

IO helpers import only stdlib + pandas. matplotlib is imported lazily inside the
plot functions, so importing this module does not require matplotlib.
"""
import json
from pathlib import Path

import pandas as pd


def make_run_dir(base_dir="runs/student_experiments", run_name=None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        nums = []
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit():
                nums.append(int(p.name[4:]))
        run_name = f"run_{(max(nums) + 1) if nums else 1:03d}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir, metadata) -> None:
    with (Path(run_dir) / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_before_after_table(run_dir, df) -> None:
    df.to_csv(Path(run_dir) / "before_after.csv", index=False, encoding="utf-8")


def save_score_summary(run_dir, df) -> None:
    df.to_csv(Path(run_dir) / "score_summary.csv", index=False, encoding="utf-8")


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj if isinstance(obj, str) else obj.to_string())


def display_before_after(df, max_rows=20) -> None:
    _display(df.head(max_rows))


def display_feature_table(df, max_rows=50) -> None:
    _display(df.head(max_rows))


def plot_tradeoff(results_df, x_col, y_col):
    """Scatter of a sweep: target-behavior reduction vs general degradation."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(results_df[x_col], results_df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    return ax


def plot_contrastive_scatter(candidates_df, x="frac_firing_control", y="frac_firing_target"):
    """CRISP Fig-3 style: each feature by control vs target firing rate."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(candidates_df[x], candidates_df[y], s=8)
    lim = max(candidates_df[x].max(), candidates_df[y].max())
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)  # diagonal = "shared" features
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Target vs control feature firing (upper-left = target features)")
    return ax
