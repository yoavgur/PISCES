"""In-notebook display and simple plots.

No disk output: results live in the notebook as cell outputs and figures.
matplotlib is imported lazily inside the plot functions, so importing this module
does not require matplotlib.
"""
import pandas as pd


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj if isinstance(obj, str) else obj.to_string())


def display_before_after(df, max_rows=20) -> None:
    """Show the first rows of a before/after comparison table."""
    _display(df.head(max_rows))


def display_feature_table(df, max_rows=50) -> None:
    """Show the first rows of a feature-candidate table."""
    _display(df.head(max_rows))


def plot_tradeoff(results_df, x_col, y_col):
    """Scatter of a sweep: e.g. target-behaviour reduction vs general degradation.

    Each point is one (tau, mu) run. Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(results_df[x_col], results_df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    return ax


def plot_contrastive_scatter(candidates_df, x="frac_firing_control", y="frac_firing_target"):
    """CRISP Fig-3 style: each feature by its control vs target firing rate.

    Points in the upper-left fire much more on target than control (candidate
    behaviour features). The dashed diagonal marks "fires equally on both".
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(candidates_df[x], candidates_df[y], s=8)
    lim = max(candidates_df[x].max(), candidates_df[y].max())
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)  # diagonal = "shared" features
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Target vs control feature firing (upper-left = target features)")
    return ax
