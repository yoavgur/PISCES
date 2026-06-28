import pandas as pd
import pytest
from student_utils import reporting as rep


def test_plot_tradeoff_returns_axes():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")  # headless backend
    df = pd.DataFrame([{"target_bad": 0.2, "general_coherent": 0.9},
                       {"target_bad": 0.5, "general_coherent": 0.95}])
    ax = rep.plot_tradeoff(df, "target_bad", "general_coherent")
    assert ax.get_xlabel() == "target_bad"
    assert ax.get_ylabel() == "general_coherent"


def test_plot_contrastive_scatter_returns_axes():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    df = pd.DataFrame([{"frac_firing_control": 0.1, "frac_firing_target": 0.5},
                       {"frac_firing_control": 0.2, "frac_firing_target": 0.3}])
    ax = rep.plot_contrastive_scatter(df)
    assert ax is not None


def test_display_helpers_do_not_crash():
    df = pd.DataFrame([{"a": 1, "b": 2}])
    rep.display_before_after(df)
    rep.display_feature_table(df)
