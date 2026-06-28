# Student notebooks

Run order:
1. `00_intro_and_sanity_edit.ipynb` — load the model, see baseline answers, run the Harry Potter sanity edit (given), and read the helpers.
2. `gaia/01_sycophancy_baseline.ipynb` or `itay/01_reliability_baseline.ipynb` — get a feel for the behavior with a few hand-written prompts, **find and load a real dataset yourself**, write a scorer, and measure the baseline.
3. `gaia/02_sycophancy_feature_search_and_edit.ipynb` or `itay/02_...` — feature search → PISCES edit → eval loop: token search, random-feature control, CRISP-style contrastive search (you write the ranking), and a tau/mu sweep.

Everything lives **in the notebook**: there are no data files to load and no output directories — your prompts, datasets, scorers, and feature sets are written in cells, and results stay as cell outputs.

`STUDENT TODO` marks the parts you write. Stubs are `raise NotImplementedError` (functions) or empty lists (prompts/features) — they fail until you fill them in, on purpose.

## What you write vs. what you import

You only ever edit **the notebook**. Nothing in `student_utils/` or the PISCES core needs editing.

- **Written by you, inline in the notebooks:** the scorers (`score_sycophancy` / `score_reliability` and a `score_general_behavior` breakage check), the contrastive ranking (`my_selection`), and all prompts, the dataset loading, search tokens, feature sets, edit configs, and sweep ranges.
- **Imported, never edited:** the PISCES core (`editor.py`, `evals.py`, `feature_finder.py`) and the `student_utils/` plumbing (model loading, generation, `make_eval_dataframe`/`dataset_to_prompts`, `apply_scorer`/`summarize_scores`, feature-search machinery, the edit wrapper, plots). Every helper has a docstring — read them to understand what they do.

Edits auto-revert: `temporary_pisces_edit(...)` restores the weights when the `with` block ends — edits do **not** accumulate across cells. If a cell errors *inside* a `with` block, just re-run the load-model cell to be safe.

The feature catalog (`features/vocab_proj_catalog_*.pkl`) is built once with
`python scripts/build_feature_catalog.py` on the GPU box, then reused by `build_or_load_feature_catalog`.
