# Student notebooks

Run order:
1. `00_intro_and_sanity_edit.ipynb` — load the model, see baseline answers, run the Harry Potter sanity edit, and read the helpers.
2. `gaia/01` or `itay/01` — manual prompt exploration + manual labels.
3. `02` — dataset-based eval (improve the scorer; add target/control rows).
4. `03` — feature search → PISCES edit → eval loop (token search, random-feature control, CRISP-style contrastive search, tau/mu sweep).

Where things are:
- Helpers: `student_utils/`. Seeds: `data/student_evals/`. Feature sets: `features/student_feature_sets/`.
- Outputs are saved under `runs/student_experiments/<run_name>/` (gitignored).

`STUDENT TODO` marks the parts you work on (scorers, prompts/tokens, feature selection, sweeps).

Edits auto-revert: `temporary_pisces_edit(...)` restores the weights when the `with` block ends — edits do **not** accumulate across cells. If a cell errors *inside* a `with` block, just re-run the load-model cell to be safe.

The feature catalog (`features/vocab_proj_catalog_*.pkl`) is built once with
`python scripts/build_feature_catalog.py` on the GPU box, then reused by `build_or_load_feature_catalog`.
