# PISCES Student Research Scaffolding — Design

**Date:** 2026-06-25
**Author:** Yoav Gur-Arieh (mentor) + Claude
**Status:** Approved design, pre-implementation

## 1. Goal & context

Two high-school students (Alpha program, ~8-day summer camp) will do research extending the
PISCES concept-erasure method to **behavior suppression** using `google/gemma-2-2b-it`:

- **Gaia** — suppress **sycophancy**: the model agreeing with false or misleading user claims.
- **Itay** — improve **reliability / truthfulness**: reduce confident-false answers and failures to
  admit uncertainty. **Refusal / over-refusal is tracked separately from truthfulness.**

The PISCES editing algorithm already exists and must **not** be rewritten. The students' real work
is *testing* (building evals) and *finding the right features*. Therefore we pre-build all fragile
plumbing as readable helpers and leave the research-bearing logic as clearly-marked `STUDENT TODO`.

Design principles (from the camp brief, the ChatGPT plan, and mentor decisions):
- Notebook-first, local, no required paid APIs / API keys.
- English for Python identifiers; concise Hebrew is fine in notebook markdown.
- Prefer explicit DataFrames / JSONL over hidden state.
- Helpers small and readable; no heavy dependency manager, no Hydra/CLI-first workflow.
- Each experiment cell framed with the camp's three questions: **why / what / what did we get**.

## 2. The canonical working PISCES API (verified against `erasing_harry_potter.ipynb`)

```python
import torch; torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer
from evals import TransformerLensModel
from editor import unlearn_concept, Feature, Concept, get_mlp_act_signs

model = HookedTransformer.from_pretrained("google/gemma-2-2b-it")
tm = TransformerLensModel(model)
tm.generate(tm.wrap_prompt(question), max_new_tokens=200)

features = [Feature(layer=1, id=8965, neg=True), ...]          # neg=True => suppress
concept  = Concept(name="...", k=0.4, value=36, features=features)  # k = tau, value = mu
signs = get_mlp_act_signs(model, pos_toks, texts)              # optional, improves results

with unlearn_concept(model, concept, linscale=True, signs=signs):   # gemma => linscale=True
    ... generate ...
```

Key facts the design relies on:
- `unlearn_concept` / `steer_features` / `replace_mlp_rows` are **context managers that snapshot and
  restore `model.blocks[layer].mlp.W_out` on exit** — edits are already temporary and do not
  accumulate across cells (as long as generation happens *inside* the `with`).
- `Feature(layer, id, neg, large=False)`: `neg=True` => negative steering value => suppression;
  `large=False` => 16k gemma-scope MLP SAE (our default), `large=True` => 65k.
- `Concept.k` is **tau**, `Concept.value` is **mu**.
- Gemma uses `linscale=True` (down-weights edits in early layers).
- The gemma-scope MLP SAE has `W_enc: [d_model=2304, 16384]`, `W_dec: [16384, 2304]`, applied at
  `hook_mlp_out` (d_model space). `Feature.id` indexes SAE features (0..16383).

## 3. Existing-file fixes (minimal, behavior-preserving)

The mentor approved touching existing files. Two import problems block the student layer today:

- **`evals.py`** — top-level `from gcg_multiple import ...`, `from openai import OpenAI`,
  `from google import generativeai as gai`, `from peft ...`, and a **missing `import os`** make
  `from evals import TransformerLensModel` fail outside the full private env.
  Fix: add `import os`; wrap those heavy imports in `try/except ImportError` (or move them into the
  functions that use them). No algorithm/behavior change — if a lib is genuinely needed and absent,
  the relevant function raises a clear error.
- **`feature_finder.py`** — three call sites use the stale signature
  `unlearn_concept(..., full=True, signed=True, ...)`, which the current `editor.py` no longer
  accepts. Fix to `unlearn_concept(model, concept, signs=signs, linscale=...)` so `search_features`
  and the filter helpers import and run. The current `editor.py` already applies the signed path
  when `signs` is provided, so behavior is preserved.

Both fixes live on `students/base`. We do **not** modify `editor.py` (its algorithm is canonical).

## 4. Branch layout

- **`students/base`** (branched from `main`): all shared scaffolding below, including both students'
  notebook folders fully prepared and runnable on seed data.
- **`students/gaia`**, **`students/itay`** (branched from `students/base`): isolation branches where
  each student commits. Helper fixes are made on `base` and merged into both.

## 5. `student_utils/` package

Import-safety rule: `datasets.py`, `scoring.py`, and feature-set validation import **nothing heavy**
(no torch/editor/evals at module load) so the test suite runs with no GPU/model. Model-dependent
modules import `torch`/`editor`/`evals`/`sae_lens` **lazily inside functions**.

### 5.1 `model_loading.py`
```python
def load_student_model(model_name="google/gemma-2-2b-it", device="cuda", dtype=None) -> tuple[HookedTransformer, TransformerLensModel]
def get_default_generation_config() -> dict   # deterministic: do_sample=False, max_new_tokens=200
```
- Loads via `HookedTransformer.from_pretrained`, returns the raw model and the `TransformerLensModel`
  wrapper. Clear, actionable error if the model/SAEs can't load (gated HF access, missing libs).
- Default model configurable at the top of each notebook. No hard-coded HF tokens.

### 5.2 `generation.py`
```python
def generate_one(tm, prompt, max_new_tokens=200, temperature=0.0) -> str
def generate_many(tm, prompts, max_new_tokens=200, temperature=0.0, batch_size=10) -> list[str]
def make_generation_dataframe(prompts, responses, ids=None, metadata=None) -> pd.DataFrame
def compare_generations_dataframe(prompts, baseline_responses, edited_responses, ids=None) -> pd.DataFrame
```
- Wraps `TransformerLensModel.generate` / `generate_multiple`; applies the gemma chat template via
  `tm.wrap_prompt`. Deterministic by default. Output easy to display in notebooks.

### 5.3 `datasets.py`
Row schema (flexible; only `id` + `prompt` strictly required):
```json
{"id": "str", "prompt": "str", "split": "dev|test|manual", "kind": "target|control|general",
 "category": "str", "ideal_behavior": "str", "notes": "str"}
```
```python
def load_jsonl(path) -> list[dict]
def save_jsonl(rows, path) -> None
def load_eval_dataset(path) -> pd.DataFrame
def validate_eval_dataset(rows_or_df, require_ids=True) -> None   # raises with helpful messages
def dataset_to_prompts(df) -> list[str]
def save_generations_csv(df, path) -> None
def save_results_json(results, path) -> None
```

### 5.4 `pisces_adapter.py`
Feature-set JSON format (the student-facing artifact):
```json
{"name": "gaia_v1", "description": "...",
 "features": [{"layer": 12, "feature_id": 3456, "sign": -1, "why": "top tokens look like agreement"}]}
```
- `sign = -1` => suppress => `Feature(neg=True)`; `sign = +1` => `Feature(neg=False)`.
```python
def load_feature_set(path) -> dict
def save_feature_set(feature_set, path) -> None
def validate_feature_set(feature_set) -> None
def feature_dicts_to_pisces_concept(feature_set, *, tau, mu, name=None) -> Concept
def temporary_pisces_edit(model, feature_set, edit_config)   # context manager
def make_random_feature_set_like(feature_set, *, n_features=None, seed=0) -> dict
```
- `edit_config = {"tau": 0.9, "mu": 8.0, "linscale": True, "use_signs": False, "signs": None, "description": "..."}`.
  Internally builds `Concept(name, k=tau, value=mu, features=...)` and enters
  `unlearn_concept(model, concept, linscale=edit_config["linscale"], signs=...)`.
- `temporary_pisces_edit` is a thin wrapper over the existing context manager; it does **not** add a
  new edit mechanism. (Edits are already auto-reverted; the wrapper adds JSON ergonomics, gemma
  defaults, and optional sign handling.)
- `make_random_feature_set_like` returns a feature set drawn from the **same layers** with random
  feature ids (and matching count), for the random-feature control experiment.

### 5.5 `feature_search.py`
**(a) VocabProj catalog — PRE-BUILT.** Reproduces the `lls` interface without the external
`mechinterp` dependency, by projecting each SAE feature's decoder direction through the unembedding.
```python
def build_feature_catalog(model, layers="all", size="16k", top_k=30, vocab_chunk=8192) -> list[LayerLens]
def save_feature_catalog(catalog, path) -> None
def build_or_load_feature_catalog(model=None, path="features/vocab_proj_catalog_gemma2_2b_16k.pkl", **build_kwargs) -> list[LayerLens]
```
- `LayerLens` is a small object with `.t` (list per feature of top token strings) and `.b` (bottom),
  so the repo's `search_features(model, lls, tokens, ...)` works **unchanged**.
- Math per layer L: `logits = sae.W_dec @ model.W_U` ( `[16384, d_vocab]` ), take top-k / bottom-k
  token ids per feature, map via `model.to_str_tokens`. **Chunked over the vocab dimension**
  (`vocab_chunk`) to avoid materializing `[16384, 256000]` at once.
- Loader can also consume an existing `lls`-style pickle (e.g. `lls_cpu.pkl`) if its `.t`/`.b`
  structure matches. Default workflow: build fresh once via `scripts/build_feature_catalog.py`.

**(b) Token search — PRE-BUILT (reuse) + student choices.**
```python
def search_features_by_tokens(model, catalog, tokens, minmatch=1, layers=None, top_k=20) -> pd.DataFrame
def show_feature_candidates(df, max_rows=50) -> None
```
- Thin wrapper over the repo's `search_features`; returns the standard candidate DataFrame.
- No implementation TODO here. Students' work: choose probe `tokens` and read candidates to judge
  relevance, then assemble a feature set.

**(c) Contrastive search — PRE-BUILT machinery + STUDENT TODO ranking (CRISP recipe).**
```python
def collect_sae_feature_activations(model, prompts, layers, size="16k", batch_size=4) -> pd.DataFrame
def find_contrastive_features(target_prompts, control_prompts, model, *, layers="all", top_k=100, catalog=None) -> pd.DataFrame
```
- `collect_sae_feature_activations` attaches gemma-scope MLP SAEs, runs
  `run_with_cache_with_saes`, reads `blocks.{layer}.hook_mlp_out.hook_sae_acts_post`, and returns a
  DataFrame indexed by `(layer, feature_id)` with **CRISP's two metrics** plus normalized forms:
  - `firing_count` = φ = Σₜ 𝟙[aᵢ⁽ᵗ⁾>0]   and `frac_firing` = φ / n_tokens
  - `sum_act` = A = Σₜ aᵢ⁽ᵗ⁾   and `mean_act` = A / n_tokens
  All fragile parts (SAE loading per layer, hook names, batching, padding, device juggling) live here.
- `find_contrastive_features` calls the collector for both prompt sets, joins them and the VocabProj
  catalog (for readable `top_tokens`/`bottom_tokens`), and formats the standard candidate DataFrame —
  **except** one clearly-marked `STUDENT TODO` block where students implement the selection, using
  CRISP (Ashuach et al., 2026) as the reference recipe:
  - contrast score from **Δφ = φ_target − φ_retain** (Eq 4), select **top-k by Δφ** (Eq 7);
  - filter by **relative activation ratio ρ = A_target/(A_retain+ε) ≥ τ** (Eq 6, 8);
  - set `sign = -1` (suppress) for features that fire more on target.
  Students then try to improve it (normalize for corpus size, sweep τ, use `frac_firing` gap, etc.).
- We adopt **CRISP-style selection → PISCES-style suppression**: selected features are edited with
  `unlearn_concept`, *not* CRISP's LoRA fine-tuning. We use MLP-output SAEs (not residual) so the
  selected features are directly editable by PISCES (which edits `W_out`).

Standard candidate DataFrame columns (all search methods): `layer, feature_id, sign, score,
top_tokens, bottom_tokens, matched_tokens, source_method, notes`.

### 5.6 `scoring.py` — STUDENT TODO (weak heuristics provided)
```python
def score_sycophancy(prompt, response) -> dict       # agrees_with_user, corrects_user, evasive_or_refusal, target_bad_behavior, notes
def score_reliability(prompt, response) -> dict       # truthful_or_cautious, admits_uncertainty, overconfident, over_refusal, target_bad_behavior, notes
def score_general_behavior(prompt, response) -> dict  # looks_coherent, answers_task, generic_refusal, notes
def apply_scorer(df, scorer_fn) -> pd.DataFrame
def summarize_scores(scored_df) -> pd.DataFrame       # target vs control breakdown
```
- Defaults are simple keyword/regex heuristics with explicit comments that they are intentionally
  weak and are the students' main thing to improve. **No LLM-as-judge by default.** For Itay,
  refusal is **not** treated as automatically good — `over_refusal` is tracked separately.

### 5.7 `reporting.py`
```python
def make_run_dir(base_dir="runs/student_experiments", run_name=None) -> Path
def save_run_metadata(run_dir, metadata) -> None
def save_before_after_table(run_dir, df) -> None
def save_score_summary(run_dir, df) -> None
def display_before_after(df, max_rows=20) -> None
def display_feature_table(df, max_rows=50) -> None
def plot_tradeoff(results_df, x_col, y_col) -> None                 # target reduction vs general degradation
def plot_contrastive_scatter(candidates_df, x="frac_firing_control", y="frac_firing_target") -> None  # CRISP Fig 3 style
```
- pandas + matplotlib only (no seaborn). Plots simple and optional. Raw outputs and summaries saved.
- `plot_contrastive_scatter` lets students visually pick upper-left "target" features vs diagonal
  "shared" features.

## 6. Notebooks

Shared on `students/base`:
- `notebooks/00_intro_and_sanity_edit.ipynb` — load model, baseline generations, run the **Harry
  Potter erase from the paper** end-to-end to confirm editing works (work-plan day-1 goal), and walk
  through each `student_utils` helper so students understand the plumbing (camp guideline #12).

Per student (`notebooks/gaia/`, `notebooks/itay/`), runnable top-to-bottom on seed data with
placeholder logic; `STUDENT TODO` marks research work; markdown concise; framed why/what/result:
1. `01_manual_*_generations.ipynb` — hand-written prompts, baseline gens, manual labels
   (agreed/corrected/evaded/unclear for Gaia; truthful/uncertain/overconfident/over-refusal/unclear
   for Itay), optional baseline-vs-edited compare, save manual table.
2. `02_*_dataset_eval.ipynb` — load seed JSONL, baseline gens, define/improve the scorer
   (STUDENT TODO), run scoring, summarize target vs control, save results.
3. `03_*_feature_search_and_edit.ipynb` — baseline eval → token search (choose tokens) → pick &
   justify 3–10 features → PISCES edit → target eval + general-control eval → before/after →
   **random-feature control** → **contrastive section** (define target/control prompts; implement
   CRISP selection TODO) → small **tau/mu sweep** (target reduction vs general degradation) → save run.

`notebooks/README.md` explains: start at `00`, then `01`→`02`→`03`; where outputs are saved; that
TODO sections are the students' work; and that edits auto-revert via `temporary_pisces_edit` (reload
the model only if a cell errors mid-edit). A short pointer is added to the main `README.md`.

## 7. Data, features, runs

- `data/student_evals/` — 3 seed JSONLs (8–12 rows each), mixed target/control, **starter only**:
  - `gaia_sycophancy_seed.jsonl` — false-user-claim / pressure-to-agree targets + reasonable-agree /
    neutral-factual controls.
  - `itay_reliability_seed.jsonl` — misleading + unanswerable/uncertain + simple-factual controls;
    truthfulness kept separate from refusal.
  - `general_behavior_controls_seed.jsonl` — simple capability prompts to detect incoherence/breakage.
  - plus a short `README.md`.
- `features/student_feature_sets/` — `README.md` (format) + `gaia_example_features.json`,
  `itay_example_features.json` as **clearly-marked placeholders with `"features": []`**. We do not
  invent realistic-looking feature ids; real ids come from the search notebooks.
- `runs/student_experiments/.gitkeep`.

## 8. Scripts & tests

- `scripts/build_feature_catalog.py` — mentor runs once on the GPU box to build & cache the VocabProj
  catalog (logs what was built; chunked over vocab). Documents output path used by notebooks. The
  catalog file and `runs/student_experiments/*` outputs are added to `.gitignore` (not committed).
- `scripts/validate_student_notebooks.py` — checks the six student notebooks (+ `00`) exist, are
  valid `.ipynb` JSON, each contains ≥1 `STUDENT TODO`, seed data files exist, and `student_utils`
  imports cleanly.
- `tests/` (model-free, pytest):
  - `test_student_datasets.py` — JSONL load/save round-trip, schema validation, missing-prompt error.
  - `test_student_scoring.py` — each scorer returns its required keys on sample inputs.
  - `test_student_feature_sets.py` — empty placeholder validates (or gives a clear message);
    malformed set raises a useful error; sign↔neg mapping.

## 9. LLM-judge

Off by default. A clearly-marked optional `GeminiEvaluator` hook (already in `evals.py`) can be
switched on later for stronger scoring; no notebook cell requires an API key.

## 10. Assumptions & runtime

- Student execution happens on a GPU box (the mentor's env) with `transformer_lens`, `sae_lens`,
  `datasets`, and gated access to `google/gemma-2-2b-it` + Gemma Scope SAEs — the env in which
  `erasing_harry_potter.ipynb` already runs.
- The VocabProj catalog (~hundreds of MB) is built once on that box; it is **gitignored**, not
  committed.
- The lightweight `student_utils` modules and the entire test suite run on any machine with no GPU.

## 11. Out of scope (non-goals)

- Rewriting or changing the PISCES algorithm (`editor.py`).
- Final datasets, final scorers, final feature lists, or research conclusions — those are the
  students' work.
- Itay's multilingual evaluation and learned "lie-detection probes", and CRISP's LoRA editing — noted
  as possible later stretch directions, not built now.
- CLI-first workflow, config frameworks, new heavy dependencies.

## 12. References

- Gur-Arieh et al. (2025), *Precise In-Parameter Concept Erasure in LLMs* (PISCES), EMNLP. arXiv:2505.22586
- Ashuach et al. (2026), *CRISP: Persistent Concept Unlearning via Sparse Autoencoders*. arXiv:2508.13650
