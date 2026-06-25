# Student eval seeds

Starter eval data — **not** final datasets. Each row:
`id, prompt, split, kind, category, ideal_behavior, notes`.

- `kind = target` — the behavior we want to suppress should show up here.
- `kind = control` — the behavior should NOT show up (agreement/answer is fine).
- `kind = general` — general capability checks (used to detect a broken model).

Files:
- `gaia_sycophancy_seed.jsonl` — false-claim / pressure-to-agree targets + reasonable-agreement controls.
- `itay_reliability_seed.jsonl` — false-premise / unanswerable / nonexistent-entity targets + simple-fact controls.
- `general_behavior_controls_seed.jsonl` — generic capability prompts.

STUDENT TODO: grow these (more rows, more categories), keep target/control balanced,
and for Itay keep truthfulness separate from refusal.
