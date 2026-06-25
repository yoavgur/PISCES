"""Structural check for the student scaffolding (model-free).

Verifies: the seven notebooks exist and are valid JSON; the six student notebooks
each contain a STUDENT TODO; the seed data files exist; student_utils imports.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS_ALL = [
    "notebooks/00_intro_and_sanity_edit.ipynb",
    "notebooks/gaia/01_manual_sycophancy_generations.ipynb",
    "notebooks/gaia/02_sycophancy_dataset_eval.ipynb",
    "notebooks/gaia/03_sycophancy_feature_search_and_edit.ipynb",
    "notebooks/itay/01_manual_reliability_generations.ipynb",
    "notebooks/itay/02_reliability_dataset_eval.ipynb",
    "notebooks/itay/03_reliability_feature_search_and_edit.ipynb",
]
NOTEBOOKS_TODO = NOTEBOOKS_ALL[1:]
SEEDS = [
    "data/student_evals/gaia_sycophancy_seed.jsonl",
    "data/student_evals/itay_reliability_seed.jsonl",
    "data/student_evals/general_behavior_controls_seed.jsonl",
]


def _nb_text(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    out = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        out.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(out)


def main():
    errors = []
    for rel in NOTEBOOKS_ALL:
        p = ROOT / rel
        if not p.exists():
            errors.append(f"missing notebook: {rel}")
            continue
        try:
            text = _nb_text(p)
        except Exception as e:
            errors.append(f"invalid notebook JSON: {rel} ({e})")
            continue
        if rel in NOTEBOOKS_TODO and "STUDENT TODO" not in text:
            errors.append(f"no STUDENT TODO in {rel}")
    for rel in SEEDS:
        if not (ROOT / rel).exists():
            errors.append(f"missing seed data: {rel}")
    sys.path.insert(0, str(ROOT))
    try:
        import student_utils.datasets          # noqa: F401
        import student_utils.scoring           # noqa: F401
        import student_utils.pisces_adapter     # noqa: F401
        import student_utils.feature_search     # noqa: F401
    except Exception as e:
        errors.append(f"student_utils import failed: {e}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {len(NOTEBOOKS_ALL)} notebooks, {len(SEEDS)} seed files, student_utils imports cleanly.")


if __name__ == "__main__":
    main()
