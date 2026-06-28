"""Structural check for the student scaffolding (model-free).

Verifies: the five notebooks exist and are valid JSON; each has the PISCES_ROOT
setup cell; the four student notebooks each contain a STUDENT TODO; no notebook
uses a removed helper (no external data files, no disk output); student_utils
imports cleanly.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS_ALL = [
    "notebooks/00_intro_and_sanity_edit.ipynb",
    "notebooks/gaia/01_sycophancy_baseline.ipynb",
    "notebooks/gaia/02_sycophancy_feature_search_and_edit.ipynb",
    "notebooks/itay/01_reliability_baseline.ipynb",
    "notebooks/itay/02_reliability_feature_search_and_edit.ipynb",
]
NOTEBOOKS_TODO = NOTEBOOKS_ALL[1:]

# Helpers removed in the redesign (data lives in the notebook; results stay as cell
# outputs). A notebook referencing any of these is stale.
FORBIDDEN = ["make_run_dir", "save_score_summary", "save_before_after_table",
             "save_run_metadata", "save_generations_csv", "save_results_json",
             "load_eval_dataset", "load_jsonl", "save_jsonl",
             "load_feature_set", "save_feature_set", "default_contrastive_selection"]


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
        if "PISCES_ROOT" not in text:
            errors.append(f"missing PISCES_ROOT setup cell in {rel}")
        if rel in NOTEBOOKS_TODO and "STUDENT TODO" not in text:
            errors.append(f"no STUDENT TODO in {rel}")
        for bad in FORBIDDEN:
            if bad in text:
                errors.append(f"{rel} uses removed helper '{bad}'")

    sys.path.insert(0, str(ROOT))
    try:
        import student_utils.datasets          # noqa: F401
        import student_utils.scoring           # noqa: F401
        import student_utils.pisces_adapter     # noqa: F401
        import student_utils.feature_search     # noqa: F401
        import student_utils.generation         # noqa: F401
        import student_utils.reporting          # noqa: F401
    except Exception as e:
        errors.append(f"student_utils import failed: {e}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {len(NOTEBOOKS_ALL)} notebooks, student_utils imports cleanly, no removed helpers used.")


if __name__ == "__main__":
    main()
