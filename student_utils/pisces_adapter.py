"""Adapter between the student-facing feature-set JSON format and PISCES.

Pure helpers (validation, sign mapping, random control, IO) import only stdlib.
editor.* is imported lazily inside the functions that build/apply edits, so this
module imports without torch.

Feature-set format:
    {"name": str, "description": str,
     "features": [{"layer": int, "feature_id": int, "sign": -1|1, "why": str}]}
sign == -1 => suppress => editor.Feature(neg=True).
"""
import json
import random
from contextlib import contextmanager
from pathlib import Path

REQUIRED_FEATURE_KEYS = ("layer", "feature_id", "sign")


def load_feature_set(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        fs = json.load(f)
    validate_feature_set(fs)
    return fs


def save_feature_set(feature_set, path) -> None:
    validate_feature_set(feature_set)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(feature_set, f, ensure_ascii=False, indent=2)


def validate_feature_set(feature_set) -> None:
    if not isinstance(feature_set, dict):
        raise ValueError(f"feature set must be a dict, got {type(feature_set).__name__}")
    if "features" not in feature_set or not isinstance(feature_set["features"], list):
        raise ValueError("feature set must have a 'features' list (use [] for an empty placeholder)")
    for i, fd in enumerate(feature_set["features"]):
        if not isinstance(fd, dict):
            raise ValueError(f"feature {i}: must be a dict, got {type(fd).__name__}")
        for key in REQUIRED_FEATURE_KEYS:
            if key not in fd:
                raise ValueError(f"feature {i}: missing required key '{key}'")
        if not isinstance(fd["layer"], int):
            raise ValueError(f"feature {i}: 'layer' must be an int")
        if not isinstance(fd["feature_id"], int):
            raise ValueError(f"feature {i}: 'feature_id' must be an int")
        if fd["sign"] not in (-1, 1):
            raise ValueError(f"feature {i}: 'sign' must be -1 (suppress) or 1, got {fd['sign']!r}")


def feature_dict_to_args(fd) -> tuple:
    """(layer, feature_id, neg) where neg = (sign == -1)."""
    return (fd["layer"], fd["feature_id"], fd["sign"] == -1)


def feature_dicts_to_pisces_concept(feature_set, *, tau, mu, name=None):
    """Build an editor.Concept from a feature set. Imports editor lazily."""
    validate_feature_set(feature_set)
    from editor import Feature, Concept  # lazy: needs torch
    features = [Feature(layer=l, id=fid, neg=neg)
                for (l, fid, neg) in (feature_dict_to_args(fd) for fd in feature_set["features"])]
    if not features:
        raise ValueError("cannot build a Concept from an empty feature set (placeholder)")
    return Concept(name=name or feature_set.get("name", "concept"), k=tau, value=mu, features=features)


def make_random_feature_set_like(feature_set, *, n_features=None, seed=0, n_sae_features=16384) -> dict:
    """A control feature set: same layers/signs, random feature ids in [0, n_sae_features)."""
    validate_feature_set(feature_set)
    rng = random.Random(seed)
    src = feature_set["features"]
    if n_features is not None:
        src = src[:n_features]
    rand_features = [
        {"layer": fd["layer"], "feature_id": rng.randrange(n_sae_features),
         "sign": fd["sign"], "why": "random control"}
        for fd in src
    ]
    return {
        "name": f"{feature_set.get('name', 'features')}_random",
        "description": f"Random control for {feature_set.get('name', 'features')} (seed={seed}).",
        "features": rand_features,
    }


@contextmanager
def temporary_pisces_edit(model, feature_set, edit_config):
    """Apply a PISCES edit for the duration of the `with` block, then auto-revert.

    Thin wrapper over editor.unlearn_concept (which already snapshots/restores
    W_out). edit_config keys: tau, mu, linscale(=True), use_signs(=False),
    signs(=None), description.
    """
    from editor import unlearn_concept  # lazy: needs torch
    concept = feature_dicts_to_pisces_concept(
        feature_set, tau=edit_config["tau"], mu=edit_config["mu"], name=feature_set.get("name")
    )
    signs = edit_config.get("signs") if edit_config.get("use_signs", False) else None
    with unlearn_concept(model, concept, linscale=edit_config.get("linscale", True), signs=signs):
        yield
