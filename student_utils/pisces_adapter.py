"""Adapter between the student-facing feature-set format and PISCES.

Pure helpers (validation, sign mapping, random control) import only stdlib.
editor.* is imported lazily inside the functions that build/apply edits, so this
module imports without torch.

Feature-set format (a plain dict you write inline in the notebook):
    {"name": str, "description": str,
     "features": [{"layer": int, "feature_id": int, "sign": -1|1, "why": str}]}
sign == -1 => suppress the feature => editor.Feature(neg=True).
"""
import random
from contextlib import contextmanager

REQUIRED_FEATURE_KEYS = ("layer", "feature_id", "sign")


def validate_feature_set(feature_set) -> None:
    """Check a feature-set dict is well-formed; raise ValueError with a clear message.

    Validates that:
      - `feature_set` is a dict with a 'features' list (use [] for an empty
        placeholder while you are still searching);
      - each feature is a dict containing 'layer' (int), 'feature_id' (int), and
        'sign' (exactly -1 for suppress or 1).

    Does NOT check that layer/feature_id are in range for the model -- that is up
    to you. Returns None; call it for its side effect (raising) before building an
    edit. 'why' is optional but recommended (record why you picked each feature).
    """
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
    """Map one feature dict to (layer, feature_id, neg), where neg = (sign == -1)."""
    return (fd["layer"], fd["feature_id"], fd["sign"] == -1)


def feature_dicts_to_pisces_concept(feature_set, *, tau, mu, name=None):
    """Build an editor.Concept from a feature set (imports editor lazily).

    tau -> Concept.k (firing threshold), mu -> Concept.value (edit strength).
    Raises if the feature set is empty (nothing to edit).
    """
    validate_feature_set(feature_set)
    from editor import Feature, Concept  # lazy: needs torch
    features = [Feature(layer=l, id=fid, neg=neg)
                for (l, fid, neg) in (feature_dict_to_args(fd) for fd in feature_set["features"])]
    if not features:
        raise ValueError("cannot build a Concept from an empty feature set (add features first)")
    return Concept(name=name or feature_set.get("name", "concept"), k=tau, value=mu, features=features)


def make_random_feature_set_like(feature_set, *, n_features=None, seed=0, n_sae_features=16384) -> dict:
    """Build a random-feature CONTROL with the same layers/signs as `feature_set`.

    Replaces each feature's id with a random id in [0, n_sae_features). Running the
    same edit config on this control tells you whether your effect is specific to
    the features you chose: if random features change behaviour just as much, your
    selection is not doing the work. Deterministic given `seed`.
    """
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
    """Apply a PISCES suppression edit for the duration of the `with` block, then revert.

    Thin wrapper over editor.unlearn_concept, which snapshots the MLP output
    weights on enter and restores them on exit -- so generations inside the block
    are edited and everything after the block is back to baseline.

    edit_config keys:
      tau       -> Concept.k  (firing threshold)
      mu        -> Concept.value (edit strength)
      linscale  -> True for gemma (default True)
      use_signs -> whether to pass activation signs (default False)
      signs     -> precomputed signs if use_signs is True
    """
    from editor import unlearn_concept  # lazy: needs torch
    concept = feature_dicts_to_pisces_concept(
        feature_set, tau=edit_config["tau"], mu=edit_config["mu"], name=feature_set.get("name")
    )
    signs = edit_config.get("signs") if edit_config.get("use_signs", False) else None
    with unlearn_concept(model, concept, linscale=edit_config.get("linscale", True), signs=signs):
        yield
