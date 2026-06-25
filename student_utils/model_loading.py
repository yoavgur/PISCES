"""Load gemma-2-2b-it as a HookedSAETransformer (subclass of HookedTransformer,
so it supports both PISCES editing and run_with_cache_with_saes) and wrap it in
the repo's TransformerLensModel. Imports are lazy so this module imports without
torch installed.
"""

DEFAULT_MODEL = "google/gemma-2-2b-it"


def load_student_model(model_name=DEFAULT_MODEL, device="cuda", dtype=None):
    """Return (model, tm). Raises a clear ImportError if deps/gemma access are missing."""
    import torch
    torch.set_grad_enabled(False)
    try:
        from sae_lens import HookedSAETransformer
    except ImportError as e:
        raise ImportError("sae_lens is required for the student notebooks (`pip install sae_lens`).") from e
    try:
        from evals import TransformerLensModel
    except ImportError as e:
        raise ImportError(
            "Could not import TransformerLensModel from evals.py. Run from the PISCES repo root "
            "with transformer_lens installed (and apply the Task 1 import fixes)."
        ) from e

    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    try:
        model = HookedSAETransformer.from_pretrained(model_name, device=device, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{model_name}'. Check Hugging Face access to the gated gemma model "
            f"(huggingface-cli login) and that Gemma Scope SAEs can be downloaded. Original: {e}"
        ) from e

    tm = TransformerLensModel(model)
    return model, tm


def get_default_generation_config():
    """Deterministic defaults used across notebooks."""
    return {"max_new_tokens": 200, "temperature": 0.0, "do_sample": False}
