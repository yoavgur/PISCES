"""Feature search: VocabProj catalog (pre-built), token search (reuse), and
CRISP-style contrastive search.

Pure helpers (default_contrastive_selection, _format_candidates) import only
pandas. Model/SAE-dependent functions import torch / editor / feature_finder
lazily and run on the GPU box.

Reference: Ashuach et al. (2026), CRISP (arXiv:2508.13650). We use CRISP-style
feature SELECTION (Eq 4 Delta-phi, Eq 6 rho, Eq 7-8) but PISCES-style
SUPPRESSION (editor.unlearn_concept), on MLP-output SAEs.
"""
import pickle
from collections import namedtuple
from pathlib import Path

import pandas as pd

LayerLens = namedtuple("LayerLens", ["t", "b"])

CANDIDATE_COLUMNS = ["layer", "feature_id", "sign", "score",
                     "top_tokens", "bottom_tokens", "matched_tokens",
                     "source_method", "notes"]


# --------------------------- pure: ranking + formatting ---------------------------

def default_contrastive_selection(merged, top_k=100, tau=2.0, eps=1e-6):
    """CRISP-style selection. STUDENT TODO: this is the reference recipe; improve it.

    merged: one row per (layer, feature_id) with columns
        firing_count_target, firing_count_control, sum_act_target, sum_act_control.
    Returns the selected rows with added delta_phi, rho, score, sign(=-1, suppress).
    """
    df = merged.copy()
    df["delta_phi"] = df["firing_count_target"] - df["firing_count_control"]          # CRISP Eq 4
    df["rho"] = df["sum_act_target"] / (df["sum_act_control"] + eps)                  # CRISP Eq 6
    df = df.sort_values("delta_phi", ascending=False).head(top_k)                     # CRISP Eq 7
    df = df[df["rho"] >= tau].copy()                                                  # CRISP Eq 8
    df["score"] = df["delta_phi"]
    df["sign"] = -1                                                                   # fires more on target => suppress
    return df.reset_index(drop=True)


def _format_candidates(selected, catalog, source_method, tokens=None, top_k_tokens=20):
    """Build the standard candidate DataFrame, pulling readable tokens from the catalog."""
    rows = []
    for _, r in selected.iterrows():
        layer = int(r["layer"]); fid = int(r["feature_id"])
        top, bot = [], []
        if catalog is not None and layer < len(catalog) and catalog[layer] is not None:
            top = list(catalog[layer].t[fid][:top_k_tokens])
            bot = list(catalog[layer].b[fid][:top_k_tokens])
        matched = sorted(set(top) & set(tokens)) if tokens else []
        rows.append({
            "layer": layer, "feature_id": fid,
            "sign": int(r.get("sign", -1)),
            "score": r.get("score", None),
            "top_tokens": top, "bottom_tokens": bot,
            "matched_tokens": matched,
            "source_method": source_method, "notes": "",
        })
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def show_feature_candidates(df, max_rows=50):
    """Display the candidate table in a notebook (falls back to print)."""
    view = df.head(max_rows)
    try:
        from IPython.display import display
        display(view)
    except Exception:
        print(view.to_string())


# --------------------------- model-dependent: VocabProj catalog ---------------------------

def build_feature_catalog(model, layers="all", size="16k", top_k=30, feat_chunk=2048):
    """VocabProj: project each SAE feature's decoder direction through W_U and
    record top_k / bottom_k token strings per feature. Returns a list indexed by
    layer; entry is a LayerLens (or None for layers not built). MLP 16k SAEs.

    Chunked over features to bound memory. Token strings are decoded with the
    tokenizer so they match user-typed tokens (e.g. ' Harry').
    """
    import torch
    from editor import SAEConfig

    if layers == "all":
        layers = list(range(model.cfg.n_layers))
    d_vocab = model.cfg.d_vocab
    id_to_str = model.tokenizer.batch_decode([[i] for i in range(d_vocab)])
    W_U = model.W_U.float()  # [d_model, d_vocab]

    per_layer = {}
    for layer in layers:
        sae = SAEConfig(model.cfg.tokenizer_name, layer, "mlp", size, device=str(W_U.device)).get().float()
        W_dec = sae.W_dec  # [n_feat, d_model]
        n_feat = W_dec.shape[0]
        tops = [None] * n_feat
        bots = [None] * n_feat
        for s in range(0, n_feat, feat_chunk):
            chunk = W_dec[s:s + feat_chunk].float()       # [c, d_model]
            logits = chunk @ W_U                          # [c, d_vocab]
            top_ids = logits.topk(top_k, dim=-1).indices.cpu().tolist()
            bot_ids = (-logits).topk(top_k, dim=-1).indices.cpu().tolist()
            for j in range(len(top_ids)):
                tops[s + j] = [id_to_str[i] for i in top_ids[j]]
                bots[s + j] = [id_to_str[i] for i in bot_ids[j]]
            del logits
        per_layer[layer] = LayerLens(t=tops, b=bots)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[catalog] layer {layer} done ({n_feat} features)")

    return [per_layer.get(l) for l in range(model.cfg.n_layers)]


def save_feature_catalog(catalog, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(catalog, f)


def build_or_load_feature_catalog(model=None, path="features/vocab_proj_catalog_gemma2_2b_16k.pkl", **build_kwargs):
    """Load the cached catalog if present; otherwise build it (model required) and cache."""
    path = Path(path)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    if model is None:
        raise FileNotFoundError(
            f"No catalog at {path} and no model given to build one. "
            f"Run scripts/build_feature_catalog.py on the GPU box first."
        )
    catalog = build_feature_catalog(model, **build_kwargs)
    save_feature_catalog(catalog, path)
    return catalog


# --------------------------- model-dependent: token + contrastive search ---------------------------

def search_features_by_tokens(model, catalog, tokens, minmatch=1, layers=None, top_k=20):
    """Reuse the repo's search_features over the VocabProj catalog; return candidates DataFrame.

    STUDENT work: choose `tokens` (must each be single tokens) and judge candidates.
    """
    from feature_finder import search_features
    feats = search_features(model, catalog, tokens, minmatch=minmatch, layers=layers, verbose=False, k=top_k)
    selected = pd.DataFrame([
        {"layer": f.layer, "feature_id": f.id, "sign": -1 if f.neg else 1, "score": None}
        for f in feats
    ])
    return _format_candidates(selected, catalog, "token", tokens=tokens, top_k_tokens=top_k)


def collect_sae_feature_activations(model, prompts, layers, size="16k", batch_size=4):
    """Run the model with MLP SAEs attached and aggregate per-feature activations.

    Returns one row per (layer, feature_id): firing_count (phi), frac_firing,
    sum_act (A), mean_act. (CRISP Eq 3/5.)
    """
    import torch
    from editor import SAEConfig

    if layers == "all":
        layers = list(range(model.cfg.n_layers))
    saes = {layer: SAEConfig(model.cfg.tokenizer_name, layer, "mlp", size,
                             device=str(model.W_U.device)).get() for layer in layers}
    sae_list = list(saes.values())

    firing = {layer: None for layer in layers}
    sumact = {layer: None for layer in layers}
    n_tokens = 0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tokens = model.to_tokens(batch)
        _, cache = model.run_with_cache_with_saes(tokens, saes=sae_list, return_type=None)
        mask = (tokens != model.tokenizer.pad_token_id).unsqueeze(-1).float()  # [b, seq, 1]
        n_tokens += int(mask.sum().item())
        for layer in layers:
            acts = cache[f"blocks.{layer}.hook_mlp_out.hook_sae_acts_post"].float()  # [b, seq, n_feat]
            acts = acts * mask
            f = (acts > 0).float().sum(dim=(0, 1)).cpu()
            a = acts.sum(dim=(0, 1)).cpu()
            firing[layer] = f if firing[layer] is None else firing[layer] + f
            sumact[layer] = a if sumact[layer] is None else sumact[layer] + a
        del cache

    rows = []
    denom = max(n_tokens, 1)
    for layer in layers:
        fc = firing[layer]; sa = sumact[layer]
        for fid in range(fc.shape[0]):
            rows.append({"layer": layer, "feature_id": fid,
                         "firing_count": float(fc[fid]), "frac_firing": float(fc[fid]) / denom,
                         "sum_act": float(sa[fid]), "mean_act": float(sa[fid]) / denom})
    return pd.DataFrame(rows)


def find_contrastive_features(target_prompts, control_prompts, model, *, layers="all",
                              size="16k", top_k=100, catalog=None, select_fn=None):
    """CRISP-style contrastive feature search feeding PISCES.

    Pre-built: collects activations for both prompt sets, merges them. The
    ranking/selection is `select_fn` (defaults to default_contrastive_selection).
    STUDENT TODO in the notebook: pass your own select_fn.
    """
    if select_fn is None:
        select_fn = default_contrastive_selection
    t = collect_sae_feature_activations(model, target_prompts, layers, size)
    c = collect_sae_feature_activations(model, control_prompts, layers, size)
    merged = t.merge(c, on=["layer", "feature_id"], suffixes=("_target", "_control"))
    selected = select_fn(merged)
    selected = selected.head(top_k) if len(selected) > top_k else selected
    return _format_candidates(selected, catalog, "contrastive")
