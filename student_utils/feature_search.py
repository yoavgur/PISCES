"""Feature search: VocabProj catalog (pre-built), token search (reuse), and
CRISP-style contrastive search.

The pure helper (_format_candidates) imports only pandas. Model/SAE-dependent
functions import torch / editor / feature_finder lazily and run on the GPU box.

Reference: Ashuach et al. (2026), CRISP (arXiv:2508.13650). We use CRISP-style
feature SELECTION but PISCES-style SUPPRESSION (editor.unlearn_concept), on
MLP-output SAEs. The contrastive ranking itself (select_fn) is left for you to
implement from the paper -- see find_contrastive_features.
"""
import pickle
from collections import namedtuple
from pathlib import Path

import pandas as pd

LayerLens = namedtuple("LayerLens", ["t", "b"])

CANDIDATE_COLUMNS = ["layer", "feature_id", "sign", "score",
                     "top_tokens", "bottom_tokens", "matched_tokens",
                     "source_method", "notes"]


# --------------------------- pure: candidate formatting ---------------------------

def _format_candidates(selected, catalog, source_method, tokens=None, top_k_tokens=10):
    """Build the standard candidate DataFrame, attaching readable tokens per feature.

    `selected` has one row per (layer, feature_id) with at least those two
    columns, plus optional 'sign', 'score', and -- for contrastive search --
    'frac_firing_target'/'frac_firing_control'. The row INDEX of `selected` is
    ignored, so a student's custom select_fn may return arbitrarily-indexed rows
    without breaking this function.

    For each feature we look up its top/bottom VocabProj tokens from `catalog`
    (a list indexed by layer; see build_feature_catalog). `matched_tokens` is the
    subset of `tokens` that appears anywhere in the feature's FULL top OR bottom
    list -- so it stays accurate even though the displayed top_tokens/bottom_tokens
    are truncated to `top_k_tokens` for readability.

    Returns a DataFrame with CANDIDATE_COLUMNS, plus frac_firing_target/control
    when those columns are present in `selected`.
    """
    has_firing = ("frac_firing_target" in selected.columns
                  and "frac_firing_control" in selected.columns)
    token_set = set(tokens) if tokens else set()

    rows = []
    frac_target, frac_control = [], []
    for _, r in selected.iterrows():
        layer = int(r["layer"]); fid = int(r["feature_id"])
        top_full, bot_full = [], []
        if catalog is not None and layer < len(catalog) and catalog[layer] is not None:
            top_full = list(catalog[layer].t[fid])
            bot_full = list(catalog[layer].b[fid])
        matched = sorted(token_set & (set(top_full) | set(bot_full))) if token_set else []
        rows.append({
            "layer": layer, "feature_id": fid,
            "sign": int(r.get("sign", -1)),
            "score": r.get("score", None),
            "top_tokens": top_full[:top_k_tokens],
            "bottom_tokens": bot_full[:top_k_tokens],
            "matched_tokens": matched,
            "source_method": source_method, "notes": "",
        })
        if has_firing:
            frac_target.append(r.get("frac_firing_target"))
            frac_control.append(r.get("frac_firing_control"))

    df = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if has_firing:
        df["frac_firing_target"] = frac_target
        df["frac_firing_control"] = frac_control
    return df


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
    """Load the cached VocabProj catalog if present; otherwise build it (model
    required) and cache it. Building scans all layers once and is the slow step,
    so it is cached to disk and reused across notebooks."""
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
    """Find SAE features whose VocabProj tokens include your probe `tokens`.

    Wraps the repo's feature_finder.search_features over the pre-built `catalog`.
    Each entry in `tokens` must be a SINGLE model token (usually with a leading
    space, e.g. ' agree') -- otherwise search_features raises an assertion.

    A feature is returned when at least `minmatch` of your tokens appear in its
    top tokens (suggested sign -1, i.e. suppress) or in its bottom tokens
    (sign +1). Returns a candidate DataFrame (see _format_candidates); the
    'matched_tokens' column shows exactly which of your tokens hit each feature.

    STUDENT work: choose `tokens` and judge which candidates are on-concept.
    """
    import contextlib
    import io
    from feature_finder import search_features
    # search_features prints one line per hit even with verbose=False; swallow it.
    with contextlib.redirect_stdout(io.StringIO()):
        feats = search_features(model, catalog, tokens, minmatch=minmatch,
                                layers=layers, verbose=False, k=top_k)
    selected = pd.DataFrame([
        {"layer": f.layer, "feature_id": f.id, "sign": -1 if f.neg else 1, "score": None}
        for f in feats
    ])
    return _format_candidates(selected, catalog, "token", tokens=tokens, top_k_tokens=10)


def collect_sae_feature_activations(model, prompts, layers, size="16k", batch_size=4):
    """Run the model with MLP SAEs attached and aggregate per-feature activations.

    Returns one row per (layer, feature_id): firing_count (how many tokens the
    feature fired on), frac_firing (that count / total tokens), sum_act (summed
    activation), mean_act. These are the raw ingredients a contrastive ranking
    compares between the target and control prompt sets.
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
    """CRISP-style contrastive feature search feeding a PISCES suppression edit.

    Runs the model with MLP SAEs over `target_prompts` (behaviour present) and
    `control_prompts` (behaviour absent), aggregates per-feature activations with
    collect_sae_feature_activations, and merges them into one row per
    (layer, feature_id). The merged DataFrame has, for each side, the columns
    firing_count_*, frac_firing_*, sum_act_*, mean_act_* (suffixes _target /
    _control).

    YOU must provide `select_fn(merged) -> DataFrame`: the contrastive ranking
    that decides which features to suppress. It should return the chosen rows with
    at least 'layer' and 'feature_id' (and normally 'sign' = -1, suppress). Read
    the CRISP paper for the idea (Ashuach et al. 2026, arXiv:2508.13650): compare
    how much each feature fires on target vs control, then keep the features that
    are both target-specific and strongly activated. Implement it yourself.

    Returns a candidate DataFrame; if your select_fn keeps the frac_firing_*
    columns they are carried through so you can plot target-vs-control firing.
    """
    if select_fn is None:
        raise ValueError(
            "find_contrastive_features requires select_fn. Write your own contrastive "
            "ranking (CRISP, arXiv:2508.13650) and pass it as select_fn=your_function."
        )
    t = collect_sae_feature_activations(model, target_prompts, layers, size)
    c = collect_sae_feature_activations(model, control_prompts, layers, size)
    merged = t.merge(c, on=["layer", "feature_id"], suffixes=("_target", "_control"))
    selected = select_fn(merged)
    selected = selected.head(top_k) if len(selected) > top_k else selected
    return _format_candidates(selected, catalog, "contrastive")
