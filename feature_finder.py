import torch
import json
import random
from typing import DefaultDict
from unittest.mock import DEFAULT
from tqdm import tqdm as _tqdm
from tqdm import tqdm as Pbar
from editor import SAEConfig
import itertools
import numpy as np
import pandas as pd
from editor import Feature, Concept
from editor import unlearn_concept
from evals import eval_alpaca, TransformerLensModel, GeminiEvaluator
from evals import evaluate_mmlu, MCQAEvaluations, evaluate_open_ended, OpenEndedQuestion, GeminiEvaluator, TransformerLensModel
from editor import get_mlp_act_signs

# Randomly selected indices from the MMLU dataset which aren't in the test set
DEFAULT_MMLU_INDICES = [3574, 12927, 2763, 3002, 10615, 7090, 10149, 4797, 5451, 8672, 2358, 7948, 11061, 12424, 8748, 5219, 12543, 2325, 3477, 1534, 4841, 11881, 1565, 13990, 7382, 4217, 8678, 13210, 1676, 2831, 5275, 4611, 9784, 12539, 13249, 5241, 13344, 11073, 8194, 3857, 1666, 7688, 7403, 7218, 2880, 9849, 5419, 11683, 8802, 5783, 6809, 2459, 2618, 4006, 13989, 10766, 12322, 9731, 13353, 6180, 7369, 1374, 4875, 9136, 13270, 4701, 7401, 11218, 6314, 3125, 7299, 6421, 12492, 12200, 11398, 297, 6604, 8779, 3117, 13061, 1559, 6801, 5450, 59, 1737, 528, 11279, 11391, 8336, 12519, 7999, 5680, 6889, 9263, 5049, 6113, 10292, 7434, 5, 12719]

# Relevant for the aformentioned DEFAULT_MMLU_INDICES
DEFAULT_MMLU_PERFORMANCE = {"google/gemma-2-2b-it": 0.61, "meta-llama/Llama-3.1-8B-Instruct": 0.6}

def harmonic_mean(values):
    """
    Calculate the harmonic mean of a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        The harmonic mean of the values
    """
    non_zero_values = [v for v in values if v != 0]
    
    if len(non_zero_values) == 0:
        return 0
    
    return len(non_zero_values) / sum(1/v for v in non_zero_values)

def search_features(model, lls, tokens, minmatch=1, layers=None, verbose=False, k=10, only_pos=False):
    results = []
    result_id = 0

    if layers is None:
        layers = range(model.cfg.n_layers)

    for token in tokens:
        assert len(model.to_tokens(token, prepend_bos=False)[0]) == 1, f"Token '{token}' is not a single token ({model.to_str_tokens(token, prepend_bos=False)})"

    for layer in layers:
        ll = lls[layer]

        for i in range(len(ll.t)):
            if len(set(ll.t[i]).intersection(set(tokens))) >= minmatch:
                results.append(Feature(layer=layer, id=i, neg=True))
                print(f"{result_id}: {(str(ll.t[i][:k])) if verbose else ''}")
                result_id += 1

            if only_pos:
                continue

            if len(set(ll.b[i]).intersection(set(tokens))) >= minmatch:
                results.append(Feature(layer=layer, id=i, neg=False))
                print(f"{result_id}: {(str(ll.b[i][:k])) if verbose else ''}")
                result_id += 1

    return results


def get_feature_saes(model, features: list[Feature]):
    layers = list({f.layer for f in features})

    saes = []
    for layer in _tqdm(layers):
        saes.append(SAEConfig(model.cfg.tokenizer_name, layer, "mlp", "16k" if "gemma" in model.cfg.tokenizer_name else "32k").get())

    return saes

def get_feature_effect(model, features: list[Feature], signs, forget_set: list[str], pos_toks_ids: list[int], neg_toks_ids: list[int], batch_size=3):
    out = DefaultDict(list)
    sim_outs = DefaultDict(list)

    for i in _tqdm(range(0, len(forget_set), batch_size)):
        batch = forget_set[i:i+batch_size]
        clean_logits = model(batch).softmax(dim=-1)

        for feature in features:
            f_concept = Concept(name=f"Feature {feature.id}", k=0.9, value=16, features=[feature])
            with unlearn_concept(model, f_concept, full=True, signed=True, signs=signs, linscale="gemma" in model.cfg.tokenizer_name.lower()):
                logits = model(batch).softmax(dim=-1)

            diff = logits[:,:,pos_toks_ids] - clean_logits[:,:,pos_toks_ids]
            sim_diff = logits[:,:,neg_toks_ids] - clean_logits[:,:,neg_toks_ids]

            out[(feature.layer, feature.id)].extend(diff.flatten().tolist())
            sim_outs[(feature.layer, feature.id)].extend(sim_diff.flatten().tolist())

    return out, sim_outs

def get_feature_activations(model, features: list[Feature], forget_set: list[str], pos_toks: list[str], neg_toks: list[str], batch_size=3):
    saes = get_feature_saes(model, features)

    results = DefaultDict(int)
    for i in _tqdm(range(0, len(forget_set), batch_size)):
        batch = forget_set[i:i+batch_size]

        cache = model.run_with_cache_with_saes(batch, saes=saes, return_type=None)[1]

        for feature in features:
        
            # Get positions where feature fired
            firing_positions = (cache[f"blocks.{feature.layer}.hook_mlp_out.hook_sae_acts_post"][:,:,feature.id] > 0)
            firings = firing_positions.sum()
            results[(feature.layer, feature.id)] += firings

    return results

def filter_features_by_effect_and_activations(model, features: list[Feature], forget_set: str, signs: torch.Tensor, pos_toks: list[str], neg_toks: list[str], filter_by_act=True, verbose=False):
    pos_tok_ids = [model.to_single_token(tok) for tok in pos_toks]
    neg_tok_ids = [model.to_single_token(tok) for tok in neg_toks]

    pos_effects, neg_effects = get_feature_effect(model, features, signs, forget_set.splitlines(), pos_tok_ids, neg_tok_ids, batch_size=3)

    final_features = []
    for feature in features:
        pos_effect = np.mean(pos_effects[(feature.layer, feature.id)])
        neg_effect = np.mean(neg_effects[(feature.layer, feature.id)])

        if pos_effect > 0 or neg_effect < -2:
            if verbose:
                print(f"Filtering feature {feature.id} (layer {feature.layer}) by effect: pos_effect={pos_effect}, neg_effect={neg_effect}")
            continue
        
        final_features.append(feature)

    if filter_by_act:
        activations = get_feature_activations(model, features, forget_set.splitlines(), pos_toks, neg_toks, batch_size=3)

        truly_final_features = []
        for feature in final_features:
            if activations[(feature.layer, feature.id)] == 0:
                if verbose:
                    print(f"Filtering feature {feature.id} (layer {feature.layer}) by activation: activations={activations[(feature.layer, feature.id)]}")
                continue
                
            truly_final_features.append(feature)

        final_features = truly_final_features

    return final_features, (pos_effects, neg_effects, activations)

def filter_features_by_mmlu(model, features: list[Feature], signs: torch.Tensor, target: float | None = None, mmlu_indices: list[int] = DEFAULT_MMLU_INDICES, max_deviation: float = 0.02, verbose=False):
    final_features = []

    if target is None:
        target = DEFAULT_MMLU_PERFORMANCE[model.cfg.tokenizer_name]

    for feature in features:
        concept = Concept(name=f"Feature {feature.id}", k=0.9, value=16, features=[feature])
        with unlearn_concept(model, concept, full=True, signed=True, signs=signs, linscale="gemma" in model.cfg.tokenizer_name.lower()):
            mmlu_res, _ = evaluate_mmlu(model, True, indices=mmlu_indices, evaluation_type=MCQAEvaluations.RANK_BASED, batch_size=3, limit=1000, verbose=False)

            if mmlu_res.score_from_total < target - max_deviation:
                if verbose:
                    print(f"Filtering feature {feature.id} (layer {feature.layer}) by MMLU: mmlu={mmlu_res.score_from_total}, target={target} (difference={target - mmlu_res.score_from_total})")
                continue

            final_features.append(feature)

    return final_features

def find_hps(
        model,
        features: list[Feature],
        pos_toks: list[str],
        neg_toks: list[str],
        concept_data: dict,
        baseline_acc: float,
        baseline_mmlu: float,
        baseline_sim: float,
        baseline_alpaca: tuple[float, float],
        signs_batch_size: int = 3,
        alpaca_batch_size: int = 10,
        max_mmlu_deviation: float= 0.02,
        verbose=True,
        min_mmlu = 0.45,
        max_acc = 0.13,
        min_alpaca = 1.7,
        vs = [4, 7, 10, 13, 18, 24, 30, 36, 42, 50],
        ks = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0.4, 0.3, 0.2],
        do_filter=True,
        output_path="./finding_features/"
    ):
    model_simple_name = "gemma" if "gemma" in model.cfg.tokenizer_name.lower() else "llama"
    linscale = model_simple_name == "gemma"

    path = f"{output_path}/{model_simple_name}_{concept_data['Concept']}_coef_signed{'_linscale' if linscale else ''}_test.csv"

    signs = get_mlp_act_signs(model, pos_toks, concept_data["wikipedia_content"].splitlines()[:1000], batch_size=signs_batch_size)

    if do_filter:
        filtered_features, _ = filter_features_by_effect_and_activations(model, features, concept_data["wikipedia_content"], signs, pos_toks, neg_toks, filter_by_act=True, verbose=verbose)
        filtered_features = filter_features_by_mmlu(model, filtered_features, signs, max_deviation=max_mmlu_deviation, verbose=verbose)
    else:
        filtered_features = features

    with open("mmlu_train_indices.json", "r") as f:
        mmlu_train_indices = json.load(f)

    with open("alpaca_train_indices.json", "r") as f:
        alpaca_train_indices = json.load(f)

    if verbose:
        print(f"Filtered to features: {filtered_features}")

    oes = [OpenEndedQuestion(question=x["q"], answer=x["a"]) for x in concept_data["QA_train"]]
    simdom = [OpenEndedQuestion(question=x["q"], answer=x["a"]) for x in concept_data["SimdomQA_train"]]

    hps = []
    for value in vs:
        hps.append([(value, k) for k in ks])

    i_indices = list(range(len(hps)))
    j_indices = list(range(len(hps[0])))
    indices = list(itertools.product(i_indices, j_indices))
    random.shuffle(indices)

    best_hps = (-1, -1)
    best_acc = -1
    best_mmlu = -1
    best_sim = -1
    best_harmonic = -1

    results = {}
    nonos = set()
    done = set()

    pbar = Pbar(total=len(vs) * len(ks))
    skipped = 0

    concept = Concept(name=concept_data["Concept"], k=-1, value=-1, features=filtered_features)
    evaluator = GeminiEvaluator()

    with open(path + ".concept", "w") as f:
        json.dump(concept.to_dict(), f)

    df = None
    for i, j in indices:
        if (i, j) in results:
            pbar.update(1)
            continue

        concept_results = {}

        value, k = hps[i][j]

        if (i, j) in nonos:
            skipped += 1
            concept_results["skipped"] = True
            results[(k, value)] = concept_results
            continue

        concept.k, concept.value = k, value

        pbar.set_description(f"hps: {value}/{k} | best_harmonic: {best_harmonic:.3f} best_hps: {best_hps} [acc={best_acc:.2f}, mmlu={best_mmlu:.3f}, simdom={best_sim:.2f}]")

        with unlearn_concept(model, concept, full=True, signed=True, signs=signs, linscale=linscale):
            wrapped = TransformerLensModel(model)
            res = evaluate_open_ended(wrapped, evaluator, oes, verbose=False, quit_thresh=max_acc)
            concept_results["accuracy"] = res.score_from_total

            if res.score_from_total > max_acc or res.quit:
                concept_results["skipped"] = True
                results[(k, value)] = concept_results
                skipped += 1

                removed = 0
                indices_removed = []
                for p in range(i+1):
                    for q in range(j+1):
                        if (p, q) not in done and (p, q) not in nonos:
                            removed += 1
                            pbar.update(1)
                            indices_removed.append(hps[p][q])

                        nonos.add((p, q))

                print(f"\nDiscarded {removed} hps (too weak [accuracy={res.score_from_total}]): {indices_removed}")

                continue

            mmlu,_ = evaluate_mmlu(model, True, indices=mmlu_train_indices, evaluation_type=MCQAEvaluations.RANK_BASED, batch_size=3, limit=1000, verbose=False)
            concept_results["mmlu"] = mmlu.score_from_total

            if mmlu.score_from_total < min_mmlu:
                concept_results["skipped"] = True
                results[(k, value)] = concept_results
                skipped += 1

                removed = 0
                indices_removed = []
                for p in range(i, len(hps)):
                    for q in range(j, len(hps[p])):
                        if (p, q) not in done and (p, q) not in nonos:
                            removed += 1
                            pbar.update(1)
                            indices_removed.append(hps[p][q])

                        nonos.add((p, q))

                print(f"\nDiscarded {removed} hps (too strong [mmlu={mmlu.score_from_total}], [accuracy={res.score_from_total}]): {indices_removed}")
                
                continue

            alpaca_res, _, _ = eval_alpaca(wrapped, evaluator, indices=alpaca_train_indices, batch_size=alpaca_batch_size, verbose=False, only_fluency=False)
            concept_results["alpaca"] = np.mean([x[0] for x in alpaca_res]), np.mean([x[1] for x in alpaca_res])
            if concept_results["alpaca"][1] < min_alpaca:
                concept_results["skipped"] = True
                results[(k, value)] = concept_results
                skipped += 1

                removed = 0
                indices_removed = []
                for p in range(i, len(hps)):
                    for q in range(j, len(hps[p])):
                        if (p, q) not in done and (p, q) not in nonos:
                            removed += 1
                            pbar.update(1)
                            indices_removed.append(hps[p][q])

                        nonos.add((p, q))

                print(f"\nDiscarded {removed} hps (too strong [alpaca={concept_results['alpaca'][1]}], [accuracy={res.score_from_total}], [mmlu={mmlu.score_from_total}]): {indices_removed}")
                
                continue

            concept_results["skipped"] = False
            results[(k, value)] = concept_results

            sim_res = evaluate_open_ended(wrapped, evaluator, simdom, verbose=False)
            concept_results["simdom"] = sim_res.score_from_total


            spec_hm = harmonic_mean([concept_results["mmlu"] / baseline_mmlu, concept_results["simdom"] / baseline_sim])
            coherence_hm = harmonic_mean([concept_results["alpaca"][0] / baseline_alpaca[0], concept_results["alpaca"][1] / baseline_alpaca[1]])
            hm = harmonic_mean([1-(concept_results["accuracy"] / baseline_acc), spec_hm, coherence_hm])

            if hm > best_harmonic:
                best_harmonic = hm
                best_hps = (k, value)
                best_acc = concept_results["accuracy"]
                best_mmlu = concept_results["mmlu"]
                best_sim = concept_results["simdom"]

            done.add((i, j))

            df = pd.DataFrame([{"v": k[1], "k": k[0], **v} for k, v in results.items() if not v.get("skipped", False)])
            df.to_csv(path, index=False)
        
            pbar.update(1)
        
    return df, best_hps, best_harmonic, features
