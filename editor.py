import math
from tqdm import tqdm as _tqdm
import torch
from sae_lens import SAE
from typing import Literal
from typing import DefaultDict
from dataclasses import dataclass
from contextlib import contextmanager
from dataclasses_json import DataClassJsonMixin

gemma_id_to_average = {
    "14-gemmascope-mlp-65k": "average_l0_89",
    "15-gemmascope-mlp-65k": "average_l0_72",
    "16-gemmascope-mlp-65k": "average_l0_66",
    "17-gemmascope-mlp-65k": "average_l0_136",
    "18-gemmascope-mlp-65k": "average_l0_88",
    "11-gemmascope-mlp-65k": "average_l0_88",
    "12-gemmascope-mlp-65k": "average_l0_96",
    "13-gemmascope-mlp-65k": "average_l0_98",
    "8-gemmascope-mlp-65k": "average_l0_110",
    "9-gemmascope-mlp-65k": "average_l0_77",
    "10-gemmascope-mlp-65k": "average_l0_95",
    "3-gemmascope-mlp-65k": "average_l0_68",
    "4-gemmascope-mlp-65k": "average_l0_66",
    "5-gemmascope-mlp-65k": "average_l0_86",
    "6-gemmascope-mlp-65k": "average_l0_101",
    "7-gemmascope-mlp-65k": "average_l0_115",
    "0-gemmascope-mlp-65k": "average_l0_72",
    "1-gemmascope-mlp-65k": "average_l0_127",
    "2-gemmascope-mlp-65k": "average_l0_134",
    "19-gemmascope-mlp-65k": "average_l0_88",
    "20-gemmascope-mlp-65k": "average_l0_88",
    "21-gemmascope-mlp-65k": "average_l0_86",
    "22-gemmascope-mlp-65k": "average_l0_92",
    "23-gemmascope-mlp-65k": "average_l0_102",
    "24-gemmascope-mlp-65k": "average_l0_128",
    "25-gemmascope-mlp-65k": "average_l0_107",
    "12-gemmascope-res-16k": "average_l0_82",
    "13-gemmascope-res-16k": "average_l0_84",
    "14-gemmascope-res-16k": "average_l0_84",
    "22-gemmascope-res-16k": "average_l0_72",
    "18-gemmascope-res-16k": "average_l0_74",
    "19-gemmascope-res-16k": "average_l0_73",
    "6-gemmascope-res-16k": "average_l0_70",
    "11-gemmascope-res-16k": "average_l0_80",
    "7-gemmascope-res-16k": "average_l0_69",
    "8-gemmascope-res-16k": "average_l0_71",
    "9-gemmascope-res-16k": "average_l0_73",
    "21-gemmascope-res-16k": "average_l0_70",
    "5-gemmascope-res-16k": "average_l0_68",
    "15-gemmascope-res-16k": "average_l0_78",
    "16-gemmascope-res-16k": "average_l0_78",
    "4-gemmascope-res-16k": "average_l0_124",
    "1-gemmascope-res-16k": "average_l0_102",
    "2-gemmascope-res-16k": "average_l0_142",
    "3-gemmascope-res-16k": "average_l0_59",
    "20-gemmascope-res-16k": "average_l0_71",
    "10-gemmascope-res-16k": "average_l0_77",
    "17-gemmascope-res-16k": "average_l0_77",
    "0-gemmascope-res-16k": "average_l0_105",
    "23-gemmascope-res-16k": "average_l0_74",
    "24-gemmascope-res-16k": "average_l0_73",
    "25-gemmascope-res-16k": "average_l0_116",
    "9-gemmascope-res-65k": "average_l0_118",
    "22-gemmascope-res-65k": "average_l0_116",
    "23-gemmascope-res-65k": "average_l0_123",
    "1-gemmascope-res-65k": "average_l0_121",
    "8-gemmascope-res-65k": "average_l0_111",
    "7-gemmascope-res-65k": "average_l0_107",
    "2-gemmascope-res-65k": "average_l0_77",
    "12-gemmascope-res-65k": "average_l0_141",
    "0-gemmascope-res-65k": "average_l0_73",
    "10-gemmascope-res-65k": "average_l0_128",
    "11-gemmascope-res-65k": "average_l0_70",
    "14-gemmascope-res-65k": "average_l0_73",
    "15-gemmascope-res-65k": "average_l0_127",
    "16-gemmascope-res-65k": "average_l0_128",
    "17-gemmascope-res-65k": "average_l0_125",
    "13-gemmascope-res-65k": "average_l0_75",
    "18-gemmascope-res-65k": "average_l0_116",
    "19-gemmascope-res-65k": "average_l0_115",
    "20-gemmascope-res-65k": "average_l0_114",
    "24-gemmascope-res-65k": "average_l0_124",
    "25-gemmascope-res-65k": "average_l0_93",
    "21-gemmascope-res-65k": "average_l0_111",
    "3-gemmascope-res-65k": "average_l0_89",
    "4-gemmascope-res-65k": "average_l0_89",
    "5-gemmascope-res-65k": "average_l0_105",
    "6-gemmascope-res-65k": "average_l0_107",
    "0-gemmascope-att-65k": "average_l0_75",
    "1-gemmascope-att-65k": "average_l0_98",
    "2-gemmascope-att-65k": "average_l0_125",
    "3-gemmascope-att-65k": "average_l0_83",
    "4-gemmascope-att-65k": "average_l0_87",
    "5-gemmascope-att-65k": "average_l0_99",
    "6-gemmascope-att-65k": "average_l0_112",
    "7-gemmascope-att-65k": "average_l0_96",
    "8-gemmascope-att-65k": "average_l0_112",
    "18-gemmascope-att-65k": "average_l0_123",
    "19-gemmascope-att-65k": "average_l0_106",
    "20-gemmascope-att-65k": "average_l0_102",
    "21-gemmascope-att-65k": "average_l0_118",
    "9-gemmascope-att-65k": "average_l0_107",
    "10-gemmascope-att-65k": "average_l0_67",
    "11-gemmascope-att-65k": "average_l0_75",
    "12-gemmascope-att-65k": "average_l0_79",
    "13-gemmascope-att-65k": "average_l0_87",
    "22-gemmascope-att-65k": "average_l0_112",
    "23-gemmascope-att-65k": "average_l0_140",
    "24-gemmascope-att-65k": "average_l0_77",
    "25-gemmascope-att-65k": "average_l0_63",
    "14-gemmascope-att-65k": "average_l0_66",
    "15-gemmascope-att-65k": "average_l0_90",
    "16-gemmascope-att-65k": "average_l0_129",
    "17-gemmascope-att-65k": "average_l0_70",
    "8-gemmascope-att-16k": "average_l0_129",
    "17-gemmascope-att-16k": "average_l0_79",
    "19-gemmascope-att-16k": "average_l0_122",
    "18-gemmascope-att-16k": "average_l0_72",
    "20-gemmascope-att-16k": "average_l0_62",
    "16-gemmascope-att-16k": "average_l0_71",
    "21-gemmascope-att-16k": "average_l0_65",
    "22-gemmascope-att-16k": "average_l0_106",
    "23-gemmascope-att-16k": "average_l0_73",
    "24-gemmascope-att-16k": "average_l0_96",
    "9-gemmascope-att-16k": "average_l0_127",
    "10-gemmascope-att-16k": "average_l0_70",
    "11-gemmascope-att-16k": "average_l0_80",
    "12-gemmascope-att-16k": "average_l0_85",
    "25-gemmascope-att-16k": "average_l0_77",
    "13-gemmascope-att-16k": "average_l0_92",
    "14-gemmascope-att-16k": "average_l0_71",
    "15-gemmascope-att-16k": "average_l0_98",
    "0-gemmascope-att-16k": "average_l0_104",
    "1-gemmascope-att-16k": "average_l0_79",
    "2-gemmascope-att-16k": "average_l0_93",
    "3-gemmascope-att-16k": "average_l0_117",
    "4-gemmascope-att-16k": "average_l0_116",
    "5-gemmascope-att-16k": "average_l0_135",
    "6-gemmascope-att-16k": "average_l0_61",
    "7-gemmascope-att-16k": "average_l0_99",
    "3-gemmascope-mlp-16k": "average_l0_95",
    "10-gemmascope-mlp-16k": "average_l0_110",
    "11-gemmascope-mlp-16k": "average_l0_98",
    "12-gemmascope-mlp-16k": "average_l0_108",
    "13-gemmascope-mlp-16k": "average_l0_112",
    "14-gemmascope-mlp-16k": "average_l0_97",
    "15-gemmascope-mlp-16k": "average_l0_80",
    "16-gemmascope-mlp-16k": "average_l0_72",
    "17-gemmascope-mlp-16k": "average_l0_68",
    "0-gemmascope-mlp-16k": "average_l0_119",
    "1-gemmascope-mlp-16k": "average_l0_105",
    "2-gemmascope-mlp-16k": "average_l0_95",
    "4-gemmascope-mlp-16k": "average_l0_85",
    "5-gemmascope-mlp-16k": "average_l0_114",
    "6-gemmascope-mlp-16k": "average_l0_133",
    "7-gemmascope-mlp-16k": "average_l0_60",
    "8-gemmascope-mlp-16k": "average_l0_136",
    "9-gemmascope-mlp-16k": "average_l0_88",
    "18-gemmascope-mlp-16k": "average_l0_106",
    "19-gemmascope-mlp-16k": "average_l0_109",
    "20-gemmascope-mlp-16k": "average_l0_109",
    "21-gemmascope-mlp-16k": "average_l0_113",
    "22-gemmascope-mlp-16k": "average_l0_121",
    "23-gemmascope-mlp-16k": "average_l0_128",
    "24-gemmascope-mlp-16k": "average_l0_73",
    "25-gemmascope-mlp-16k": "average_l0_126",
}

@dataclass
class SAEConfig:
    model_name: Literal["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b", "gemma-2-9b-it", "llama-3-8b", "llama-3-8b-it"]
    layer: int
    type: Literal["mlp", "res", "att"]
    size: Literal["16k", "32k", "65k", "131k"]
    device: Literal["mps", "cpu", "cuda"] = "cuda"

    llama_types = {"mlp": "m", "res": "r", "att": "a"}
    llama_sizes = {"131k": "32", "32k": "8"}

    @property
    def release(self):
        if "llama" in self.model_name:
            return f"llama_scope_lx{self.llama_types[self.type]}_{self.llama_sizes[self.size]}x"

        elif "gemma" in self.model_name:
            assert "2b" in self.model_name or "9b" in self.model_name, "Invalid model name"
            model_size = "2b" if "2b" in self.model_name else "9b"
            return f"gemma-scope-{model_size}-pt-{self.type}"
        
        assert False, "Invalid model name"

    @property
    def sae_id(self):
        if "llama" in self.model_name:
            return f"l{self.layer}{self.llama_types[self.type]}_{self.llama_sizes[self.size]}x"

        elif "gemma" in self.model_name:
            average = gemma_id_to_average[f"{self.layer}-gemmascope-{self.type}-{self.size}"]
            return f"layer_{self.layer}/width_{self.size}/{average}"

        assert False, "Invalid model name"
        
    def get(self):
        return SAE.from_pretrained(release = self.release, sae_id = self.sae_id, device = self.device)[0]


@dataclass(frozen=True)
class Feature(DataClassJsonMixin):
    layer: int
    id: int
    neg: bool
    large: bool = False

    def __hash__(self):
        return hash((self.layer, self.id, self.neg, self.large))

@dataclass
class Concept(DataClassJsonMixin):
    name: str

    # Tau from the paper
    k: float

    # Mu from the paper
    value: float

    features: list[Feature]


def get_mlp_act_signs(model, tokens, text, batch_size=3):
    token_ids = [model.to_single_token(token) for token in tokens]

    n_prompts = 100000
    acts = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp, n_prompts))

    count = 0

    for i in _tqdm(range(0, len(text), batch_size)):
        batch = text[i:i+batch_size]

        toks = model.to_tokens(batch)
        cache = model.run_with_cache(batch, return_type=None)[1]

        for i in range(len(toks)):
            batch_toks = toks[i]

            for tok_pos in range(len(batch_toks)):
                tok = batch_toks[tok_pos]

                if tok.item() in token_ids:
                    for layer in range(model.cfg.n_layers):
                        acts[layer,:,count] = cache[f"blocks.{layer}.mlp.hook_post"][i,tok_pos]
                        count += 1

    acts = acts[:,:,:count]

    signs = torch.sign(acts)
    sum_signs = signs.sum(dim=-1)
    majority_sign = torch.sign(sum_signs)

    return majority_sign

def get_hswaps_full_signed(model, layer, features: list[Feature], k, val, signs: torch.Tensor, all_features=False, linscale=False):
    if linscale:
        original_val = val
        if layer < 10:
            # Linear scaling coefficient: 0.1 for layer 0, increasing to 1.0 for layer 9
            scaling_factor = 0.1 + (0.9 * layer / 9)
            val = original_val * scaling_factor
    if linscale and layer < 9:
        assert val < original_val, "what"

    if "gemma" in model.cfg.tokenizer_name:
        if features[0].large:
            size = "65k"
        else:
            size = "16k"
    else:
        if features[0].large:
            size = "131k"
        else:
            size = "32k"

    sae = SAEConfig(model_name=model.cfg.tokenizer_name, layer=layer, type="mlp", size=size).get().float()

    hindices = DefaultDict(list)

    total_max = 0
    total_max_index_encoded = 0
    for feature in features:
        encoded = model.blocks[layer].mlp.W_out @ sae.W_enc[:, feature.id]

        thresh = encoded.abs().max() * k

        total_max = max(encoded.abs().max(), total_max)

        high_indices = torch.where(encoded.abs() > thresh)[0].tolist()
        for index in high_indices:
            index_encoded = model.blocks[layer].mlp.W_out[index] @ sae.W_enc
            max_index_encoded = index_encoded.abs().max()
            total_max_index_encoded = max(max_index_encoded, total_max_index_encoded)
            hindices[index].append((feature, signs[layer, index], max_index_encoded))

    hswaps = []
    for index, swap_features in hindices.items():
        clean = model.blocks[layer].mlp.W_out[index]
        dirty = sae.decode(sae.encode(model.blocks[layer].mlp.W_out[index]))

        error_term = clean - dirty

        proj = sae.encode(model.blocks[layer].mlp.W_out[index])

        if all_features:
            for feature in features:
                sign_int = -1 if feature.neg else 1
                proj[feature.id] = total_max * val * sign_int
        else:
            for feature, msign, max_index_encoded in swap_features:
                sign_int = -1 if feature.neg else 1
                proj[feature.id] = total_max * val * sign_int * msign
                # proj[feature.id] = total_max_index_encoded * val * sign_int * msign
                # proj[feature.id] = max_index_encoded * val * sign_int * msign
                # print(f"[{index}] proj[{feature.id}] = {proj[feature.id]}")
        
        affected = sae.decode(proj)
        fixed_affected = affected + error_term
        
        if not torch.allclose(fixed_affected, clean):
            hswaps.append((index, fixed_affected))

    return hswaps

def get_hswaps_full(model, layer, features: list[Feature], k, val, all_features=False, linscale=False):
    if linscale:
        original_val = val
        if layer < 10:
            # Linear scaling coefficient: 0.1 for layer 0, increasing to 1.0 for layer 9
            scaling_factor = 0.1 + (0.9 * layer / 9)
            val = original_val * scaling_factor
    if linscale and layer < 9:
        assert val < original_val, "what"

    if "gemma" in model.cfg.tokenizer_name:
        if features[0].large:
            size = "65k"
        else:
            size = "16k"
    else:
        if features[0].large:
            size = "131k"
        else:
            size = "32k"

    sae = SAEConfig(model_name=model.cfg.tokenizer_name, layer=layer, type="mlp", size=size).get().float()

    hindices = DefaultDict(list)

    total_max = 0
    for feature in features:
        encoded = model.blocks[layer].mlp.W_out @ sae.W_enc[:, feature.id]

        thresh = encoded.abs().max() * k

        total_max = max(encoded.abs().max(), total_max)

        high_indices = torch.where(encoded.abs() > thresh)[0].tolist()
        for index in high_indices:
            index_encoded = model.blocks[layer].mlp.W_out[index] @ sae.W_enc
            max_index_encoded = index_encoded.abs().max()

            hindices[index].append((feature, encoded[index] >= 0, max_index_encoded))

    hswaps = []
    for index, swap_features in hindices.items():
        clean = model.blocks[layer].mlp.W_out[index]
        dirty = sae.decode(sae.encode(model.blocks[layer].mlp.W_out[index]))

        error_term = clean - dirty

        proj = sae.encode(model.blocks[layer].mlp.W_out[index])

        if all_features:
            for feature in features:
                sign_int = -1 if feature.neg else 1
                proj[feature.id] = total_max * val * sign_int
        else:
            for feature, is_pos, max_index_encoded in swap_features:
                sign_int = -1 if feature.neg else 1
                proj[feature.id] = total_max * val * sign_int
                # print(f"[{index}] proj[{feature.id}] = {proj[feature.id]}")
        
        affected = sae.decode(proj)
        fixed_affected = affected + error_term
        
        if not torch.allclose(fixed_affected, clean):
            hswaps.append((index, fixed_affected))

    return hswaps

@contextmanager
@torch.no_grad()
def replace_mlp_rows(model, all_switches):
    layer_backups = {}
    for layer, switches in all_switches.items():
        layer_backups[layer] = model.blocks[layer].mlp.W_out.clone()

        changed = layer_backups[layer].clone()

        for index, new in switches:
            changed[index] = new

        model.blocks[layer].mlp.W_out.set_(changed)
        assert not torch.allclose(model.blocks[layer].mlp.W_out, layer_backups[layer]), f"No changes made to the model in layer {layer}"

    try:
        yield

    finally:
        for layer, backup in layer_backups.items():
            model.blocks[layer].mlp.W_out.set_(backup)
            assert torch.allclose(model.blocks[layer].mlp.W_out, backup), f"Changes not reverted in layer {layer}"

@contextmanager
def steer_features(model, flayers: dict[int, list[Feature]], k: float, val: float, linscale=False, signs=None):
    all_switches = {}
    for layer, features in flayers.items():
        if signs is not None:
            all_switches[layer] = get_hswaps_full_signed(model, layer, features, k=k, val=val, signs=signs, linscale=linscale)
        else:
            all_switches[layer] = get_hswaps_full(model, layer, features, k=k, val=val, all_features=False, linscale=linscale)

    # grad_enabled = torch.is_grad_enabled()
    with replace_mlp_rows(model, all_switches):
        yield all_switches

@contextmanager
def unlearn_concept(model, concept: Concept, signs=None, linscale=False):
    flayers = DefaultDict(list)
    for feature in concept.features:
        flayers[feature.layer].append(feature)

    with steer_features(model, flayers, concept.k, concept.value, linscale=linscale, signs=signs):
        yield
