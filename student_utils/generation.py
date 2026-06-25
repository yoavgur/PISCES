"""Generation helpers + result/compare DataFrames.

The DataFrame builders are model-free (pandas only). generate_* wrap the repo's
TransformerLensModel and run on the GPU box.
"""
import pandas as pd


def generate_one(tm, prompt, max_new_tokens=200, temperature=0.0):
    """Single deterministic (by default) generation through the wrapped model."""
    return tm.generate(tm.wrap_prompt(prompt), max_new_tokens=max_new_tokens,
                       temperature=temperature, do_sample=temperature > 0)


def generate_many(tm, prompts, max_new_tokens=200, temperature=0.0, batch_size=10):
    """Batched generation. Returns one response string per prompt."""
    wrapped = [tm.wrap_prompt(p) for p in prompts]
    return tm.generate_multiple(wrapped, max_new_tokens=max_new_tokens,
                                do_sample=temperature > 0, batch_size=batch_size)


def make_generation_dataframe(prompts, responses, ids=None, metadata=None):
    if len(prompts) != len(responses):
        raise ValueError(f"prompts ({len(prompts)}) and responses ({len(responses)}) length mismatch")
    ids = ids if ids is not None else [str(i) for i in range(len(prompts))]
    df = pd.DataFrame({"id": ids, "prompt": prompts, "response": responses})
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)) and len(value) == len(df):
                df[key] = list(value)
            else:
                df[key] = value
    return df


def compare_generations_dataframe(prompts, baseline_responses, edited_responses, ids=None):
    n = len(prompts)
    if not (len(baseline_responses) == len(edited_responses) == n):
        raise ValueError("prompts, baseline_responses, edited_responses must have equal length")
    ids = ids if ids is not None else [str(i) for i in range(n)]
    df = pd.DataFrame({"id": ids, "prompt": prompts,
                       "baseline_response": baseline_responses,
                       "edited_response": edited_responses})
    df["changed"] = df["baseline_response"].str.strip() != df["edited_response"].str.strip()
    return df
