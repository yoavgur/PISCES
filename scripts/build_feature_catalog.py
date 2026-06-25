"""Build and cache the VocabProj feature catalog (run once, on the GPU box).

Usage:
    python scripts/build_feature_catalog.py
    python scripts/build_feature_catalog.py --layers 3 7 12 --top-k 30 \
        --out features/vocab_proj_catalog_gemma2_2b_16k.pkl

The catalog is large and is gitignored; do not commit it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from student_utils.model_loading import load_student_model
from student_utils.feature_search import build_feature_catalog, save_feature_catalog


def main():
    ap = argparse.ArgumentParser(description="Build the VocabProj SAE feature catalog.")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", default="16k")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--feat-chunk", type=int, default=2048)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: all layers")
    ap.add_argument("--out", default="features/vocab_proj_catalog_gemma2_2b_16k.pkl")
    args = ap.parse_args()

    print(f"[build_feature_catalog] loading {args.model} on {args.device} ...")
    model, _ = load_student_model(args.model, device=args.device)
    layers = "all" if args.layers is None else args.layers
    print(f"[build_feature_catalog] building catalog (layers={layers}, size={args.size}, top_k={args.top_k}) ...")
    catalog = build_feature_catalog(model, layers=layers, size=args.size,
                                    top_k=args.top_k, feat_chunk=args.feat_chunk)
    save_feature_catalog(catalog, args.out)
    print(f"[build_feature_catalog] saved -> {args.out}")


if __name__ == "__main__":
    main()
