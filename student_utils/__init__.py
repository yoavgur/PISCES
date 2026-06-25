"""Student-facing helpers for PISCES behavior-suppression research.

Heavy dependencies (torch / editor / evals / sae_lens / matplotlib) are imported
lazily inside the functions that need them, so datasets/scoring/feature-set
helpers and the test suite run with only pandas/numpy installed.
"""
