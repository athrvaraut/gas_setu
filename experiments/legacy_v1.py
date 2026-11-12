"""
The v1 synthetic data generator, kept only so the old numbers can be reproduced
(experiments/baseline_check.py). It is NOT used by the system any more.

Why it was retired (build_plan.md §13.2, D24): one threshold separates its classes perfectly, its
"normal" home books 12 cylinders a year (the real average is about 6), and every "fraud" row books
faster than the 25-day rule allows. See core/synthetic_population.py for v2.
"""
import numpy as np


def generate_synthetic_data(n_normal=8000, n_fraud=2000, seed=42):
    np.random.seed(seed)
    col = np.column_stack
    normal = col([np.random.normal(27, 4, n_normal), np.random.normal(2.3, 0.8, n_normal),
                  np.random.normal(0.15, 0.1, n_normal), np.random.normal(0, 1, n_normal),
                  np.random.normal(28, 6, n_normal), np.random.normal(0.1, 0.05, n_normal),
                  np.random.normal(12, 3, n_normal), np.random.normal(0.05, 0.03, n_normal)])
    n = n_fraud // 4
    sellers = col([np.random.normal(15.5, 0.8, n), np.random.normal(1.1, 0.2, n), np.random.normal(0.9, 0.1, n),
                   np.random.normal(0, 0.5, n), np.random.normal(15, 1, n), np.random.normal(0.8, 0.1, n),
                   np.random.normal(24, 1, n), np.random.normal(0.02, 0.01, n)])
    ghosts = col([np.random.normal(15.2, 0.3, n), np.random.normal(0.0, 0.1, n), np.random.normal(0.95, 0.05, n),
                  np.random.normal(3, 1, n), np.random.normal(15.1, 0.2, n), np.random.normal(0.95, 0.05, n),
                  np.random.normal(24, 0.5, n), np.random.normal(0.001, 0.001, n)])
    hijacks = col([np.random.normal(16, 1, n), np.random.normal(2.5, 0.5, n), np.random.normal(0.85, 0.1, n),
                   np.random.normal(1, 0.5, n), np.random.normal(16, 1.5, n), np.random.normal(0.7, 0.2, n),
                   np.random.normal(22, 2, n), np.random.normal(0.01, 0.01, n)])
    distributors = col([np.random.normal(15.0, 0.1, n), np.random.normal(4.5, 0.5, n), np.random.normal(0.98, 0.02, n),
                        np.random.normal(0, 0.1, n), np.random.normal(15.0, 0.1, n), np.random.normal(0.99, 0.01, n),
                        np.random.normal(24, 0.2, n), np.random.normal(0.0, 0.001, n)])
    X = np.vstack([normal, sellers, ghosts, hijacks, distributors])
    y = np.array([0] * n_normal + [1] * n_fraud)
    return X, y
