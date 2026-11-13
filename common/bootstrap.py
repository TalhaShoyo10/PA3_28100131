import numpy as np

from common.seed import SEED


def bootstrap_indices(n_units, n_resamples, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_resamples, n_units))


def percentile_ci(samples, alpha=0.05):
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(values_a, values_b, n_resamples, seed=SEED, alpha=0.05):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1
    idx = bootstrap_indices(len(a), n_resamples, seed)
    diffs = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = percentile_ci(diffs, alpha)
    return {
        "mean_diff": float((a - b).mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n_units": int(len(a)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }
