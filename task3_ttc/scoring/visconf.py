import numpy as np

EPS = 1e-8


def gather_tokens(token_values, offsets, rows):
    pieces = [token_values[offsets[r]:offsets[r + 1]] for r in rows]
    lengths = np.array([len(p) for p in pieces], dtype=np.int64)
    return np.concatenate(pieces), lengths


def calibrated_confidence(k_pooled):
    q05, q95 = np.percentile(k_pooled, [5, 95], method="linear")
    return np.clip((k_pooled - q05) / max(q95 - q05, EPS), 0.0, 1.0)


def visconf_scores(token_arrays, rows):
    offsets = token_arrays["offsets"]
    k, lengths = gather_tokens(token_arrays["k"], offsets, rows)
    v, _ = gather_tokens(token_arrays["v"], offsets, rows)
    c = calibrated_confidence(k)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    mean_c = np.add.reduceat(c, starts) / lengths
    mean_v = np.add.reduceat(v, starts) / lengths
    mean_cv = np.add.reduceat(c * v, starts) / lengths
    return {"C": mean_c, "Z": mean_v * mean_c, "Ztok": mean_cv, "Vprefix": mean_v}


def token_covariance(token_arrays, rows):
    offsets = token_arrays["offsets"]
    k, lengths = gather_tokens(token_arrays["k"], offsets, rows)
    v, _ = gather_tokens(token_arrays["v"], offsets, rows)
    c = calibrated_confidence(k)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    out = []
    for s, n in zip(starts, lengths):
        ci, vi = c[s:s + n], v[s:s + n]
        out.append(float(np.mean((ci - ci.mean()) * (vi - vi.mean()))))
    return np.array(out)
