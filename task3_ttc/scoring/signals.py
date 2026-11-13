import numpy as np


def per_rollout_mean(flat, offsets):
    lengths = np.diff(offsets)
    assert (lengths > 0).all(), "every rollout must have at least one retained token"
    sums = np.add.reduceat(flat.astype(np.float64), offsets[:-1])
    return sums / lengths


def visual_share(tokens):
    img = tokens["img_attn_total"][0].astype(np.float64)
    prompt = tokens["prompt_text_attn_total"][0].astype(np.float64)
    gen = tokens["generated_text_attn_total"][0].astype(np.float64)
    denom = img + prompt + gen
    assert (denom > 0).all(), "attention denominator is zero for some token"
    return img / denom


def rollout_signals(meta, tokens, vlm):
    offsets = tokens["logp"][1]
    signals = {
        "T": np.diff(offsets).astype(np.int64),
        "L": per_rollout_mean(tokens["logp"][0], offsets),
        "K": per_rollout_mean(tokens["kl_u_p"][0], offsets),
    }
    assert np.isfinite(signals["L"]).all() and np.isfinite(signals["K"]).all()
    token_arrays = {"k": tokens["kl_u_p"][0].astype(np.float64), "offsets": offsets}
    if vlm:
        v = visual_share(tokens)
        signals["V"] = per_rollout_mean(v, offsets)
        token_arrays["v"] = v
    return signals, token_arrays
