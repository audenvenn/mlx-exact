"""Do a token's logits depend on how the computation was scheduled? Full prefill vs token-by-token decode vs chunked
prefill, and greedy decoding with a KV cache vs recomputing the whole sequence, in stock MLX and with mlx-exact.
Usage: python experiments/schedule_invariance.py [--model mlx-community/Qwen3-0.6B-4bit]
"""
import argparse

import mlx.core as mx
from mlx_exact.testing import bits_equal
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

import mlx_exact

PARA = ("Floating-point addition is not associative, so the order in which a GPU kernel sums the terms of a dot "
        "product changes the last bits of the result. Inference engines pick different kernels for different "
        "shapes: one for a single token during decoding, another for a long prompt during prefill, and yet "
        "another for a small batch. Each kernel reduces in its own order and sometimes rounds intermediate "
        "values to a narrower type. ")


def run(model, ids, tag, chunks=(1, 4, 9, 16, 64, 100), n_gen=64):
    fwd = lambda x, c=None: model(mx.array(x)[None], cache=c)[0]
    ref = fwd(ids)
    mx.eval(ref)
    for c in chunks:
        cache = make_prompt_cache(model)
        same, worst = 0, 0.0
        for s in range(0, len(ids), c):
            o = fwd(ids[s:s + c], cache)
            for i in range(o.shape[0]):
                same += int(bits_equal(o[i], ref[s + i]).item())
                worst = max(worst, float(mx.max(mx.abs(o[i].astype(mx.float32) - ref[s + i].astype(mx.float32))).item()))
        name = "decode (1 token per step)" if c == 1 else f"chunks of {c}"
        print(f"[{tag}] {name:26s} vs full prefill: {same}/{len(ids)} positions bit-identical, max |Δlogit| {worst:.3g}")
    prompt = ids[:40]
    seq, cache = list(prompt), make_prompt_cache(model)
    lg = fwd(seq, cache)[-1]
    for _ in range(n_gen):
        t = int(mx.argmax(lg).item())
        seq.append(t)
        lg = fwd([t], cache)[-1]
    rec = list(prompt)
    for _ in range(n_gen):
        rec.append(int(mx.argmax(fwd(rec)[-1]).item()))
    div = next((i for i, (a, b) in enumerate(zip(seq[40:], rec[40:])) if a != b), None)
    print(f"[{tag}] greedy, KV cache vs recomputing the sequence ({n_gen} tokens): "
          + ("identical" if div is None else f"diverge at generated token {div}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    a = ap.parse_args()
    model, tok = load(a.model)
    ids = tok.encode(PARA * 6)[:360]
    run(model, ids, "stock")
    mlx_exact.enable()
    run(model, ids, "mlx-exact")
    mlx_exact.disable()
