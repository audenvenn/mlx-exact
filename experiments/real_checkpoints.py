"""Real checkpoints, stock MLX vs mlx-exact: is a token's logits row the same however the prompt was scheduled?
For each model, with a 1,500-token prompt (GSM8K questions, concatenated), logits rows compared bit for bit:
  - chunks: one 1,500-token chunk vs chunks of 512 (109 checked rows);
  - decode: a 1,400-token chunk, then the last 100 tokens one at a time, vs the one-chunk rows;
  - batch: a left-padded batch (1,500 and 1,200 tokens) vs each sequence alone: the last row and 10 forced decode
    steps (plain KV / rotating caches only; --no-batch skips it);
  - greedy: 64 generated tokens with prefill steps of 2,048 vs 512.
Usage: python experiments/real_checkpoints.py [--no-batch] mlx-community/Llama-3.2-1B-Instruct-4bit [...]
"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx_exact  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache, make_prompt_cache  # noqa: E402

P = sorted({0, 511, 512, 1000, 1023, 1024, 1025, 1100, 1300} | set(range(1400, 1500)))


def schedules(model, ids):
    full = model(mx.array(ids)[None], cache=make_prompt_cache(model))
    ref = {p: full[0, p] for p in P}
    mx.eval(list(ref.values()))
    del full
    cache, chunk = make_prompt_cache(model), {}
    for s in range(0, len(ids), 512):
        o = model(mx.array(ids[s:s + 512])[None], cache=cache)
        chunk.update({p: o[0, p - s] for p in P if s <= p < s + o.shape[1]})
        mx.eval(list(chunk.values()))
    cache = make_prompt_cache(model)
    model(mx.array(ids[:1400])[None], cache=cache)
    dec = {}
    for p in range(1400, 1500):
        dec[p] = model(mx.array([[ids[p]]]), cache=cache)[0, -1]
        mx.eval(dec[p])
    return (sum(int(mx.array_equal(ref[p], chunk[p]).item()) for p in P),
            sum(int(mx.array_equal(ref[p], dec[p]).item()) for p in range(1400, 1500)))


def batched(model, a, b):
    pads = [max(len(a), len(b)) - len(a), max(len(a), len(b)) - len(b)]
    forced = [[int(t) for t in np.random.default_rng(s).integers(0, 1000, 2)] for s in range(10)]
    alone = []
    for i, p in enumerate((a, b)):
        cache = make_prompt_cache(model)
        rows = [model(mx.array(p)[None], cache=cache)[0, -1]]
        rows += [model(mx.array([[forced[s][i]]]), cache=cache)[0, -1] for s in range(10)]
        mx.eval(rows)
        alone.append(rows)
    caches = [BatchRotatingKVCache(c.max_size, pads) if hasattr(c, "max_size") else BatchKVCache(pads)
              for c in make_prompt_cache(model)]
    o = model(mx.array([[0] * d + p for p, d in zip((a, b), pads)]), cache=caches)
    got = [[o[0, -1]], [o[1, -1]]]
    for s in range(10):
        r = model(mx.array(forced[s])[:, None], cache=caches)[:, -1]
        got[0].append(r[0])
        got[1].append(r[1])
        mx.eval(r)
    return [sum(int(mx.array_equal(x, y).item()) for x, y in zip(got[i], alone[i])) for i in range(2)]


def greedy(model, ids, step):
    return [int(t) for (t, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=64, prefill_step_size=step),
                                        range(64))]


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    no_batch = "--no-batch" in sys.argv
    repos = [a for a in sys.argv[1:] if not a.startswith("--")]
    qs = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                                       repo_type="dataset")).column("question").to_pylist()
    for repo in repos:
        model, tok = load(repo)
        ids, ids_b = tok.encode("\n\n".join(qs[:80]))[:1500], tok.encode("\n\n".join(qs[80:160]))[:1200]
        plain = all(type(c) in (KVCache, RotatingKVCache) for c in make_prompt_cache(model)) and not no_batch
        for mode in ("stock", "exact"):
            if mode == "exact":
                mlx_exact.enable()
            try:
                t0 = time.time()
                c, d = schedules(model, ids)
                bt = batched(model, ids, ids_b) if plain else None
                g = sum(x == y for x, y in zip(greedy(model, ids, 2048), greedy(model, ids, 512)))
                b = "n/a" if bt is None else f"{bt[0]}/11 and {bt[1]}/11"
                print(f"{repo.split('/')[-1]:34s} {mode}: chunks {c}/{len(P)}, decode {d}/100, batch {b}, greedy {g}/64 "
                      f"({time.time() - t0:.0f} s)", flush=True)
            finally:
                if mode == "exact":
                    mlx_exact.disable()
        del model
        mx.clear_cache()


if __name__ == "__main__":
    main()
