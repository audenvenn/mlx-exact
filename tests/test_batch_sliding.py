"""Batched sliding-window attention (mlx-lm BatchRotatingKVCache, e.g. BatchGenerator or mlx_lm.server with Gemma 3 /
Cohere 2 / EXAONE 4): with mlx-exact, every sequence of a batch gets the greedy tokens it gets alone. Each sequence's
window is put back in temporal order and shifted to its absolute positions mod 32 (offsets route), or only put in
temporal order (fixed-order route, head dim 256 with one KV head). Tiny random-weight models; no download."""
import importlib

import numpy as np
import mlx.core as mx
import pytest

BASE = dict(hidden_size=256, num_hidden_layers=2, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
            head_dim=64, rms_norm_eps=1e-6, vocab_size=1024, rope_theta=10000.0, max_position_embeddings=2048,
            tie_word_embeddings=True)
CFG = {
    "gemma3_text": dict(query_pre_attn_scalar=64, sliding_window=32, sliding_window_pattern=2),
    "cohere2": dict(layer_norm_eps=1e-5, sliding_window=32, sliding_window_pattern=2, logit_scale=1.0),
    "exaone4": dict(rope_scaling=None, sliding_window=32, sliding_window_pattern="LG"),
    "gemma3_text_d256_1kv": dict(model_type="gemma3_text", head_dim=256, num_key_value_heads=1,
                                 query_pre_attn_scalar=256, sliding_window=32, sliding_window_pattern=2),
}


@pytest.mark.parametrize("name", list(CFG))
def test_batched_sliding_window_generation_equals_each_prompt_alone(name):
    from mlx_lm.generate import BatchGenerator, generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx_exact
    extra = dict(CFG[name])
    arch = extra.pop("model_type", name)
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(1)
    model = mod.Model(mod.ModelArgs.from_dict(dict(BASE, model_type=arch, **extra)))
    model.set_dtype(mx.bfloat16)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, 1024, n).tolist() for n in (20, 45, 70)]    # shorter and longer than the window
    N = 40
    mlx_exact.enable()
    try:
        alone = []
        for p in prompts:
            out = []
            for (t, _), _ in zip(generate_step(mx.array(p), model, max_tokens=N), range(N)):
                out.append(int(t))
            alone.append(out)
        gen = BatchGenerator(model, stop_tokens=[], sampler=make_sampler(temp=0.0), completion_batch_size=3,
                             prefill_batch_size=3, max_tokens=N)
        uids = gen.insert([list(p) for p in prompts], [N] * 3)
        got = {u: [] for u in uids}
        while responses := gen.next_generated():
            for r in responses:
                got[r.uid].append(int(r.token))
        gen.close()
        st = mlx_exact.stats()["attention"]
        assert st.get("sdpa_batch_rotating_aligned", 0) + st.get("sdpa_batch_rotating_temporal", 0) > 0, st
    finally:
        mlx_exact.disable()
    for b, u in enumerate(uids):
        assert got[u][:N] == alone[b], (name, b)


def _model(name):
    extra = dict(CFG[name])
    arch = extra.pop("model_type", name)
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(1)
    model = mod.Model(mod.ModelArgs.from_dict(dict(BASE, model_type=arch, **extra)))
    model.set_dtype(mx.bfloat16)
    return model


@pytest.mark.parametrize("name", list(CFG))
@pytest.mark.parametrize("chunk", [70, 16])
def test_batched_sliding_window_logits_equal_each_sequence_alone(name, chunk):
    """Strict: a left-padded batch through the model's own cache types (BatchRotatingKVCache for sliding layers),
    prefilled in chunks, then 40 forced decode steps; every step's logits equal each sequence alone, bit for bit."""
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, make_prompt_cache
    import mlx_exact
    model = _model(name)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, 1024, n).tolist() for n in (20, 45, 70)]
    N = 40
    mlx_exact.enable()
    try:
        solo, toks = [], []
        for p in prompts:
            cache = make_prompt_cache(model)
            for s0 in range(0, len(p), chunk):
                out = model(mx.array(p[s0:s0 + chunk])[None], cache=cache)
            rows, ts = [out[0, -1]], []
            for _ in range(N):
                t = int(mx.argmax(rows[-1]).item())
                ts.append(t)
                rows.append(model(mx.array([[t]]), cache=cache)[0, -1])
            solo.append(rows)
            toks.append(ts)
        n = max(len(p) for p in prompts)
        pads = [n - len(p) for p in prompts]
        caches = [BatchRotatingKVCache(c.max_size, pads) if hasattr(c, "max_size") else BatchKVCache(pads)
                  for c in make_prompt_cache(model)]
        x = mx.array([[0] * d + p for p, d in zip(prompts, pads)])
        for s0 in range(0, n, chunk):
            out = model(x[:, s0:s0 + chunk], cache=caches)
        batch = [out[:, -1]]
        for t in range(N):
            batch.append(model(mx.array([toks[b][t] for b in range(len(prompts))])[:, None], cache=caches)[:, -1])
    finally:
        mlx_exact.disable()
    for step in range(N + 1):
        for b in range(len(prompts)):
            assert mx.array_equal(batch[step][b], solo[b][step]).item(), (name, chunk, step, b)
