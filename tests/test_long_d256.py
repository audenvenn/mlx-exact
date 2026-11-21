"""Head dims 192/256 with several KV heads (Gemma 3 4B and larger).
- Prompt chunks of >= 1024 tokens with an array mask (a sliding window, or a left-padded batch) raised
  NotImplementedError before 0.12.0. Tiny random-weight Gemma 3 (head dim 256, 4 query / 2 KV heads, window 32: one
  sliding and one global layer); no download.
- Since 0.12.1 these head dims run mlx-exact's composition of MLX's unfused attention with a softmax whose bits do not
  depend on where the keys sit (mod 32) or on the key count. MLX's own softmax does depend on it, which made real
  Gemma 3 4B decode and padded batches differ from prefill (FINDINGS F46). Regression tests at real sizes below."""
import importlib

import numpy as np
import mlx.core as mx

BASE = dict(model_type="gemma3_text", hidden_size=256, num_hidden_layers=2, intermediate_size=512,
            num_attention_heads=4, num_key_value_heads=2, head_dim=256, query_pre_attn_scalar=256, rms_norm_eps=1e-6,
            vocab_size=1024, rope_theta=10000.0, max_position_embeddings=4096, tie_word_embeddings=True,
            sliding_window=32, sliding_window_pattern=2)


def _model():
    mod = importlib.import_module("mlx_lm.models.gemma3_text")
    mx.random.seed(2)
    model = mod.Model(mod.ModelArgs.from_dict(dict(BASE)))
    model.set_dtype(mx.bfloat16)
    return model


def _prefill(model, ids, chunk):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    outs = [model(mx.array(ids[s:s + chunk])[None], cache=cache)[0] for s in range(0, len(ids), chunk)]
    return mx.concatenate(outs, axis=0), cache


def test_long_chunks_equal_short_chunks_and_decode():
    import mlx_exact
    import mlx_exact.attention as A
    model = _model()
    ids = np.random.default_rng(0).integers(0, 1024, 1100).tolist()
    mlx_exact.enable()
    try:
        A.STATS.clear()
        full, _ = _prefill(model, ids, 1100)                      # one chunk: 1100 rows, split into 1008 + 92
        assert A.STATS.get("sdpa_unfused_exact", 0) >= 1, dict(A.STATS)
        chunked, _ = _prefill(model, ids, 300)
        part, cache = _prefill(model, ids[:1080], 1080)           # 1080 rows, then token-by-token decode
        dec = [model(mx.array([[t]]), cache=cache)[0, -1] for t in ids[1080:]]
    finally:
        mlx_exact.disable()
    assert mx.array_equal(full, chunked).item()
    assert mx.array_equal(full[:1080], part).item()
    for i, row in enumerate(dec):
        assert mx.array_equal(full[1080 + i], row).item(), i


def test_long_left_padded_batch_equals_each_sequence_alone():
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, make_prompt_cache
    import mlx_exact
    import mlx_exact.attention as A
    model = _model()
    rng = np.random.default_rng(1)
    prompts = [rng.integers(0, 1024, n).tolist() for n in (1100, 1040)]
    forced = rng.integers(0, 1024, (5, 2)).tolist()
    n = max(len(p) for p in prompts)
    pads = [n - len(p) for p in prompts]
    mlx_exact.enable()
    try:
        alone = []
        for b, p in enumerate(prompts):
            pre, cache = _prefill(model, p, len(p))
            alone.append((pre, [model(mx.array([[forced[s][b]]]), cache=cache)[0, -1] for s in range(5)]))
        caches = [BatchRotatingKVCache(c.max_size, pads) if hasattr(c, "max_size") else BatchKVCache(pads)
                  for c in make_prompt_cache(model)]
        A.STATS.clear()
        out = model(mx.array([[0] * d + p for p, d in zip(prompts, pads)]), cache=caches)
        assert A.STATS.get("sdpa_batch_rotating_aligned", 0) >= 1 and A.STATS.get("sdpa_unfused_exact", 0) >= 2
        steps = [model(mx.array(forced[s])[:, None], cache=caches)[:, -1] for s in range(5)]
    finally:
        mlx_exact.disable()
    for b, d in enumerate(pads):
        assert mx.array_equal(out[b, d:], alone[b][0]).item(), b
        for s in range(5):
            assert mx.array_equal(steps[s][b], alone[b][1][s]).item(), (b, s)


def _bf(rng, *shape, mult=1.0):
    return mx.array(rng.standard_normal(shape).astype(np.float32) * mult).astype(mx.bfloat16)


def test_softmax_lanes_is_unchanged_by_mod32_shifts_and_trailing_masked_entries():
    from mlx_exact.kernels import softmax_lanes
    rng = np.random.default_rng(5)
    x = _bf(rng, 8, 16, 1024, mult=3.0)
    ref = softmax_lanes(x)
    lo = mx.array(float(mx.finfo(mx.bfloat16).min), mx.bfloat16)
    for s in (32, 64, 128, 256, 352, 512):
        xs = mx.concatenate([mx.full((8, 16, s), lo), x], axis=-1)
        assert mx.array_equal(softmax_lanes(xs)[..., s:], ref).item(), s
    xt = mx.concatenate([x, mx.full((8, 16, 77), lo)], axis=-1)
    assert mx.array_equal(softmax_lanes(xt)[..., :1024], ref).item()
    exact = mx.softmax(x.astype(mx.float32), axis=-1)
    assert float(mx.max(mx.abs(ref.astype(mx.float32) - exact)).item()) < 4e-3


def _attention_layouts(attend, trials=24, seed=0):
    """Mismatching outputs between a window's decode-like layouts (mod-32 aligned / shifted by multiples of 32) and its
    prefill-like layout (all keys at their positions, many query rows): Gemma 3 4B shapes, window 1024."""
    rng = np.random.default_rng(seed)
    T, W, bad = 1500, 1024, 0
    for _ in range(trials):
        K, V, q = _bf(rng, 1, 4, T, 256), _bf(rng, 1, 4, T, 256), _bf(rng, 1, 8, 1, 256)
        p = int(rng.integers(W, T))
        lo = p - W + 1
        pos = mx.arange(T)
        mrow = ((pos >= lo) & (pos <= p)).reshape(1, 1, 1, T)
        pre = attend(mx.repeat(q, 64, axis=2), K, V, mx.broadcast_to(mrow, (1, 1, 64, T)))[:, :, -1]
        for s in (lo % 32, lo % 32 + 64, lo % 32 + 352):
            Kd = mx.concatenate([mx.zeros((1, 4, s, 256), mx.bfloat16), K[:, :, lo:p + 1]], axis=2)
            Vd = mx.concatenate([mx.zeros((1, 4, s, 256), mx.bfloat16), V[:, :, lo:p + 1]], axis=2)
            md = (mx.arange(W + s) >= s).reshape(1, 1, 1, W + s)
            bad += int(mx.sum(attend(mx.repeat(q, 16, axis=2), Kd, Vd, md)[:, :, -1] != pre).item())
    return bad


def test_head_dim_256_window_layouts_are_bit_identical():
    from mlx_exact.attention import unfused_attention_exact
    assert _attention_layouts(lambda q, k, v, m: unfused_attention_exact(q, k, v, 1 / 16, m)) == 0
