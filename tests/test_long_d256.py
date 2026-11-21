"""Head dims 192/256 with several KV heads (Gemma 3 4B and larger): prompt chunks of >= 1024 tokens with an array mask
(a sliding window, or a left-padded batch) are split into <= 1008-row blocks on MLX's unfused attention path, like
decode and short chunks. Before 0.12.0 they raised NotImplementedError. Tiny random-weight Gemma 3 (head dim 256,
4 query / 2 KV heads, window 32: one sliding and one global layer); no download."""
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
        assert A.STATS.get("sdpa_split_1008_arraymask", 0) >= 1, dict(A.STATS)
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
        assert A.STATS.get("sdpa_batch_rotating_aligned", 0) >= 1 and A.STATS.get("sdpa_split_1008_arraymask", 0) >= 2
        steps = [model(mx.array(forced[s])[:, None], cache=caches)[:, -1] for s in range(5)]
    finally:
        mlx_exact.disable()
    for b, d in enumerate(pads):
        assert mx.array_equal(out[b, d:], alone[b][0]).item(), b
        for s in range(5):
            assert mx.array_equal(steps[s][b], alone[b][1][s]).item(), (b, s)
