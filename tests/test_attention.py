"""Short query blocks and left-padded batches give the same rows as full prefill."""
import numpy as np
import mlx.core as mx
from mlx_exact.testing import bits_equal
import pytest

from mlx_exact.attention import exact_sdpa


def _mk(rng, *s):
    return mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)


@pytest.mark.parametrize("D,HQ,HKV", [(128, 16, 8), (256, 8, 2)])
@pytest.mark.parametrize("T", [3, 12, 40, 300])
def test_decode_rows_equal_prefill_rows(D, HQ, HKV, T):
    rng = np.random.default_rng(T + D)
    q, k, v = _mk(rng, 1, HQ, T, D), _mk(rng, 1, HKV, T, D), _mk(rng, 1, HKV, T, D)
    full = exact_sdpa(q, k, v, None, D ** -0.5, "causal" if T > 1 else None)
    for t in range(T):
        one = exact_sdpa(q[:, :, t:t + 1], k[:, :, :t + 1], v[:, :, :t + 1], None, D ** -0.5, None)
        assert bits_equal(one[:, :, 0], full[:, :, t]).item(), t
    for c in (4, 9, 16):
        for s0 in range(0, T, c):
            e0 = min(T, s0 + c)
            blk = exact_sdpa(q[:, :, s0:e0], k[:, :, :e0], v[:, :, :e0], None, D ** -0.5,
                             "causal" if e0 - s0 > 1 else None)
            assert bits_equal(blk, full[:, :, s0:e0]).item(), (c, s0)


@pytest.mark.parametrize("n", [17, 30, 59])
def test_left_padding_is_realigned(n):
    rng = np.random.default_rng(n)
    D = 128
    q, k, v = _mk(rng, 1, 16, 1, D), _mk(rng, 1, 8, n, D), _mk(rng, 1, 8, n, D)
    solo = exact_sdpa(q, k, v, None, D ** -0.5, None)
    for p in (1, 7, 13, 29, 42):
        kp = mx.concatenate([_mk(rng, 1, 8, p, D), k], axis=2)
        vp = mx.concatenate([_mk(rng, 1, 8, p, D), v], axis=2)
        m = mx.array(np.arange(n + p) >= p)[None, None, None, :]
        assert bits_equal(exact_sdpa(q, kp, vp, None, D ** -0.5, m), solo).item(), p


def _window_mask(L, kL, window):
    """Causal sliding-window boolean mask for L queries ending at key kL - 1."""
    q = mx.arange(kL - L, kL).reshape(L, 1)
    k = mx.arange(kL).reshape(1, kL)
    return mx.logical_and(k <= q, k > q - window)


@pytest.mark.parametrize("L", [1, 5, 17, 70])
def test_realignment_preserves_values_for_any_boolean_mask(L):
    """Regression (fixed in 0.5.3): re-alignment shifted keys by the LAST query row's first valid key, which drops
    valid keys of earlier rows under a sliding-window mask. Exact attention must give stock's values (up to rounding)."""
    from mlx_exact.attention import exact_sdpa
    rng = np.random.default_rng(L)
    kL, H, D = 70, 4, 64
    q = mx.array(rng.standard_normal((1, H, L, D)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, H, kL, D)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, H, kL, D)).astype(np.float32)).astype(mx.bfloat16)
    m = _window_mask(L, kL, 32)
    ours = exact_sdpa(q, k, v, None, D ** -0.5, m).astype(mx.float32)
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                                               scale=D ** -0.5, mask=m)
    assert float(mx.max(mx.abs(ours - ref)).item()) < 3e-2


def test_additive_masks_pass_through_with_warning():
    import warnings
    from mlx_exact import attention
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((1, 2, 3, 64)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, 2, 20, 64)).astype(np.float32)).astype(mx.bfloat16)
    add = mx.array(rng.standard_normal((1, 2, 3, 20)).astype(np.float32)).astype(mx.bfloat16)
    attention._WARNED.discard("additive")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ours = attention.exact_sdpa(q, k, k, None, 0.125, add)
    assert any("additive" in str(r.message) for r in rec)
    assert bits_equal(ours, mx.fast.scaled_dot_product_attention(q, k, k, scale=0.125, mask=add)).item()


def test_sliding_window_model_values_match_stock():
    """A sliding-window model (gemma3_text, random weights, window 32): exact mode gives stock's values up to
    rounding, and decode is bit-identical to full prefill (here only the first 32 positions; see the next test)."""
    from mlx_lm.models import gemma3_text
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_exact
    cfg = dict(model_type="gemma3_text", hidden_size=256, num_hidden_layers=2, intermediate_size=512,
               num_attention_heads=4, num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6, vocab_size=1024,
               query_pre_attn_scalar=64, sliding_window=32, sliding_window_pattern=2, max_position_embeddings=2048)
    mx.random.seed(1)
    model = gemma3_text.Model(gemma3_text.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    ids = np.random.default_rng(0).integers(0, 1024, 70).tolist()
    stock = model(mx.array(ids)[None])[0].astype(mx.float32)
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        assert float(mx.max(mx.abs(full.astype(mx.float32) - stock)).item()) < 0.15
        cache = make_prompt_cache(model)
        for t in range(32):
            assert bits_equal(model(mx.array(ids[t:t + 1])[None], cache=cache)[0, -1], full[t]).item(), t
    finally:
        mlx_exact.disable()



@pytest.mark.parametrize("arch,bits", [("gemma3_text", None), ("gemma3_text", 4), ("gpt_oss", 4)])
def test_sliding_window_models_are_bit_exact(arch, bits):
    """Rotating KV caches (sliding-window layers): the window is put in temporal order and aligned to MLX's 32-key
    attention blocks, so decode and chunked prefill equal one full prefill at every position, also after the window
    has filled and the ring buffer has wrapped. gpt_oss also uses attention sinks; its experts are quantized here
    (unquantized MoE experts are not routed)."""
    import importlib
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_exact
    base = dict(hidden_size=256, num_hidden_layers=2, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
                head_dim=64, rms_norm_eps=1e-6, vocab_size=1024, rope_theta=10000.0, max_position_embeddings=2048,
                tie_word_embeddings=True, model_type=arch)
    extra = {"gemma3_text": dict(query_pre_attn_scalar=64, sliding_window=32, sliding_window_pattern=2),
             "gpt_oss": dict(num_local_experts=8, num_experts_per_tok=2, sliding_window=32)}[arch]
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(1)
    model = mod.Model(mod.ModelArgs.from_dict(dict(base, **extra)))
    model.set_dtype(mx.bfloat16)
    if bits:
        nn.quantize(model, group_size=64, bits=bits,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0)
    ids = np.random.default_rng(0).integers(0, 1024, 100).tolist()
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        for c in (1, 3, 17, 40):
            cache = make_prompt_cache(model)
            for s0 in range(0, len(ids), c):
                out = model(mx.array(ids[s0:s0 + c])[None], cache=cache)[0]
                for i in range(out.shape[0]):
                    assert bits_equal(out[i], full[s0 + i]).item(), (arch, bits, c, s0 + i)
    finally:
        mlx_exact.disable()


def test_fixed_order_attention_accuracy():
    """mlx-exact's fixed-order attention (head dims 192/256 with one KV head) matches an fp32 reference to bf16 level."""
    from mlx_exact.kernels import dense_attention_exact
    rng = np.random.default_rng(0)
    q = mx.array(rng.standard_normal((1, 4, 37, 256)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, 1, 300, 256)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, 1, 300, 256)).astype(np.float32)).astype(mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                                               scale=0.0625, mask="causal")
    ours = dense_attention_exact(q, k, v, 0.0625, "causal").astype(mx.float32)
    assert float(mx.max(mx.abs(ours - ref)).item()) < 3e-3


@pytest.mark.parametrize("bits", [None, 4])
def test_single_kv_head_256_model_is_bit_exact(bits):
    """Head dim 256 with one KV head (Gemma-3-1B-like), sliding window 32: decode and chunks equal full prefill."""
    import mlx.nn as nn
    from mlx_lm.models import gemma3_text
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_exact
    cfg = dict(model_type="gemma3_text", hidden_size=256, num_hidden_layers=2, intermediate_size=512,
               num_attention_heads=2, num_key_value_heads=1, head_dim=256, rms_norm_eps=1e-6, vocab_size=1024,
               rope_theta=10000.0, max_position_embeddings=4096, tie_word_embeddings=True, sliding_window=32,
               sliding_window_pattern=2, query_pre_attn_scalar=256)
    mx.random.seed(1)
    model = gemma3_text.Model(gemma3_text.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    if bits:
        nn.quantize(model, group_size=64, bits=bits)
    ids = np.random.default_rng(0).integers(0, 1024, 100).tolist()
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        for c in (1, 3, 17, 40):
            cache = make_prompt_cache(model)
            for s0 in range(0, len(ids), c):
                out = model(mx.array(ids[s0:s0 + c])[None], cache=cache)[0]
                for i in range(out.shape[0]):
                    assert bits_equal(out[i], full[s0 + i]).item(), (bits, c, s0 + i)
    finally:
        mlx_exact.disable()
