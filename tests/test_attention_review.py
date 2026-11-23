"""Regression tests from an independent pre-release review of 0.8.0: masks that broadcast along the key axis or are
not contiguous, inputs whose head dimension is not contiguous, masks changed in place, additive masks with a quantized
KV cache, rows without any valid key (NaN in the fixed-order route before 0.8.0, which reached the real rows of later
layers in a left-padded batch), input validation, mixed dtypes, and unrestricted attention on the padded fallback.
Gradients are checked against an independent oracle: stock attention's VJP with the queries padded to 9 rows."""
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
from mlx_exact.testing import bits_equal
import pytest

import mlx_exact.attention as A
from mlx_exact.attention import exact_sdpa
from mlx_exact.attn_nax import attention_offsets


def _mk(rng, *s, dt=mx.bfloat16):
    return mx.array(rng.standard_normal(s).astype(np.float32)).astype(dt)


def _stock_f32(q, k, v, sc, mask):
    m = mask if (mask is None or isinstance(mask, str) or mask.dtype == mx.bool_) else mask.astype(mx.float32)
    return mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                                                scale=sc, mask=m)


def _close(a, b, tol=2e-2):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return a.shape == b.shape and float(mx.max(mx.abs(a - b)).item()) <= tol * (float(mx.max(mx.abs(b)).item()) + 1e-3)


def _stock_vjp(q, k, v, sc, mask, cot, P=9):
    """Independent gradient oracle: stock attention's VJP, queries padded to P rows (placed first, zero cotangent)."""
    L = q.shape[2]
    pad = max(0, P - L)
    qp = mx.concatenate([mx.repeat(q[:, :, :1], pad, axis=2), q], axis=2) if pad else q
    cp = mx.concatenate([mx.zeros((*cot.shape[:2], pad, cot.shape[3]), cot.dtype), cot], axis=2) if pad else cot
    if mask is None or isinstance(mask, str):
        mp = mask
    else:
        m4 = mask if mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
        mp = m4 if (m4.shape[-2] == 1 or not pad) else mx.concatenate([mx.repeat(m4[..., :1, :], pad, axis=-2), m4], axis=-2)
    _, g = mx.vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(a, b, c, scale=sc, mask=mp), [qp, k, v], [cp])
    return g[0][:, :, pad:], g[1], g[2]


# --- masks ---------------------------------------------------------------------------------------------------------
def test_masks_broadcast_along_keys_or_not_contiguous():
    rng = np.random.default_rng(0)
    B, Hq, Hkv, L, kL, D = 2, 16, 8, 1, 40, 128
    q, k, v = _mk(rng, B, Hq, L, D), _mk(rng, B, Hkv, kL, D), _mk(rng, B, Hkv, kL, D)
    sc = D ** -0.5
    full = exact_sdpa(q, k, v, None, sc, mx.ones((B, 1, L, kL), mx.bool_))
    assert _close(full, _stock_f32(q, k, v, sc, None))
    row = mx.array(np.arange(kL) >= 5)[None, None, None, :]                       # 5 masked leading keys
    wide = mx.repeat(row, 2, axis=-1)                                             # stepped view below = row
    cases = {"scalar True": (mx.array(True), mx.ones((B, 1, L, kL), mx.bool_)),
             "key axis of size 1": (mx.ones((B, 1, L, 1), mx.bool_), mx.ones((B, 1, L, kL), mx.bool_)),
             "broadcast view": (mx.broadcast_to(row, (B, 1, L, kL)), mx.array(np.broadcast_to(np.array(row), (B, 1, L, kL)))),
             "stepped view": (wide[..., ::2], mx.array(np.array(wide)[..., ::2]))}
    for name, (m, materialized) in cases.items():
        out = exact_sdpa(q, k, v, None, sc, m)
        assert bits_equal(out, exact_sdpa(q, k, v, None, sc, materialized)).item(), name
        assert _close(out, _stock_f32(q, k, v, sc, materialized)), name


def test_a_mask_changed_in_place_is_read_again():
    rng = np.random.default_rng(1)
    B, Hq, Hkv, L, kL, D = 1, 16, 8, 1, 20, 128
    q, k, v = _mk(rng, B, Hq, L, D), _mk(rng, B, Hkv, kL, D), _mk(rng, B, Hkv, kL, D)
    m = mx.array(np.arange(kL) >= 3)[None, None, None, :]
    first = exact_sdpa(q, k, v, None, D ** -0.5, m)
    mx.eval(first)
    m[..., 0] = True                                                              # same Python object, new contents
    fresh = mx.array(np.array(m))
    assert bits_equal(exact_sdpa(q, k, v, None, D ** -0.5, m), exact_sdpa(q, k, v, None, D ** -0.5, fresh)).item()
    assert not bits_equal(first, exact_sdpa(q, k, v, None, D ** -0.5, fresh)).item()


# --- inputs ----------------------------------------------------------------------------------------------------------
def test_inputs_with_a_non_contiguous_head_dimension():
    rng = np.random.default_rng(2)
    B, Hq, Hkv, L, kL, D = 1, 16, 8, 3, 30, 128
    q = _mk(rng, B, Hq, L, D)
    kw, vw = _mk(rng, B, Hkv, kL, 2 * D), _mk(rng, B, Hkv, kL, 2 * D)
    k, v = kw[..., ::2], vw[..., 1::2]                                            # head-dim stride 2
    qt = _mk(rng, B, Hq, D, L).swapaxes(-1, -2)                                   # head dim not last in memory
    for qq in (q, qt):
        ref = exact_sdpa(mx.contiguous(qq), mx.contiguous(k), mx.contiguous(v), None, D ** -0.5, "causal")
        assert bits_equal(exact_sdpa(qq, k, v, None, D ** -0.5, "causal"), ref).item()


def test_kv_cache_slices_are_read_in_place():
    rng = np.random.default_rng(3)
    B, Hq, Hkv, kL, D = 3, 16, 8, 77, 128
    bk, bv = _mk(rng, B, Hkv, 256, D), _mk(rng, B, Hkv, 256, D)
    k, v = bk[:, :, :kL], bv[:, :, :kL]
    q = _mk(rng, B, Hq, 1, D)
    pads = mx.array([0, 5, 40], mx.int32)
    m = mx.array(np.arange(kL)[None, :] >= np.array([0, 5, 40])[:, None])[:, None, None, :]
    inplace = attention_offsets(q, k, v, D ** -0.5, m, pads, kv_inplace=True)
    assert bits_equal(inplace, attention_offsets(q, mx.contiguous(k), mx.contiguous(v), D ** -0.5, m, pads)).item()


def test_invalid_inputs_are_rejected():
    rng = np.random.default_rng(4)
    q, k = _mk(rng, 1, 16, 1, 128), _mk(rng, 1, 8, 30, 128)
    with pytest.raises(ValueError):
        attention_offsets(q, k, _mk(rng, 1, 8, 29, 128), 0.1)                     # V shorter than K
    with pytest.raises(ValueError):
        attention_offsets(q, _mk(rng, 1, 8, 30, 64), _mk(rng, 1, 8, 30, 128), 0.1)  # K head dim differs
    with pytest.raises(ValueError):
        attention_offsets(q, k, k, 0.1, mx.ones((1, 1, 1, 7), mx.bool_))          # mask key axis does not broadcast


def test_mixed_value_dtype_falls_back_to_stock_promotion():
    rng = np.random.default_rng(5)
    q, k = _mk(rng, 1, 16, 1, 128), _mk(rng, 1, 8, 30, 128)
    v = _mk(rng, 1, 8, 30, 128, dt=mx.float32)
    assert _close(exact_sdpa(q, k, v, None, 128 ** -0.5, None), _stock_f32(q, k, v, 128 ** -0.5, None))


def test_unrestricted_attention_stays_unrestricted_on_the_padded_path():
    """mask=None with several queries means every query sees every key (head dim 192: the padded fallback)."""
    rng = np.random.default_rng(6)
    for kL in (16, 5):                                                            # padded and square paths
        q, k, v = _mk(rng, 1, 2, 2, 192), _mk(rng, 1, 2, kL, 192), _mk(rng, 1, 2, kL, 192)
        assert _close(exact_sdpa(q, k, v, None, 192 ** -0.5, None), _stock_f32(q, k, v, 192 ** -0.5, None)), kL


# --- rows without a valid key -----------------------------------------------------------------------------------------
def test_rows_without_a_valid_key_stay_finite_and_get_no_gradient():
    rng = np.random.default_rng(7)
    B, L, kL = 2, 12, 30
    for D, Hq, Hkv in ((128, 16, 8), (256, 4, 1)):                                # offsets route, fixed-order route
        q, k, v = _mk(rng, B, Hq, L, D), _mk(rng, B, Hkv, kL, D), _mk(rng, B, Hkv, kL, D)
        lp = np.array([0, 22])                                                    # sequence 1: rows 0-3 are padding
        m = mx.array((np.arange(kL)[None, None, :] <= (np.arange(L)[None, :, None] + kL - L))
                     & (np.arange(kL)[None, None, :] >= lp[:, None, None]))[:, None]
        valid = mx.any(m, axis=-1)[..., None]                                     # [B, 1, L, 1]
        out = exact_sdpa(q, k, v, None, D ** -0.5, m)
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item()), D
        ref = _stock_f32(q, k, v, D ** -0.5, m)
        assert _close(mx.where(valid, out, 0), mx.where(valid, ref, 0)), D
        cot = _mk(rng, B, Hq, L, D)
        _, g = mx.vjp(lambda a, b, c: exact_sdpa(a, b, c, None, D ** -0.5, m), [q, k, v], [cot])
        r = _stock_vjp(q, k, v, D ** -0.5, m, mx.where(valid, cot, 0))            # empty rows contribute nothing
        for x, y in zip(g, r):
            assert bool(mx.all(mx.isfinite(x.astype(mx.float32))).item()) and _close(x, y), D
        assert float(mx.max(mx.abs(g[2][1:, :, :22].astype(mx.float32))).item()) == 0.0   # padding values: no gradient


def test_left_padded_batch_with_head_dim_256_and_one_kv_head_matches_each_sequence_alone():
    """0.7.x: padding rows gave NaN in the fixed-order route; in the next layer the padding values were NaN and
    0 * NaN reached every real row. Two layers, random weights, no download."""
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import BatchKVCache, make_prompt_cache
    import mlx_exact
    mx.random.seed(0)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(dict(
        model_type="qwen3", hidden_size=256, num_hidden_layers=2, intermediate_size=512, num_attention_heads=2,
        num_key_value_heads=1, head_dim=256, rms_norm_eps=1e-6, vocab_size=512, rope_theta=1e6,
        max_position_embeddings=2048, tie_word_embeddings=True)))
    model.set_dtype(mx.bfloat16)
    prompts = [list(range(3, 15)), list(range(40, 47))]
    n = max(len(p) for p in prompts)
    pads = [n - len(p) for p in prompts]
    mlx_exact.enable()
    try:
        solo_caches = [make_prompt_cache(model) for _ in prompts]
        solo = [model(mx.array(p)[None], cache=c)[0, -1] for p, c in zip(prompts, solo_caches)]
        caches = [BatchKVCache(pads) for _ in model.layers]
        batch = model(mx.array([[0] * d + p for p, d in zip(prompts, pads)]), cache=caches)[:, -1]
        assert A.STATS.get("sdpa_fixed_order", 0) > 0
        for step in range(3):
            for b in range(len(prompts)):
                assert bool(mx.all(mx.isfinite(batch[b].astype(mx.float32))).item()), (step, b)
                assert bits_equal(batch[b], solo[b]).item(), (step, b)
            nxt = [int(mx.argmax(s).item()) for s in solo]
            solo = [model(mx.array([[t]]), cache=c)[0, -1] for t, c in zip(nxt, solo_caches)]
            batch = model(mx.array(nxt)[:, None], cache=caches)[:, -1]
    finally:
        mlx_exact.disable()


# --- gradients against the independent oracle ------------------------------------------------------------------------
@pytest.mark.parametrize("D,Hq,Hkv", [(128, 16, 8), (64, 8, 2), (256, 8, 2), (256, 4, 1)])
@pytest.mark.parametrize("L", [1, 4, 12])
def test_gradients_match_stock_attention(D, Hq, Hkv, L):
    rng = np.random.default_rng(D + L)
    B, kL, sc = 2, 30, D ** -0.5
    q, k, v = _mk(rng, B, Hq, L, D), _mk(rng, B, Hkv, kL, D), _mk(rng, B, Hkv, kL, D)
    lp = np.array([0, 7])
    m = mx.array((np.arange(kL)[None, None, :] <= (np.arange(L)[None, :, None] + kL - L))
                 & (np.arange(kL)[None, None, :] >= lp[:, None, None]))[:, None]
    for mask in (m, "causal"):
        cot = _mk(rng, B, Hq, L, D)
        _, g = mx.vjp(lambda a, b, c: exact_sdpa(a, b, c, None, sc, mask), [q, k, v], [cot])
        r = _stock_vjp(q, k, v, sc, mask, cot)
        assert all(_close(x, y) for x, y in zip(g, r)), (D, L, mask if isinstance(mask, str) else "array")


@pytest.mark.parametrize("additive", [None, "zeros", "random"])
def test_quantized_kv_query_gradient(additive):
    rng = np.random.default_rng(8)
    B, Hq, Hkv, L, kL, D = 1, 16, 8, 3, 40, 128
    q, k, v = _mk(rng, B, Hq, L, D), _mk(rng, B, Hkv, kL, D), _mk(rng, B, Hkv, kL, D)
    kq, vq = mx.quantize(k, group_size=64, bits=8), mx.quantize(v, group_size=64, bits=8)
    kd, vd = mx.dequantize(*kq, group_size=64, bits=8), mx.dequantize(*vq, group_size=64, bits=8)
    if additive is None:
        mask = "causal"
    else:
        mask = mx.zeros((1, 1, L, kL), mx.bfloat16) if additive == "zeros" else _mk(rng, 1, 1, L, kL)
    cache = SimpleNamespace(bits=8, group_size=64)
    cot = _mk(rng, B, Hq, L, D)
    _, (g,) = mx.vjp(lambda a: exact_sdpa(a, kq, vq, cache, D ** -0.5, mask), [q], [cot])
    r = _stock_vjp(q, kd, vd, D ** -0.5, mask, cot)[0]
    assert _close(g, r), additive


# --- second review round ----------------------------------------------------------------------------------------------
def test_k_v_returned_by_a_cache_with_a_strided_head_dimension():
    """A cache can return the caller's own K/V (a rotating cache's first multi-token update; a KV cache whose first
    update exactly fills its buffer). Exact mode stores contiguous K/V, so in-place reads stay right."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    import mlx_exact
    rng = np.random.default_rng(9)
    H, D = 8, 64
    mlx_exact.enable()
    try:
        for name, L, mask in (("rotating", 2, "causal"), ("full first buffer", 16, "array")):
            cache = RotatingKVCache(max_size=32) if name == "rotating" else KVCache()
            if name != "rotating":
                cache.step = 16
            q, k = _mk(rng, 1, 16, L, D), _mk(rng, 1, H, L, D)
            v = _mk(rng, 1, H, L, 2 * D)[..., ::2]                                # head-dim stride 2
            m = mask if mask == "causal" else mx.array(np.tril(np.ones((L, L), bool)))[None, None]
            kk, vv = cache.update_and_fetch(k, v)
            out = exact_sdpa(q, kk, vv, cache, D ** -0.5, m)
            ref = exact_sdpa(q, mx.contiguous(k), mx.contiguous(v), None, D ** -0.5, m)
            assert bits_equal(out, ref).item(), name
    finally:
        mlx_exact.disable()


def test_an_explicit_mask_that_shows_keys_before_the_cache_padding():
    from mlx_lm.models.cache import BatchKVCache
    rng = np.random.default_rng(10)
    cache = BatchKVCache([1])                                                     # key 0 recorded as padding
    q, k, v = _mk(rng, 1, 16, 1, 128), _mk(rng, 1, 8, 2, 128), _mk(rng, 1, 8, 2, 128)
    kk, vv = cache.update_and_fetch(k, v)
    m = mx.ones((1, 1, 1, 2), mx.bool_)                                           # ...but the caller shows it
    assert _close(exact_sdpa(q, kk, vv, cache, 128 ** -0.5, m), _stock_f32(q, kk, vv, 128 ** -0.5, m))


def test_more_queries_than_keys():
    rng = np.random.default_rng(11)
    for D, Hq, Hkv, mask in ((192, 2, 2, None), (64, 8, 2, "causal")):
        q, k, v = _mk(rng, 1, Hq, 3, D), _mk(rng, 1, Hkv, 2, D), _mk(rng, 1, Hkv, 2, D)
        out = exact_sdpa(q, k, v, None, D ** -0.5, mask)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=mask)
        assert out.shape == (1, Hq, 3, D) and bits_equal(out, ref, equal_nan=True).item(), (D, mask)


def test_rotating_cache_with_a_mask_broadcast_along_keys():
    from mlx_lm.models.cache import RotatingKVCache
    import mlx_exact
    rng = np.random.default_rng(12)
    mlx_exact.enable()
    try:
        cache = RotatingKVCache(max_size=32)
        for _ in range(34):                                                       # wraps the 32-slot ring
            kk, vv = cache.update_and_fetch(_mk(rng, 1, 8, 1, 128), _mk(rng, 1, 8, 1, 128))
        q = _mk(rng, 1, 16, 1, 128)
        one = exact_sdpa(q, kk, vv, cache, 128 ** -0.5, mx.ones((1, 1, 1, 1), mx.bool_))
        full = exact_sdpa(q, kk, vv, cache, 128 ** -0.5, mx.ones((1, 1, 1, kk.shape[2]), mx.bool_))
        assert bits_equal(one, full).item()
    finally:
        mlx_exact.disable()


def test_direct_helper_edge_cases():
    rng = np.random.default_rng(13)
    q = _mk(rng, 1, 16, 1, 128)
    with pytest.raises(ValueError):
        attention_offsets(q, mx.zeros((1, 0, 4, 128), mx.bfloat16), mx.zeros((1, 0, 4, 128), mx.bfloat16), 0.1)
    out = attention_offsets(mx.zeros((0, 16, 1, 128), mx.bfloat16), mx.zeros((0, 8, 4, 128), mx.bfloat16),
                            mx.zeros((0, 8, 4, 128), mx.bfloat16), 0.1)
    assert out.shape == (0, 16, 1, 128)
