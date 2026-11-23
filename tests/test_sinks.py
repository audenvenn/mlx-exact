"""Attention sinks (gpt-oss) on the offsets route: bit-identical to the padded path that runs stock MLX's own kernel
with sinks, for decode, short chunks and left-padded batches; the reference backward includes the sink logit."""
import numpy as np
import mlx.core as mx
from mlx_exact.testing import bits_equal
import pytest

import mlx_exact.attention as A
from mlx_exact.attention import exact_sdpa


def _mk(rng, *s, dt=mx.bfloat16):
    return mx.array(rng.standard_normal(s).astype(np.float32)).astype(dt)


def _reference(*args, **kw):
    saved = A.OFFSETS_HEAD_DIMS
    A.OFFSETS_HEAD_DIMS = ()
    try:
        return exact_sdpa(*args, **kw)
    finally:
        A.OFFSETS_HEAD_DIMS = saved


@pytest.mark.parametrize("D,Hq,Hkv", [(64, 8, 2), (128, 16, 8)])
@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
def test_sinks_offsets_route_matches_the_stock_kernel_path(D, Hq, Hkv, dt):
    from mlx_lm.models.base import create_causal_mask
    rng = np.random.default_rng(D + Hq)
    sinks = mx.array(rng.standard_normal(Hq).astype(np.float32) * 2).astype(dt)   # stock needs the query dtype
    sc = D ** -0.5
    for L, kL, mask in ((1, 1, None), (1, 9, None), (1, 100, None), (3, 40, "causal"), (8, 8, "causal")):
        q, k, v = _mk(rng, 1, Hq, L, D, dt=dt), _mk(rng, 1, Hkv, kL, D, dt=dt), _mk(rng, 1, Hkv, kL, D, dt=dt)
        A.STATS.clear()
        out = exact_sdpa(q, k, v, None, sc, mask, sinks=sinks)
        assert A.STATS.get("sdpa_offsets", 0) == 1
        assert bits_equal(out, _reference(q, k, v, None, sc, mask, sinks=sinks)).item(), (L, kL, mask)
    B, kL = 3, 60
    for L in (1, 4, 16):
        q, k, v = _mk(rng, B, Hq, L, D, dt=dt), _mk(rng, B, Hkv, kL, D, dt=dt), _mk(rng, B, Hkv, kL, D, dt=dt)
        m = create_causal_mask(L, offset=kL - L, left_padding=mx.array([0, 5, 33]))
        valid = mx.broadcast_to(mx.any(m, axis=-1)[..., None], (B, Hq, L, D))
        out = exact_sdpa(q, k, v, None, sc, m, sinks=sinks)
        ref = _reference(q, k, v, None, sc, m, sinks=sinks)
        assert bits_equal(mx.where(valid, out, 0), mx.where(valid, ref, 0)).item(), L
        for b, p in enumerate([0, 5, 33]):
            alone = _reference(q[b:b + 1], k[b:b + 1, :, p:], v[b:b + 1, :, p:], None, sc,
                               "causal" if L > 1 else None, sinks=sinks)
            assert bits_equal(mx.where(valid[b:b + 1], out[b:b + 1], 0), mx.where(valid[b:b + 1], alone, 0)).item()


def test_reference_attention_with_sinks_matches_stock_and_differentiates():
    rng = np.random.default_rng(3)
    B, Hq, Hkv, L, kL, D = 1, 8, 2, 12, 30, 64
    q, k, v = _mk(rng, B, Hq, L, D), _mk(rng, B, Hkv, kL, D), _mk(rng, B, Hkv, kL, D)
    sinks = mx.array(rng.standard_normal(Hq).astype(np.float32)).astype(mx.bfloat16)
    ref = A.reference_attention(q, k, v, D ** -0.5, "causal", sinks).astype(mx.float32)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal", sinks=sinks).astype(mx.float32)
    assert float(mx.max(mx.abs(ref - stock)).item()) < 2e-2
    cot = _mk(rng, B, Hq, 4, D)
    _, g = mx.vjp(lambda a, b, c, s: exact_sdpa(a, b, c, None, D ** -0.5, "causal", sinks=s),
                  [q[:, :, :4], k, v, sinks], [cot])
    assert all(bool(mx.all(mx.isfinite(x.astype(mx.float32))).item()) for x in g)
    assert float(mx.max(mx.abs(g[3])).item()) > 0          # the sinks get a gradient
