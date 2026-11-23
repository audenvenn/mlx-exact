"""The fixed-order attention kernels (head dims 192/256 with one KV head) of 0.8.1 (register-blocked, tiled, one SIMD
group per softmax row) compute exactly the arithmetic of the one-thread-per-output kernels of 0.7.0-0.8.0, which are
kept here as the reference: every output bit-identical."""
import numpy as np
import mlx.core as mx
from mlx_exact.testing import bits_equal
import pytest

from mlx_exact.kernels import dense_attention_exact

_REF_SCORES = r"""
    const uint j = thread_position_in_grid.x;
    const uint l = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    if (int(j) >= KL) { return; }
    const uint b = bh / HQ, h = bh % HQ, kvh = h / (HQ / HKV);
    auto qr = q + (size_t(bh) * L + l) * D;
    auto kr = k + ((size_t(b) * HKV + kvh) * KL + j) * D;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) {
        acc = fma(float(qr[d]), float(kr[d]), acc);
    }
    out[(size_t(bh) * L + l) * KL + j] = acc * float(scale[0]);
"""
_REF_SOFTMAX = r"""
    const uint l = thread_position_in_grid.x;
    const uint bh = thread_position_in_grid.y;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    if (int(l) >= L) { return; }
    auto row = s + (size_t(bh) * L + l) * KL;
    auto prow = out + (size_t(bh) * L + l) * KL;
    float m = -INFINITY;
    for (int j = 0; j < KL; j++) { m = max(m, float(row[j])); }
    float sum = 0.0f;
    for (int j = 0; j < KL; j++) {
        const float e = (row[j] == -INFINITY) ? 0.0f : exp(float(row[j]) - m);
        prow[j] = e;
        sum += e;
    }
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (int j = 0; j < KL; j++) { prow[j] = prow[j] * inv; }
"""
_REF_OUT = r"""
    const uint d = thread_position_in_grid.x;
    const uint l = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    if (int(d) >= D) { return; }
    const uint b = bh / HQ, h = bh % HQ, kvh = h / (HQ / HKV);
    auto pr = p + (size_t(bh) * L + l) * KL;
    auto vr = v + (size_t(b) * HKV + kvh) * size_t(KL) * D + d;
    float acc = 0.0f;
    for (int j = 0; j < KL; j++) {
        acc = fma(pr[j], float(vr[size_t(j) * D]), acc);
    }
    out[(size_t(bh) * L + l) * D + d] = static_cast<T>(acc);
"""
_K = {}


def _reference(queries, keys, values, scale, mask):
    if not _K:
        opts = {"math_mode": "safe"}
        _K["s"] = mx.fast.metal_kernel(name="ref_datt_s", input_names=["q", "k", "dims", "scale"], output_names=["out"],
                                       source=_REF_SCORES, compile_options=opts)
        _K["m"] = mx.fast.metal_kernel(name="ref_datt_m", input_names=["s", "dims"], output_names=["out"],
                                       source=_REF_SOFTMAX, compile_options=opts)
        _K["o"] = mx.fast.metal_kernel(name="ref_datt_o", input_names=["p", "v", "dims"], output_names=["out"],
                                       source=_REF_OUT, compile_options=opts)
    B, HQ, L, D = queries.shape
    HKV, KL = keys.shape[1], keys.shape[2]
    dims = mx.array([L, KL], dtype=mx.int32)
    tmpl = [("T", queries.dtype), ("HQ", HQ), ("HKV", HKV), ("D", D)]
    s = _K["s"](inputs=[queries, keys, dims, mx.array([scale], dtype=mx.float32)], template=tmpl, grid=(KL, L, B * HQ),
                threadgroup=(min(256, KL), 1, 1), output_shapes=[(B, HQ, L, KL)], output_dtypes=[mx.float32])[0]
    if mask is not None:
        if isinstance(mask, str):
            mask = mx.arange(KL)[None, :] <= (mx.arange(L)[:, None] + KL - L)
        s = mx.where(mask, s, -mx.inf)
    p = _K["m"](inputs=[s, dims], template=[("T", queries.dtype)], grid=(L, B * HQ, 1), threadgroup=(min(64, L), 1, 1),
                output_shapes=[(B, HQ, L, KL)], output_dtypes=[mx.float32])[0]
    return _K["o"](inputs=[p, values, dims], template=tmpl, grid=(D, L, B * HQ), threadgroup=(min(256, D), 1, 1),
                   output_shapes=[(B, HQ, L, D)], output_dtypes=[queries.dtype])[0]


@pytest.mark.parametrize("D", [256, 192])
@pytest.mark.parametrize("L,KL,mask", [(1, 1, None), (1, 37, None), (5, 40, "causal"), (33, 33, "causal"),
                                       (100, 260, "causal"), (64, 64, "array"), (70, 75, "empty rows")])
@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
def test_fixed_order_kernels_are_bit_identical_to_the_reference(D, L, KL, mask, dt):
    rng = np.random.default_rng(L * 1000 + KL + D)
    q, k, v = (mx.array(rng.standard_normal(s).astype(np.float32)).astype(dt)
               for s in ((2, 4, L, D), (2, 1, KL, D), (2, 1, KL, D)))
    m = mask
    if mask == "array":
        m = mx.broadcast_to(mx.array(np.arange(KL)[None, None, None, :] >= np.array([0, 9])[:, None, None, None]),
                            (2, 1, L, KL))
    elif mask == "empty rows":                       # left padding: the first rows of sequence 1 see no key
        m = mx.array((np.arange(KL)[None, None, :] <= (np.arange(L)[None, :, None] + KL - L))
                     & (np.arange(KL)[None, None, :] >= np.array([0, 20])[:, None, None]))[:, None]
    out = dense_attention_exact(q, k, v, D ** -0.5, m)
    assert bits_equal(out, _reference(q, k, v, D ** -0.5, m), equal_nan=True).item()
