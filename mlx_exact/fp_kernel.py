"""Exact MX-format (mxfp4) quantized matmul: MLX's own fp_qmm_t_nax implementation (fp_quantized_nax.h, vendored
verbatim from v0.32.4, see _vendor/) run with a 16-row tile, like the affine v4 kernel (nax_kernel.py).

Stock MLX runs fp_qmm_t_nax with 64 x 64 x 64 tiles and 2 x 2 SIMD groups for long prefills; that is the canonical
arithmetic. Here BM = 16 and WM = 1: every SIMD group still owns 32-column pairs (the 16x32x16 accelerator instruction,
FINDINGS F36), the K loop (64-wide tiles, 32-wide steps, 16-wide fragments) and the dequantization (e8m0 scale x e2m1
value in fp32, rounded to bf16 in threadgroup memory) are unchanged, so each output element is computed exactly as in
the prefill kernel, for any number of rows.

Edits to MLX's code, made when the header is assembled: includes and the three function-constant declarations are
dropped, and the impl takes K, N, M by value instead of `constant int&`.
"""
from pathlib import Path

import mlx.core as mx

TP = Path(__file__).resolve().parent / "_vendor" / "mlx_v0_32_4"


def _strip(txt):
    return "\n".join(l for l in txt.splitlines() if not l.startswith('#include "mlx/') and not l.startswith("#pragma once"))


def _fp_header():
    parts = [_strip((TP / f).read_text()) for f in ("defines.h", "type_traits.h", "integral_constant.h", "nax.h",
                                                    "fp4.h", "fp8.h")]
    lines = (TP / "fp_quantized_nax.h").read_text().splitlines()
    end = next(i for i, l in enumerate(lines) if l.startswith("METAL_FUNC void fp_qmm_n_impl("))
    end = max(i for i in range(end) if lines[i] == "template <")            # the template line of fp_qmm_n_impl
    q = [l for l in lines[:end] if not l.startswith('#include "mlx/') and not l.startswith("constant bool align_")]
    qs = "\n".join(q)
    for v in ("K", "N", "M"):
        assert qs.count(f"const constant int& {v},") == 1, v
        qs = qs.replace(f"const constant int& {v},", f"const int {v},")
    assert "METAL_FUNC void fp_qmm_t_impl(" in qs
    parts.append(qs)
    return "\n\n".join(parts) + "\n"


_SRC = r"""
    threadgroup bfloat Ws[64 * (64 + 16 / sizeof(bfloat))];
    fp_qmm_t_impl<T, GS, BITS, ALIGNED_N, false, 16, 64, 64, 1, 2, bfloat>(
        w, scales, static_cast<const device float*>(nullptr), x, out, Ws, int(dims[0]), int(dims[1]), int(dims[2]),
        threadgroup_position_in_grid, thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_k = {}


def fp_qmm_exact_v4(x, w, scales, group_size=32, bits=4):
    """x: [M, K] (bf16 / fp16); w: [N, K * bits / 32] uint32 (mxfp4: bits 4, group size 32); scales: [N, K / 32]
    uint8 (e8m0). Output identical (expected bit for bit) to stock fp_qmm_t_nax with 64-row tiles."""
    assert bits == 4 and group_size == 32, "mxfp4 only"
    N, kp = w.shape
    K = kp * (32 // bits)
    assert K % 64 == 0
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "fp4" not in _k:
        _k["fp4"] = mx.fast.metal_kernel(
            name="fp_qmm_exact_v4", input_names=["x", "w", "scales", "dims"], output_names=["out"],
            source=_SRC, header=_fp_header(), compile_options={"math_mode": "safe"})
    y = _k["fp4"](inputs=[x2, w, scales, mx.array([K, N, M], dtype=mx.int32)],
                  template=[("T", x.dtype), ("ALIGNED_N", N % 64 == 0), ("BITS", bits), ("GS", group_size)],
                  grid=(((N + 63) // 64) * 64, (M + 15) // 16, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y


# Expert-indexed variant (MoE): pair p = (row p of x, expert idx[p]). Each threadgroup runs MLX's impl on ONE row with
# that expert's weights, so every output element has the same arithmetic as the dense kernel (and as stock's sorted
# expert kernel, fp_gather_qmm_rhs_nax).
_SRC_GATHER = r"""
    const uint p = threadgroup_position_in_grid.y;
    const int K = int(dims[0]);
    const int N = int(dims[1]);
    const size_t e = size_t(idx[p]);
    threadgroup bfloat Ws[64 * (64 + 16 / sizeof(bfloat))];
    fp_qmm_t_impl<T, GS, BITS, ALIGNED_N, false, 16, 64, 64, 1, 2, bfloat>(
        w + e * size_t(N) * size_t(K) * BITS / 32, scales + e * size_t(N) * size_t(K / GS),
        static_cast<const device float*>(nullptr), x + size_t(p) * K, out + size_t(p) * N, Ws, K, N, 1,
        uint3(threadgroup_position_in_grid.x, 0, 0), thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


def fp_gather_exact_v4(x, w, scales, idx, group_size=32, bits=4):
    """x: [P, K]; w: [E, N, K * bits / 32] uint32; scales: [E, N, K / 32] uint8; idx: [P] expert per row.
    Row p of the output = x[p] @ dequant(w[idx[p]]).T with the canonical (prefill) arithmetic."""
    assert bits == 4 and group_size == 32, "mxfp4 only"
    E, N, kp = w.shape
    K = kp * (32 // bits)
    assert K % 64 == 0
    P = x.shape[0]
    if "fp4g" not in _k:
        _k["fp4g"] = mx.fast.metal_kernel(
            name="fp_gather_exact_v4", input_names=["x", "w", "scales", "idx", "dims"], output_names=["out"],
            source=_SRC_GATHER, header=_fp_header(), compile_options={"math_mode": "safe"})
    return _k["fp4g"](inputs=[x, w, scales, idx.astype(mx.uint32), mx.array([K, N], dtype=mx.int32)],
                      template=[("T", x.dtype), ("ALIGNED_N", N % 64 == 0), ("BITS", bits), ("GS", group_size)],
                      grid=(((N + 63) // 64) * 64, P, 1), threadgroup=(64, 1, 1),
                      output_shapes=[(P, N)], output_dtypes=[x.dtype])[0]
