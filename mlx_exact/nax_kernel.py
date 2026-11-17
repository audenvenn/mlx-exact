"""v4 exact decode kernel: MLX's own qmm_t_nax implementation (vendored verbatim from v0.32.4, see third_party/), run
with a 16-row tile so that a single activation row (decode) or any small M uses the SAME accelerator instruction
(mpp matmul2d 16x32x16, relaxed precision), the same dequantization (T-rounded, threadgroup), the same 64-wide K tiles
and the same fragment order as MLX's 64-row prefill kernel. Exact by construction if the per-element arithmetic of the
instruction does not depend on the other rows of the tile (verified empirically; see FINDINGS).
Only edits to MLX's code: the three unused function-constant declarations and the two adjust_matrix_offsets helpers
are dropped, and the impl takes K, N, M by value instead of `constant int&`.
"""
from pathlib import Path

import mlx.core as mx

TP = Path(__file__).resolve().parent / "_vendor" / "mlx_v0_32_4"


def _header():
    parts = []
    for f in ("defines.h", "type_traits.h", "integral_constant.h", "nax.h"):
        txt = (TP / f).read_text()
        parts.append("\n".join(l for l in txt.splitlines() if not l.startswith('#include "mlx/')
                               and not l.startswith("#pragma once")))
    q = (TP / "quantized_nax.h").read_text().splitlines()[:1166]       # helpers, loaders, qmm_{t,n}_nax_tgp_impl
    q = q[:929] + q[1021:]          # drop the two adjust_matrix_offsets overloads (lines 930-1021; batching helpers that
    #                                 need elem_to_loc_broadcast, used only by MLX's top-level kernels, not by the impl)
    q = [l for l in q if not l.startswith("constant bool align_")]
    qs = "\n".join(q)
    for v in ("K", "N", "M"):
        qs = qs.replace(f"const constant int& {v},", f"const int {v},")
    parts.append(qs)
    return "\n\n".join(parts)


_SRC = r"""
    threadgroup T Ws[64 * (64 + 16 / sizeof(T))];
    qmm_t_nax_tgp_impl<T, 64, BITS, ALIGNED_N, 16, 64, 64, 1, 2>(
        w, scales, biases, x, out, Ws, int(dims[0]), int(dims[1]), int(dims[2]),
        threadgroup_position_in_grid, thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_k = {}


def qmm_nax_exact_v4(x, w, scales, biases, group_size=64, bits=4):
    """x: [K] or [M, K] (bf16/fp16); w: [N, K * bits / 32] uint32 (4- or 8-bit, MLX layout); scales, biases: [N, K/64].
    Output identical (expected bit for bit) to stock qmm_t_nax with 64-row tiles."""
    assert bits in (4, 8) and group_size == 64
    N, kp = w.shape
    K = kp * (32 // bits)
    assert K % 64 == 0
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "v4" not in _k:
        _k["v4"] = mx.fast.metal_kernel(
            name="qmm_nax_exact_v4", input_names=["x", "w", "scales", "biases", "dims"], output_names=["out"],
            source=_SRC, header=_header(), compile_options={"math_mode": "safe"})
    # K, N and the row count are runtime arguments (as in MLX's own kernels), so new layer shapes and chunk / batch
    # sizes do not compile new kernel variants
    y = _k["v4"](inputs=[x2, w, scales, biases, mx.array([K, N, M], dtype=mx.int32)],
                 template=[("T", x.dtype), ("ALIGNED_N", N % 64 == 0), ("BITS", bits)],
                 grid=(((N + 63) // 64) * 64, (M + 15) // 16, 1), threadgroup=(64, 1, 1),
                 output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y
