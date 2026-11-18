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


# ---------------------------------------------------------------------------------------------------------------------
# Dense v4: MLX's own dense accelerator GEMM loop (steel `gemm_loop`, vendored verbatim from v0.32.4 gemm_nax.h) with a
# 16-row tile. Stock MLX's regular dense GEMM on this GPU class uses BM = BN = 128, BK = 512, SK = 32, WM = WN = 4; only
# BM / WM change here (16 / 1), which does not change any output element's K order. K, N and the row count are
# runtime arguments.
def _dense_header():
    parts = []
    for f in ("defines.h", "type_traits.h", "integral_constant.h", "nax.h", "gemm_nax.h"):
        txt = (TP / f).read_text()
        parts.append("\n".join(l for l in txt.splitlines() if not l.startswith('#include "mlx/')
                               and not l.startswith("#pragma once")))
    # MLX appends the generated `template <...>` line directly after the header: end with a newline, or the last line's
    # comment (`} // namespace mlx::steel`) would swallow it
    return "\n\n".join(parts) + "\n"


_SRC_DENSE = r"""
    using namespace mlx::steel;
    constexpr int BM = BMT, BN = 128, BK = 512, WN = 4;
    constexpr int WM = (BMT == 16) ? 1 : 4;                // BMT = 128 is stock's own configuration (SM = 32)
    constexpr short SM = BM / WM, SN = 32, SK = 32;
    constexpr short TM = SM / 16, TN = SN / 16;
    const int M = int(dims[0]), N = int(dims[1]), K = int(dims[2]);
    const uint sg = simdgroup_index_in_threadgroup;
    const int c_row = int(threadgroup_position_in_grid.y) * BM;
    const int c_col = int(threadgroup_position_in_grid.x) * BN;
    const short tm = SM * short(sg / WN);
    const short tn = SN * short(sg % WN);
    const device T* A = x + size_t(c_row + tm) * K;
    const device T* B = w + size_t(c_col + tn) * K;
    device T* D = out + size_t(c_row + tm) * N + c_col + tn;
    const short sgp_sm = short(min(int(SM), M - (c_row + tm)));
    const short sgp_sn = short(min(int(SN), N - (c_col + tn)));
    const bool is_unaligned_sm = sgp_sm != SM;
    const bool is_unaligned_sn = sgp_sn != SN;
    NAXTile<float, TM, TN> Dtile;
    dispatch_bool(K % BK == 0, [&](auto kAlignedK) {
      dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(!is_unaligned_sn, [&](auto kAlignedN) {
          Dtile = gemm_loop<T, SM, SN, SK, BK, false, true, kAlignedM.value, kAlignedN.value, kAlignedK.value, float>(
              A, B, K, K, K, K / BK, sgp_sm, sgp_sn);
          if ((kAlignedM.value || sgp_sm > 0) && (kAlignedN.value || sgp_sn > 0)) {
            if constexpr (kAlignedM.value && kAlignedN.value) {
              Dtile.store(D, N);
            } else {
              Dtile.store_safe(D, N, short2(sgp_sn, sgp_sm));
            }
          }
        });
      });
    });
"""


def dense_nax_exact_v4(x, w, tile_rows=None):
    """x: [..., K]; w: [N, K] (same dtype, bf16/fp16). x @ w.T with MLX's canonical dense accelerator GEMM arithmetic,
    for any number of rows. tile_rows: 16 (default up to 32 rows) or 128 (stock's own tile; default above)."""
    N, K = w.shape
    assert x.dtype == w.dtype and x.dtype in (mx.bfloat16, mx.float16)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "dense_v4" not in _k:
        _k["dense_v4"] = mx.fast.metal_kernel(
            name="mlx_exact_dense_v4", input_names=["x", "w", "dims"], output_names=["out"],
            source=_SRC_DENSE, header=_dense_header(), compile_options={"math_mode": "safe"})
    bm = tile_rows or (16 if M <= 32 else 128)
    assert bm in (16, 128)
    tg = 128 if bm == 16 else 512
    y = _k["dense_v4"](inputs=[x2, w, mx.array([M, N, K], dtype=mx.int32)], template=[("T", x.dtype), ("BMT", bm)],
                       grid=(((N + 127) // 128) * tg, (M + bm - 1) // bm, 1), threadgroup=(tg, 1, 1),
                       output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y.reshape(*lead, N)
