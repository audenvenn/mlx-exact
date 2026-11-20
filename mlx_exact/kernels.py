"""Exact 4-bit matmul kernel for MLX on Apple M5/M6: reproduces, bit for bit, the arithmetic of MLX's neural-accelerator
prefill kernel (qmm_t_nax, 64-row tiles) for any number of rows, including a single decode row.

Arithmetic (reverse-engineered with crafted inputs, validated bit for bit against the hardware on random data):
  w_k  = T(float(s) * q_k + float(b))          dequantize in float, round once to the activation type T
  p_k  = float(w_k) * float(x_k)               exact in fp32 for bf16/fp16 operands
  K in sequential chunks of 16; quads r = 0..3 at offsets {2r, 2r+1, 2r+8, 2r+9};
  quad = (p[2r] + p[2r+1]) + (p[2r+8] + p[2r+9]);  acc = acc + quad   (fp32, in this order)
  y    = T(acc)
One thread per output (no cross-thread reduction). Compiled with math_mode "safe" (IEEE; no reassociation).
"""
import mlx.core as mx

_SRC_QMV_NAX_EXACT = r"""
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;          // activation row (decode: 0 only)
    if (n >= N_OUT) {
        return;
    }
    auto xr = x + size_t(m) * K_IN;
    auto wr = w + size_t(n) * (K_IN / 8);
    auto sr = scales + size_t(n) * (K_IN / GS);
    auto br = biases + size_t(n) * (K_IN / GS);
    float acc = 0.0f;
    for (int c0 = 0; c0 < K_IN; c0 += 16) {
        const float s = float(sr[c0 / GS]);
        const float b = float(br[c0 / GS]);
        const uint32_t w0 = wr[c0 / 8];
        const uint32_t w1 = wr[c0 / 8 + 1];
        float p[16];
        for (int i = 0; i < 8; i++) {
            const T wa = static_cast<T>(s * float((w0 >> (4 * i)) & 0xFu) + b);
            const T wb = static_cast<T>(s * float((w1 >> (4 * i)) & 0xFu) + b);
            p[i] = float(wa) * float(xr[c0 + i]);
            p[i + 8] = float(wb) * float(xr[c0 + 8 + i]);
        }
        for (int r = 0; r < 4; r++) {
            const float quad = (p[2 * r] + p[2 * r + 1]) + (p[2 * r + 8] + p[2 * r + 9]);
            acc = acc + quad;
        }
    }
    out[size_t(m) * N_OUT + n] = static_cast<T>(acc);
"""

_kernels = {}


def qmv_nax_exact(x, w, scales, biases, group_size=64, bits=4):
    """x: [K] or [M, K] (bf16 or fp16); w: [N, K/8] uint32 packed 4-bit (MLX layout); scales, biases: [N, K/gs].
    Returns [N] or [M, N] with the arithmetic of qmm_t_nax (64-row tiles), bit for bit."""
    assert bits == 4 and group_size % 16 == 0
    N, kp = w.shape
    K = kp * 8
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    assert K % 16 == 0
    if "qmv_nax_exact" not in _kernels:
        _kernels["qmv_nax_exact"] = mx.fast.metal_kernel(
            name="qmv_nax_exact", input_names=["x", "w", "scales", "biases"], output_names=["out"],
            source=_SRC_QMV_NAX_EXACT, compile_options={"math_mode": "safe"})
    k = _kernels["qmv_nax_exact"]
    y = k(inputs=[x2, w, scales, biases],
          template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("GS", group_size)],
          grid=(N, M, 1), threadgroup=(min(256, N), 1, 1),
          output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y


# ---------------------------------------------------------------------------------------------------------------------
# v6: same arithmetic (bit for bit). Each SIMD group owns 32 output rows and a private threadgroup tile (32 x 33 words):
# per 256-weight K slice, lane l loads word l of each of the 32 rows (coalesced 128-byte rows) one slice AHEAD into
# registers, stores the current slice to its tile, then (after a SIMD-level barrier only) runs its own row's sequential
# chain exactly as v0. 4 SIMD groups per threadgroup, no threadgroup-wide barriers. Requires K % 256 == 0, group 64.
_SRC_QMV_NAX_EXACT_V6 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = (threadgroup_position_in_grid.x * 4 + sg) * 32;
    const uint m = threadgroup_position_in_grid.y;
    threadgroup uint32_t tiles[4 * 32 * 33];
    threadgroup uint32_t* wt = tiles + sg * (32 * 33);
    const uint n = n0 + lane;
    const uint nn = (n < N_OUT) ? n : (N_OUT - 1);
    auto xr = x + size_t(m) * K_IN;
    auto sr = scales + size_t(nn) * (K_IN / 64);
    auto br = biases + size_t(nn) * (K_IN / 64);
    uint32_t pre[32];
    for (int r = 0; r < 32; r++) {
        const uint nr = min(n0 + r, uint(N_OUT - 1));
        pre[r] = w[size_t(nr) * (K_IN / 8) + lane];
    }
    float acc = 0.0f;
    for (int kb = 0; kb < K_IN / 256; kb++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < 32; r++) {
            wt[r * 33 + lane] = pre[r];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 1 < K_IN / 256) {
            for (int r = 0; r < 32; r++) {
                const uint nr = min(n0 + r, uint(N_OUT - 1));
                pre[r] = w[size_t(nr) * (K_IN / 8) + (kb + 1) * 32 + lane];
            }
        }
        for (int c = 0; c < 16; c++) {
            const int k0 = kb * 256 + c * 16;
            const float s = float(sr[k0 / 64]);
            const float b = float(br[k0 / 64]);
            const uint32_t w0 = wt[lane * 33 + 2 * c];
            const uint32_t w1 = wt[lane * 33 + 2 * c + 1];
            float p[16];
            for (int i = 0; i < 8; i++) {
                const T wa = static_cast<T>(s * float((w0 >> (4 * i)) & 0xFu) + b);
                const T wb = static_cast<T>(s * float((w1 >> (4 * i)) & 0xFu) + b);
                p[i] = float(wa) * float(xr[k0 + i]);
                p[i + 8] = float(wb) * float(xr[k0 + 8 + i]);
            }
            for (int r = 0; r < 4; r++) {
                acc = acc + ((p[2 * r] + p[2 * r + 1]) + (p[2 * r + 8] + p[2 * r + 9]));
            }
        }
    }
    if (n < N_OUT) {
        out[size_t(m) * N_OUT + n] = static_cast<T>(acc);
    }
"""


def qmv_nax_exact_v6(x, w, scales, biases, group_size=64, bits=4):
    """Same contract and arithmetic as qmv_nax_exact (v0); needs K % 256 == 0 and group_size == 64."""
    assert bits == 4 and group_size == 64
    N, kp = w.shape
    K = kp * 8
    assert K % 256 == 0
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "v6" not in _kernels:
        _kernels["v6"] = mx.fast.metal_kernel(
            name="qmv_nax_exact_v6", input_names=["x", "w", "scales", "biases"], output_names=["out"],
            source=_SRC_QMV_NAX_EXACT_V6, compile_options={"math_mode": "safe"})
    y = _kernels["v6"](inputs=[x2, w, scales, biases],
                       template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K)],
                       grid=(((N + 127) // 128) * 128, M, 1), threadgroup=(128, 1, 1),
                       output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y


# ---------------------------------------------------------------------------------------------------------------------
# Expert-indexed (MoE) version of v0: row b of x uses expert idx[b]; same arithmetic, so each row equals what MLX's
# canonical sorted gather kernel (gather_qmm_rhs on NAX devices) computes for it (FINDINGS F16).
_SRC_GATHER_EXACT = r"""
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;          // row (token-expert pair)
    if (n >= N_OUT) {
        return;
    }
    const uint e = idx[m];
    auto xr = x + size_t(m) * K_IN;
    auto wr = w + (size_t(e) * N_OUT + n) * (K_IN / 8);
    auto sr = scales + (size_t(e) * N_OUT + n) * (K_IN / GS);
    auto br = biases + (size_t(e) * N_OUT + n) * (K_IN / GS);
    float acc = 0.0f;
    for (int c0 = 0; c0 < K_IN; c0 += 16) {
        const float s = float(sr[c0 / GS]);
        const float b = float(br[c0 / GS]);
        const uint32_t w0 = wr[c0 / 8];
        const uint32_t w1 = wr[c0 / 8 + 1];
        float p[16];
        for (int i = 0; i < 8; i++) {
            const T wa = static_cast<T>(s * float((w0 >> (4 * i)) & 0xFu) + b);
            const T wb = static_cast<T>(s * float((w1 >> (4 * i)) & 0xFu) + b);
            p[i] = float(wa) * float(xr[c0 + i]);
            p[i + 8] = float(wb) * float(xr[c0 + 8 + i]);
        }
        for (int r = 0; r < 4; r++) {
            const float quad = (p[2 * r] + p[2 * r + 1]) + (p[2 * r + 8] + p[2 * r + 9]);
            acc = acc + quad;
        }
    }
    out[size_t(m) * N_OUT + n] = static_cast<T>(acc);
"""


def gather_qmv_nax_exact(x, w, scales, biases, idx, group_size=64, bits=4):
    """x: [Bp, K] rows; w: [E, N, K/8] uint32; scales, biases: [E, N, K/gs]; idx: [Bp] expert per row -> [Bp, N]."""
    assert bits == 4 and group_size % 16 == 0
    E, N, kp = w.shape
    K = kp * 8
    x2 = x.reshape(-1, K)
    Bp = x2.shape[0]
    if "gather" not in _kernels:
        _kernels["gather"] = mx.fast.metal_kernel(
            name="gather_qmv_nax_exact", input_names=["x", "w", "scales", "biases", "idx"], output_names=["out"],
            source=_SRC_GATHER_EXACT, compile_options={"math_mode": "safe"})
    return _kernels["gather"](inputs=[x2, w, scales, biases, idx.reshape(-1).astype(mx.uint32)],
                              template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("GS", group_size)],
                              grid=(N, Bp, 1), threadgroup=(min(256, N), 1, 1),
                              output_shapes=[(Bp, N)], output_dtypes=[x.dtype])[0]


# ---------------------------------------------------------------------------------------------------------------------
# Bit-width-generic v0 (BITS in {4, 8}): per 16-weight chunk, unpack 16*BITS/32 words (low bits first, MLX layout).
_SRC_QMV_EXACT_BITS = r"""
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;
    if (n >= N_OUT) {
        return;
    }
    constexpr int PER_WORD = 32 / BITS;                 // values per uint32
    constexpr uint MASK = (1u << BITS) - 1u;
    auto xr = x + size_t(m) * K_IN;
    auto wr = w + size_t(n) * (K_IN / PER_WORD);
    auto sr = scales + size_t(n) * (K_IN / GS);
    auto br = biases + size_t(n) * (K_IN / GS);
    float acc = 0.0f;
    for (int c0 = 0; c0 < K_IN; c0 += 16) {
        const float s = float(sr[c0 / GS]);
        const float b = float(br[c0 / GS]);
        float p[16];
        for (int i = 0; i < 16; i++) {
            const uint32_t word = wr[(c0 + i) / PER_WORD];
            const uint q = (word >> (BITS * ((c0 + i) % PER_WORD))) & MASK;
            const T wv = static_cast<T>(s * float(q) + b);
            p[i] = float(wv) * float(xr[c0 + i]);
        }
        for (int r = 0; r < 4; r++) {
            acc = acc + ((p[2 * r] + p[2 * r + 1]) + (p[2 * r + 8] + p[2 * r + 9]));
        }
    }
    out[size_t(m) * N_OUT + n] = static_cast<T>(acc);
"""


def qmv_exact_bits(x, w, scales, biases, group_size=64, bits=8):
    """Exact kernel for bits in {4, 8} (affine): same contract as qmv_nax_exact."""
    assert bits in (4, 8) and group_size % 16 == 0
    N = w.shape[0]
    K = w.shape[1] * (32 // bits)
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "bits" not in _kernels:
        _kernels["bits"] = mx.fast.metal_kernel(
            name="qmv_exact_bits", input_names=["x", "w", "scales", "biases"], output_names=["out"],
            source=_SRC_QMV_EXACT_BITS, compile_options={"math_mode": "safe"})
    y = _kernels["bits"](inputs=[x2, w, scales, biases],
                         template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("GS", group_size), ("BITS", bits)],
                         grid=(N, M, 1), threadgroup=(min(256, N), 1, 1),
                         output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y


# ---------------------------------------------------------------------------------------------------------------------
# v6 generalized to BITS in {4, 8} (v6b): same tile/prefetch structure; a slice is 32 packed words per row
# (256 weights at 4 bits, 128 at 8 bits). Same arithmetic. Requires K % slice == 0 and group size 64.
_SRC_QMV_EXACT_V6B = r"""
    constexpr int PER_WORD = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int SLICE = 32 * PER_WORD;             // weights per 32-word row slice
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = (threadgroup_position_in_grid.x * 4 + sg) * 32;
    const uint m = threadgroup_position_in_grid.y;
    threadgroup uint32_t tiles[4 * 32 * 33];
    threadgroup uint32_t* wt = tiles + sg * (32 * 33);
    const uint n = n0 + lane;
    const uint nn = (n < N_OUT) ? n : (N_OUT - 1);
    auto xr = x + size_t(m) * K_IN;
    auto sr = scales + size_t(nn) * (K_IN / 64);
    auto br = biases + size_t(nn) * (K_IN / 64);
    uint32_t pre[32];
    for (int r = 0; r < 32; r++) {
        const uint nr = min(n0 + r, uint(N_OUT - 1));
        pre[r] = w[size_t(nr) * (K_IN / PER_WORD) + lane];
    }
    float acc = 0.0f;
    for (int kb = 0; kb < K_IN / SLICE; kb++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < 32; r++) {
            wt[r * 33 + lane] = pre[r];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 1 < K_IN / SLICE) {
            for (int r = 0; r < 32; r++) {
                const uint nr = min(n0 + r, uint(N_OUT - 1));
                pre[r] = w[size_t(nr) * (K_IN / PER_WORD) + (kb + 1) * 32 + lane];
            }
        }
        for (int c = 0; c < SLICE / 16; c++) {
            const int k0 = kb * SLICE + c * 16;
            const float s = float(sr[k0 / 64]);
            const float b = float(br[k0 / 64]);
            float p[16];
            for (int i = 0; i < 16; i++) {
                const uint32_t word = wt[lane * 33 + (c * 16 + i) / PER_WORD];
                const uint q = (word >> (BITS * ((c * 16 + i) % PER_WORD))) & MASK;
                const T wv = static_cast<T>(s * float(q) + b);
                p[i] = float(wv) * float(xr[k0 + i]);
            }
            for (int r = 0; r < 4; r++) {
                acc = acc + ((p[2 * r] + p[2 * r + 1]) + (p[2 * r + 8] + p[2 * r + 9]));
            }
        }
    }
    if (n < N_OUT) {
        out[size_t(m) * N_OUT + n] = static_cast<T>(acc);
    }
"""


def qmv_exact_v6b(x, w, scales, biases, group_size=64, bits=8):
    assert bits in (4, 8) and group_size == 64
    N = w.shape[0]
    per_word = 32 // bits
    K = w.shape[1] * per_word
    assert K % (32 * per_word) == 0
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "v6b" not in _kernels:
        _kernels["v6b"] = mx.fast.metal_kernel(
            name="qmv_exact_v6b", input_names=["x", "w", "scales", "biases"], output_names=["out"],
            source=_SRC_QMV_EXACT_V6B, compile_options={"math_mode": "safe"})
    y = _kernels["v6b"](inputs=[x2, w, scales, biases],
                        template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("BITS", bits)],
                        grid=(((N + 127) // 128) * 128, M, 1), threadgroup=(128, 1, 1),
                        output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y


# ---------------------------------------------------------------------------------------------------------------------
# Dense (unquantized) exact kernel: y = x @ W.T with W [N, K] in T (bf16/fp16); the F3 order with products of the stored
# T values (exact in fp32). Reproduces MLX's regular NAX GEMM (the canonical dense prefill path) bit for bit.
_SRC_DENSE_EXACT = r"""
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;
    if (n >= N_OUT) {
        return;
    }
    auto xr = x + size_t(m) * K_IN;
    auto wr = w + size_t(n) * K_IN;
    float acc = 0.0f;
    for (int c0 = 0; c0 < K_IN; c0 += 16) {
        float p[16];
        for (int i = 0; i < 16; i++) {
            p[i] = float(wr[c0 + i]) * float(xr[c0 + i]);
        }
        for (int r = 0; r < 4; r++) {
            acc = acc + ((p[2 * r] + p[2 * r + 1]) + (p[2 * r + 8] + p[2 * r + 9]));
        }
    }
    out[size_t(m) * N_OUT + n] = static_cast<T>(acc);
"""


def dense_exact(x, w):
    """x: [..., K]; w: [N, K] (same dtype as x, bf16/fp16); K % 16 == 0. Returns x @ w.T with the canonical arithmetic."""
    N, K = w.shape
    assert K % 16 == 0 and x.dtype == w.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "dense" not in _kernels:
        _kernels["dense"] = mx.fast.metal_kernel(
            name="dense_exact", input_names=["x", "w"], output_names=["out"],
            source=_SRC_DENSE_EXACT, compile_options={"math_mode": "safe"})
    y = _kernels["dense"](inputs=[x2, w], template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K)],
                          grid=(N, M, 1), threadgroup=(min(256, N), 1, 1),
                          output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y.reshape(*lead, N)


# ---------------------------------------------------------------------------------------------------------------------
# fp32 sequential matmul for LoRA adapters (no stock canonical exists for this added path, so we define one that is the
# same for any number of rows): out[m, n] = fma-chain over k of float(x[m, k]) * W[k, n], k ascending. W: [K, N] fp32.
_SRC_FP32_SEQ = r"""
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;
    if (n >= N_OUT) {
        return;
    }
    auto xr = x + size_t(m) * K_IN;
    float acc = 0.0f;
    for (int k = 0; k < K_IN; k++) {
        acc = fma(float(xr[k]), w[size_t(k) * N_OUT + n], acc);
    }
    out[size_t(m) * N_OUT + n] = acc;
"""


def fp32_seq_matmul(x, w):
    """x: [..., K] (any float dtype); w: [K, N] float32 -> [..., N] float32, row-count-invariant arithmetic."""
    K, N = w.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "fp32seq" not in _kernels:
        _kernels["fp32seq"] = mx.fast.metal_kernel(
            name="fp32_seq_matmul", input_names=["x", "w"], output_names=["out"], source=_SRC_FP32_SEQ,
            compile_options={"math_mode": "safe"})
    y = _kernels["fp32seq"](inputs=[x2, w.astype(mx.float32)], template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K)],
                            grid=(N, M, 1), threadgroup=(min(256, N), 1, 1),
                            output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]
    return y.reshape(*lead, N)


# ---------------------------------------------------------------------------------------------------------------------
# Quantized-KV-cache attention (mlx-lm's quantized_scaled_dot_product_attention) with a fixed arithmetic for any number
# of queries. Keys/values are affine-quantized along the head dim D (groups of GS within each key/value row).
#   scores[b, h, l, j] = sum_d q[b, h, l, d] * deq(K[b, kvh(h), j, d])     (fp32, d ascending, fma)
#   out[b, h, l, d]    = sum_j p[b, h, l, j] * deq(V[b, kvh(h), j, d])     (fp32, j ascending, fma)
# deq(v) = float(s) * q + float(b) (MLX's affine dequantization). Scores/outputs are cast to the query dtype at the end,
# as mx.quantized_matmul returns the input dtype.
_SRC_QKV_SCORES = r"""
    const uint j = thread_position_in_grid.x;            // key index
    const uint l = thread_position_in_grid.y;            // query index
    const uint bh = thread_position_in_grid.z;           // b * HQ + h
    const int L = int(dims[0]);                           // runtime: no new kernel variant per decode step
    const int KL = int(dims[1]);
    if (int(j) >= KL) { return; }
    const uint b = bh / HQ, h = bh % HQ, kvh = h / (HQ / HKV);
    constexpr int PER_WORD = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    auto qr = q + (size_t(bh) * L + l) * D;
    const size_t krow = (size_t(b) * HKV + kvh) * KL + j;
    auto kw = kq + krow * (D / PER_WORD);
    auto ks = kscales + krow * (D / GS);
    auto kb = kbiases + krow * (D / GS);
    float acc = 0.0f;
    for (int d = 0; d < D; d++) {
        const uint qv = (kw[d / PER_WORD] >> (BITS * (d % PER_WORD))) & MASK;
        const float kv = float(ks[d / GS]) * float(qv) + float(kb[d / GS]);
        acc = fma(float(qr[d]), kv, acc);
    }
    out[(size_t(bh) * L + l) * KL + j] = static_cast<T>(acc);
"""

_SRC_QKV_OUT = r"""
    const uint d = thread_position_in_grid.x;            // head-dim index
    const uint l = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    if (int(d) >= D) { return; }
    const uint b = bh / HQ, h = bh % HQ, kvh = h / (HQ / HKV);
    constexpr int PER_WORD = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    auto pr = p + (size_t(bh) * L + l) * KL;
    float acc = 0.0f;
    for (int j = 0; j < KL; j++) {
        const size_t vrow = (size_t(b) * HKV + kvh) * KL + j;
        const uint qv = (vq[vrow * (D / PER_WORD) + d / PER_WORD] >> (BITS * (d % PER_WORD))) & MASK;
        const float vv = float(vscales[vrow * (D / GS) + d / GS]) * float(qv) + float(vbiases[vrow * (D / GS) + d / GS]);
        acc = fma(float(pr[j]), vv, acc);
    }
    out[(size_t(bh) * L + l) * D + d] = static_cast<T>(acc);
"""


def qkv_attention_exact(queries, q_keys, q_values, scale, mask, group_size=64, bits=8):
    """Drop-in for mlx-lm's quantized_scaled_dot_product_attention with a query-count-invariant arithmetic.
    queries [B, HQ, L, D]; q_keys / q_values = (packed [B, HKV, kL, D*bits/32], scales, biases)."""
    B, HQ, L, D = queries.shape
    kq, ks, kb = q_keys
    vq, vs, vb = q_values
    HKV, KL = kq.shape[1], kq.shape[2]
    q = (queries * scale)                                   # same scaling as mlx-lm (in the query dtype)
    tmpl = [("T", queries.dtype), ("HQ", HQ), ("HKV", HKV), ("D", D), ("GS", group_size), ("BITS", bits)]
    dims = mx.array([L, KL], dtype=mx.int32)
    if "qkv_s" not in _kernels:
        _kernels["qkv_s"] = mx.fast.metal_kernel(
            name="qkv_scores_exact", input_names=["q", "kq", "kscales", "kbiases", "dims"], output_names=["out"],
            source=_SRC_QKV_SCORES, compile_options={"math_mode": "safe"})
        _kernels["qkv_o"] = mx.fast.metal_kernel(
            name="qkv_out_exact", input_names=["p", "vq", "vscales", "vbiases", "dims"], output_names=["out"],
            source=_SRC_QKV_OUT, compile_options={"math_mode": "safe"})
    scores = _kernels["qkv_s"](inputs=[q, kq, ks, kb, dims], template=tmpl, grid=(KL, L, B * HQ),
                               threadgroup=(min(256, KL), 1, 1), output_shapes=[(B, HQ, L, KL)],
                               output_dtypes=[queries.dtype])[0]
    if mask is not None:
        if isinstance(mask, str):
            qi = mx.arange(KL - L, KL)
            ki = mx.arange(KL)
            mask = qi[:, None] >= ki[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    p = mx.softmax(scores, axis=-1, precise=True)
    return _kernels["qkv_o"](inputs=[p, vq, vs, vb, dims], template=tmpl, grid=(D, L, B * HQ),
                             threadgroup=(min(256, D), 1, 1), output_shapes=[(B, HQ, L, D)],
                             output_dtypes=[queries.dtype])[0]


# ---------------------------------------------------------------------------------------------------------------------
# v7: exact decode kernel with enough parallelism for narrow layers (4- and 8-bit, group 64, K % 256 == 0).
# v0/v6 give each lane a whole output row, so a layer with 1024 outputs runs only 32 SIMD groups, and a decode step
# (a chain of dependent matmuls) waits on each row's long serial loop. v7 splits each row over lanes and keeps the
# canonical arithmetic: per threadgroup, 32 rows; per 256-weight K tile,
#   producers (8 SIMD groups, lane = 16-weight chunk) compute the four canonical quad sums of each chunk into one half
#   of a double buffer in threadgroup memory, while
#   a chain SIMD group (lane = row) adds the previous tile's quad sums into the row's fp32 accumulator in canonical
#   order; one barrier per tile.
# The pair sums use fma(a, b, c * d): products of bf16/fp16 values are exact in fp32, so this equals the canonical
# round(a * b + c * d) bit for bit.
# ---------------------------------------------------------------------------------------------------------------------
_SRC_QMV_EXACT_V7 = r"""    // DQ: round the dequantized fp32 value to T, as MLX's loader does (static_cast<T>), and widen back to float
    #define DQ(v) float(static_cast<T>(v))

    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint m = threadgroup_position_in_grid.y;
    const uint hf = lane / 16;
    const uint chunk = lane % 16;
    constexpr int NT = K_IN / 256;
    constexpr int PER_WORD = 32 / BITS;
    constexpr int WPC = 16 / PER_WORD;
    constexpr int BUF = 4 * 16 * 33;
    threadgroup float qs[2 * BUF];
    auto xr = x + size_t(m) * K_IN;
    float acc = 0.0f;
    for (int kb = 0; kb <= NT; kb++) {
        if (sg < P) {
            if (kb < NT) {
                threadgroup float* q = qs + (kb & 1) * BUF;
                const int k0 = kb * 256 + int(chunk) * 16;
                float xs[16];
                for (int i = 0; i < 16; i++) {
                    xs[i] = float(xr[k0 + i]);
                }
                for (int p = 0; p < 16 / P; p++) {
                    const uint rl = (sg * (16 / P) + p) * 2 + hf;
                    const uint n = min(n0 + rl, uint(N_OUT - 1));
                    const size_t wi = size_t(n) * (K_IN / PER_WORD) + k0 / PER_WORD;
                    uint32_t ww[WPC];
                    for (int j = 0; j < WPC; j++) {
                        ww[j] = w[wi + j];
                    }
                    const float s = float(scales[size_t(n) * (K_IN / 64) + k0 / 64]);
                    const float b = float(biases[size_t(n) * (K_IN / 64) + k0 / 64]);
                    float wv[16];
                    for (int i = 0; i < 16; i++) {
                        wv[i] = DQ(fma(s, float(extract_bits(ww[i / PER_WORD], uint(BITS * (i % PER_WORD)), uint(BITS))), b));
                    }
                    // products are exact in fp32, so fma(a, b, c * d) == round(a * b + c * d): same bits as the
                    // canonical pair sum p[2r] + p[2r+1]
                    for (int r = 0; r < 4; r++) {
                        const float pa = fma(wv[2 * r], xs[2 * r], wv[2 * r + 1] * xs[2 * r + 1]);
                        const float pb = fma(wv[2 * r + 8], xs[2 * r + 8], wv[2 * r + 9] * xs[2 * r + 9]);
                        q[(r * 16 + chunk) * 33 + rl] = pa + pb;
                    }
                }
            }
        } else if (kb > 0) {
            threadgroup const float* q = qs + ((kb - 1) & 1) * BUF;
            for (int c = 0; c < 16; c++) {
                for (int r = 0; r < 4; r++) {
                    acc = acc + q[(r * 16 + c) * 33 + lane];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == P && n0 + lane < N_OUT) {
        out[size_t(m) * N_OUT + n0 + lane] = static_cast<T>(acc);
    }
"""


def qmv_exact_v7(x, w, scales, biases, group_size=64, bits=4, producers=8):
    """x[M, K] @ dequant(w).T with the canonical arithmetic (same contract as qmv_nax_exact); 4- or 8-bit affine,
    group 64, K % 256 == 0. Fastest for decode (M = 1); one threadgroup row per x row."""
    assert bits in (4, 8) and group_size == 64 and producers in (2, 4, 8)
    N, kp = w.shape
    K = kp * (32 // bits)
    assert K % 256 == 0
    single = x.ndim == 1
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "v7" not in _kernels:
        _kernels["v7"] = mx.fast.metal_kernel(
            name="mlx_exact_qmv_v7", input_names=["x", "w", "scales", "biases"], output_names=["out"],
            source=_SRC_QMV_EXACT_V7, compile_options={"math_mode": "safe"})
    tg = (producers + 1) * 32
    y = _kernels["v7"](inputs=[x2, w, scales, biases],
                       template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("P", producers), ("BITS", bits)],
                       grid=(((N + 31) // 32) * tg, M, 1), threadgroup=(tg, 1, 1),
                       output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y[0] if single else y


# dense v7: the v7 design for unquantized bf16/fp16 weights (no dequantization); same canonical order as dense_exact.
_SRC_DENSE_EXACT_V7 = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint m = threadgroup_position_in_grid.y;
    const uint hf = lane / 16;
    const uint chunk = lane % 16;
    constexpr int NT = K_IN / 256;
    constexpr int BUF = 4 * 16 * 33;
    threadgroup float qs[2 * BUF];
    auto xr = x + size_t(m) * K_IN;
    float acc = 0.0f;
    for (int kb = 0; kb <= NT; kb++) {
        if (sg < P) {
            if (kb < NT) {
                threadgroup float* q = qs + (kb & 1) * BUF;
                const int k0 = kb * 256 + int(chunk) * 16;
                float xs[16];
                for (int i = 0; i < 16; i++) {
                    xs[i] = float(xr[k0 + i]);
                }
                for (int p = 0; p < 16 / P; p++) {
                    const uint rl = (sg * (16 / P) + p) * 2 + hf;
                    const uint n = min(n0 + rl, uint(N_OUT - 1));
                    const size_t wi = size_t(n) * K_IN + k0;
                    float wv[16];
                    for (int i = 0; i < 16; i++) {
                        wv[i] = float(w[wi + i]);
                    }
                    // products are exact in fp32, so fma(a, b, c * d) == round(a * b + c * d): same bits as the
                    // canonical pair sum p[2r] + p[2r+1]
                    for (int r = 0; r < 4; r++) {
                        const float pa = fma(wv[2 * r], xs[2 * r], wv[2 * r + 1] * xs[2 * r + 1]);
                        const float pb = fma(wv[2 * r + 8], xs[2 * r + 8], wv[2 * r + 9] * xs[2 * r + 9]);
                        q[(r * 16 + chunk) * 33 + rl] = pa + pb;
                    }
                }
            }
        } else if (kb > 0) {
            threadgroup const float* q = qs + ((kb - 1) & 1) * BUF;
            for (int c = 0; c < 16; c++) {
                for (int r = 0; r < 4; r++) {
                    acc = acc + q[(r * 16 + c) * 33 + lane];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == P && n0 + lane < N_OUT) {
        out[size_t(m) * N_OUT + n0 + lane] = static_cast<T>(acc);
    }
"""


def dense_exact_v7(x, w, producers=8):
    """x: [..., K]; w: [N, K] (same dtype as x, bf16/fp16); K % 256 == 0. x @ w.T with the canonical arithmetic;
    fastest for decode-sized calls (M <= 4)."""
    N, K = w.shape
    assert K % 256 == 0 and x.dtype == w.dtype and producers in (2, 4, 8)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if "dense_v7" not in _kernels:
        _kernels["dense_v7"] = mx.fast.metal_kernel(
            name="mlx_exact_dense_v7", input_names=["x", "w"], output_names=["out"],
            source=_SRC_DENSE_EXACT_V7, compile_options={"math_mode": "safe"})
    tg = (producers + 1) * 32
    y = _kernels["dense_v7"](inputs=[x2, w], template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("P", producers)],
                             grid=(((N + 31) // 32) * tg, M, 1), threadgroup=(tg, 1, 1),
                             output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y.reshape(*lead, N)


# v7 for 2-4 x rows (speculative verification, small batches): 8 output rows per threadgroup, so the double buffer of
# quad sums (rows x 4.6 KB) stays small; producers dequantize each weight once and reuse it for every x row; the chain
# SIMD group gives each lane one (output row = lane % 8, x row = lane / 8) chain. Same arithmetic as v0/v7.
_SRC_QMV_EXACT_V7_ROWS = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.x * 8;
    const uint hf = lane / 16;
    const uint chunk = lane % 16;
    constexpr int NT = K_IN / 256;
    constexpr int PER_WORD = 32 / BITS;
    constexpr int WPC = 16 / PER_WORD;
    constexpr int BUF = MR * 4 * 16 * 9;
    threadgroup float qs[2 * BUF];
    float acc = 0.0f;
    const uint crow = lane % 8;
    const uint cm = lane / 8;
    for (int kb = 0; kb <= NT; kb++) {
        if (sg < 2) {
            if (kb < NT) {
                threadgroup float* q = qs + (kb & 1) * BUF;
                const int k0 = kb * 256 + int(chunk) * 16;
                for (int p = 0; p < 2; p++) {
                    const uint rl = (sg * 2 + p) * 2 + hf;
                    const uint n = min(n0 + rl, uint(N_OUT - 1));
                    const size_t wi = size_t(n) * (K_IN / PER_WORD) + k0 / PER_WORD;
                    uint32_t ww[WPC];
                    for (int j = 0; j < WPC; j++) {
                        ww[j] = w[wi + j];
                    }
                    const float s = float(scales[size_t(n) * (K_IN / 64) + k0 / 64]);
                    const float b = float(biases[size_t(n) * (K_IN / 64) + k0 / 64]);
                    float wv[16];
                    for (int i = 0; i < 16; i++) {
                        wv[i] = float(static_cast<T>(fma(s, float(extract_bits(ww[i / PER_WORD], uint(BITS * (i % PER_WORD)), uint(BITS))), b)));
                    }
                    for (int m = 0; m < MR; m++) {
                        float xs[16];
                        for (int i = 0; i < 16; i++) {
                            xs[i] = float(x[size_t(m) * K_IN + k0 + i]);
                        }
                        for (int r = 0; r < 4; r++) {
                            const float pa = fma(wv[2 * r], xs[2 * r], wv[2 * r + 1] * xs[2 * r + 1]);
                            const float pb = fma(wv[2 * r + 8], xs[2 * r + 8], wv[2 * r + 9] * xs[2 * r + 9]);
                            q[((m * 4 + r) * 16 + chunk) * 9 + rl] = pa + pb;
                        }
                    }
                }
            }
        } else if (kb > 0 && int(cm) < MR) {
            threadgroup const float* q = qs + ((kb - 1) & 1) * BUF;
            for (int c = 0; c < 16; c++) {
                for (int r = 0; r < 4; r++) {
                    acc = acc + q[((int(cm) * 4 + r) * 16 + c) * 9 + crow];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 2 && int(cm) < MR && n0 + crow < N_OUT) {
        out[size_t(cm) * N_OUT + n0 + crow] = static_cast<T>(acc);
    }
"""


def qmv_exact_v7_rows(x, w, scales, biases, group_size=64, bits=4):
    """x[M, K] @ dequant(w).T for 1 <= M <= 4 rows; 4- or 8-bit affine, group 64, K % 256 == 0; canonical arithmetic."""
    assert bits in (4, 8) and group_size == 64
    N, kp = w.shape
    K = kp * (32 // bits)
    assert K % 256 == 0
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    assert 1 <= M <= 4
    if "v7_rows" not in _kernels:
        _kernels["v7_rows"] = mx.fast.metal_kernel(
            name="mlx_exact_qmv_v7_rows", input_names=["x", "w", "scales", "biases"], output_names=["out"],
            source=_SRC_QMV_EXACT_V7_ROWS, compile_options={"math_mode": "safe"})
    y = _kernels["v7_rows"](inputs=[x2, w, scales, biases],
                            template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("MR", M), ("BITS", bits)],
                            grid=(((N + 7) // 8) * 96, 1, 1), threadgroup=(96, 1, 1),
                            output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y.reshape(*lead, N)


# gather v7: the v7 design for MoE expert layers; x row m uses expert idx[m]. Same arithmetic as MLX's canonical sorted
# expert kernel (gather_qmm_rhs on accelerator devices), 4- and 8-bit.
_SRC_GATHER_EXACT_V7 = r"""    // DQ: round the dequantized fp32 value to T, as MLX's loader does (static_cast<T>), and widen back to float
    #define DQ(v) float(static_cast<T>(v))

    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint m = threadgroup_position_in_grid.y;
    const size_t eoff = size_t(idx[m]) * N_OUT;      // this row's expert
    const uint hf = lane / 16;
    const uint chunk = lane % 16;
    constexpr int NT = K_IN / 256;
    constexpr int PER_WORD = 32 / BITS;
    constexpr int WPC = 16 / PER_WORD;
    constexpr int BUF = 4 * 16 * 33;
    threadgroup float qs[2 * BUF];
    auto xr = x + size_t(m) * K_IN;
    float acc = 0.0f;
    for (int kb = 0; kb <= NT; kb++) {
        if (sg < P) {
            if (kb < NT) {
                threadgroup float* q = qs + (kb & 1) * BUF;
                const int k0 = kb * 256 + int(chunk) * 16;
                float xs[16];
                for (int i = 0; i < 16; i++) {
                    xs[i] = float(xr[k0 + i]);
                }
                for (int p = 0; p < 16 / P; p++) {
                    const uint rl = (sg * (16 / P) + p) * 2 + hf;
                    const uint n = min(n0 + rl, uint(N_OUT - 1));
                    const size_t wi = (eoff + n) * (K_IN / PER_WORD) + k0 / PER_WORD;
                    uint32_t ww[WPC];
                    for (int j = 0; j < WPC; j++) {
                        ww[j] = w[wi + j];
                    }
                    const float s = float(scales[(eoff + n) * (K_IN / 64) + k0 / 64]);
                    const float b = float(biases[(eoff + n) * (K_IN / 64) + k0 / 64]);
                    float wv[16];
                    for (int i = 0; i < 16; i++) {
                        wv[i] = DQ(fma(s, float(extract_bits(ww[i / PER_WORD], uint(BITS * (i % PER_WORD)), uint(BITS))), b));
                    }
                    // products are exact in fp32, so fma(a, b, c * d) == round(a * b + c * d): same bits as the
                    // canonical pair sum p[2r] + p[2r+1]
                    for (int r = 0; r < 4; r++) {
                        const float pa = fma(wv[2 * r], xs[2 * r], wv[2 * r + 1] * xs[2 * r + 1]);
                        const float pb = fma(wv[2 * r + 8], xs[2 * r + 8], wv[2 * r + 9] * xs[2 * r + 9]);
                        q[(r * 16 + chunk) * 33 + rl] = pa + pb;
                    }
                }
            }
        } else if (kb > 0) {
            threadgroup const float* q = qs + ((kb - 1) & 1) * BUF;
            for (int c = 0; c < 16; c++) {
                for (int r = 0; r < 4; r++) {
                    acc = acc + q[(r * 16 + c) * 33 + lane];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == P && n0 + lane < N_OUT) {
        out[size_t(m) * N_OUT + n0 + lane] = static_cast<T>(acc);
    }
"""


def gather_qmv_exact_v7(x, w, scales, biases, idx, group_size=64, bits=4, producers=8):
    """x: [B, K] rows; w: [E, N, K*bits/32] uint32; scales, biases: [E, N, K/64]; idx: [B] expert per row -> [B, N].
    4- or 8-bit affine, group 64, K % 256 == 0; canonical arithmetic."""
    assert bits in (4, 8) and group_size == 64 and producers in (2, 4, 8)
    E, N, kp = w.shape
    K = kp * (32 // bits)
    assert K % 256 == 0 and x.shape[-1] == K
    B = x.shape[0]
    if "gather_v7" not in _kernels:
        _kernels["gather_v7"] = mx.fast.metal_kernel(
            name="mlx_exact_gather_v7", input_names=["x", "w", "scales", "biases", "idx"], output_names=["out"],
            source=_SRC_GATHER_EXACT_V7, compile_options={"math_mode": "safe"})
    tg = (producers + 1) * 32
    return _kernels["gather_v7"](inputs=[x, w, scales, biases, idx.reshape(-1).astype(mx.uint32)],
                                 template=[("T", x.dtype), ("N_OUT", N), ("K_IN", K), ("P", producers), ("BITS", bits)],
                                 grid=(((N + 31) // 32) * tg, B, 1), threadgroup=(tg, 1, 1),
                                 output_shapes=[(B, N)], output_dtypes=[x.dtype])[0]


# ---------------------------------------------------------------------------------------------------------------------
# Fixed-order attention for dense K/V (for head dims where MLX's own attention is shape-dependent): scores (fp32
# sequential fma over d, times scale), masked entries set to -inf, softmax with one thread per row (sequential max and
# sum over keys in index order), output (fp32 sequential fma over keys). Masked keys contribute exact zeros, so the
# result depends only on the order of the valid keys, not on their indices, the key count or the query count.
# Query / key counts are runtime arguments (no recompilation per decode step).
# ---------------------------------------------------------------------------------------------------------------------
# Fixed-order attention (head dims 192/256 with one KV head). Every output element is ONE sequential fp32 chain:
# scores = (sum over d of fma(q, k)) * scale; softmax: max, exp, sequential sum over keys, scale; output = sum over
# keys of fma(p, v). The kernels below compute exactly these chains (0.8.1: register-blocked 4 x 4 outputs per
# thread with tiles staged through threadgroup memory, and one SIMD group per softmax row: max and exp in
# parallel, the sum still sequential); bit-identical to the one-thread-per-output kernels of 0.7.0-0.8.0.
_SRC_DATT_SCORES = r"""
    constexpr int R = 4, TJ = 32, TL = 32, DC = 16;
    threadgroup T qs[TL][DC + 1];
    threadgroup T ks[TJ][DC + 1];
    const uint tx = thread_position_in_threadgroup.x;          // key group (8)
    const uint ty = thread_position_in_threadgroup.y;          // row group (8)
    const uint j0 = threadgroup_position_in_grid.x * TJ;
    const uint l0 = threadgroup_position_in_grid.y * TL;
    const uint bh = threadgroup_position_in_grid.z;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    const uint b = bh / HQ, h = bh % HQ, kvh = h / (HQ / HKV);
    const uint tid = ty * 8 + tx;
    auto qb = q + (size_t(bh) * L) * D;
    auto kb = k + ((size_t(b) * HKV + kvh) * KL) * D;
    float acc[R][R];
    for (int a = 0; a < R; a++) for (int c = 0; c < R; c++) acc[a][c] = 0.0f;
    for (int d0 = 0; d0 < D; d0 += DC) {
        for (uint i = tid; i < TL * DC; i += 64) {
            const int r = int(l0) + int(i / DC), c = d0 + int(i % DC);
            qs[i / DC][i % DC] = (r < L) ? qb[size_t(r) * D + c] : T(0);
        }
        for (uint i = tid; i < TJ * DC; i += 64) {
            const int r = int(j0) + int(i / DC), c = d0 + int(i % DC);
            ks[i / DC][i % DC] = (r < KL) ? kb[size_t(r) * D + c] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int dd = 0; dd < DC; dd++) {
            float qv[R], kv[R];
            for (int a = 0; a < R; a++) qv[a] = float(qs[ty * R + a][dd]);
            for (int c = 0; c < R; c++) kv[c] = float(ks[tx * R + c][dd]);
            for (int a = 0; a < R; a++) for (int c = 0; c < R; c++) acc[a][c] = fma(qv[a], kv[c], acc[a][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sc = float(scale[0]);
    for (int a = 0; a < R; a++) {
        const int l = int(l0 + ty * R) + a;
        for (int c = 0; c < R; c++) {
            const int j = int(j0 + tx * R) + c;
            if (l < L && j < KL) out[(size_t(bh) * L + l) * KL + j] = acc[a][c] * sc;
        }
    }
"""
_SRC_DATT_SOFTMAX = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint bh = threadgroup_position_in_grid.y;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    if (int(row) >= L) { return; }
    auto srow = s + (size_t(bh) * L + row) * KL;
    auto prow = out + (size_t(bh) * L + row) * KL;
    float m = -INFINITY;
    for (int j = int(lane); j < KL; j += 32) { m = max(m, float(srow[j])); }
    m = simd_max(m);
    for (int j = int(lane); j < KL; j += 32) {
        prow[j] = (srow[j] == -INFINITY) ? 0.0f : exp(float(srow[j]) - m);
    }
    simdgroup_barrier(mem_flags::mem_device);
    float sum = 0.0f;
    if (lane == 0) {
        for (int j = 0; j < KL; j++) { sum += prow[j]; }
    }
    sum = simd_broadcast(sum, 0);
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    simdgroup_barrier(mem_flags::mem_device);
    for (int j = int(lane); j < KL; j += 32) { prow[j] = prow[j] * inv; }
"""
_SRC_DATT_OUT = r"""
    constexpr int R = 4, TD = 32, TL = 32, JC = 16;
    threadgroup float ps[TL][JC + 1];
    threadgroup T vs[JC][TD + 1];
    const uint tx = thread_position_in_threadgroup.x;          // head-dim group (8)
    const uint ty = thread_position_in_threadgroup.y;          // row group (8)
    const uint d0 = threadgroup_position_in_grid.x * TD;
    const uint l0 = threadgroup_position_in_grid.y * TL;
    const uint bh = threadgroup_position_in_grid.z;
    const int L = int(dims[0]);
    const int KL = int(dims[1]);
    const uint b = bh / HQ, h = bh % HQ, kvh = h / (HQ / HKV);
    const uint tid = ty * 8 + tx;
    auto pb = p + (size_t(bh) * L) * KL;
    auto vb = v + ((size_t(b) * HKV + kvh) * size_t(KL)) * D;
    float acc[R][R];
    for (int a = 0; a < R; a++) for (int c = 0; c < R; c++) acc[a][c] = 0.0f;
    for (int j0 = 0; j0 < KL; j0 += JC) {
        for (uint i = tid; i < TL * JC; i += 64) {
            const int r = int(l0) + int(i / JC), c = j0 + int(i % JC);
            ps[i / JC][i % JC] = (r < L && c < KL) ? pb[size_t(r) * KL + c] : 0.0f;
        }
        for (uint i = tid; i < JC * TD; i += 64) {
            const int r = j0 + int(i / TD), c = int(d0) + int(i % TD);
            vs[i / TD][i % TD] = (r < KL && c < D) ? vb[size_t(r) * D + c] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int jn = min(JC, KL - j0);
        for (int jj = 0; jj < jn; jj++) {
            float pv[R], vv[R];
            for (int a = 0; a < R; a++) pv[a] = ps[ty * R + a][jj];
            for (int c = 0; c < R; c++) vv[c] = float(vs[jj][tx * R + c]);
            for (int a = 0; a < R; a++) for (int c = 0; c < R; c++) acc[a][c] = fma(pv[a], vv[c], acc[a][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int a = 0; a < R; a++) {
        const int l = int(l0 + ty * R) + a;
        for (int c = 0; c < R; c++) {
            const int dcol = int(d0 + tx * R) + c;
            if (l < L && dcol < D) out[(size_t(bh) * L + l) * D + dcol] = static_cast<T>(acc[a][c]);
        }
    }
"""


def dense_attention_exact(queries, keys, values, scale, mask):
    """queries [B, HQ, L, D]; keys / values [B, HKV, kL, D] (same dtype); mask None, "causal" (bottom-right aligned)
    or boolean array broadcastable to [B, HQ, L, kL]. Fixed-order attention, independent of L and kL."""
    B, HQ, L, D = queries.shape
    HKV, KL = keys.shape[1], keys.shape[2]
    assert D % 16 == 0, D
    dims = mx.array([L, KL], dtype=mx.int32)
    if "datt_s" not in _kernels:
        _kernels["datt_s"] = mx.fast.metal_kernel(name="mlx_exact_datt_scores_rb", input_names=["q", "k", "dims", "scale"],
                                                  output_names=["out"], source=_SRC_DATT_SCORES,
                                                  compile_options={"math_mode": "safe"})
        _kernels["datt_m"] = mx.fast.metal_kernel(name="mlx_exact_datt_softmax_sg", input_names=["s", "dims"],
                                                  output_names=["out"], source=_SRC_DATT_SOFTMAX,
                                                  compile_options={"math_mode": "safe"})
        _kernels["datt_o"] = mx.fast.metal_kernel(name="mlx_exact_datt_out_rb", input_names=["p", "v", "dims"],
                                                  output_names=["out"], source=_SRC_DATT_OUT,
                                                  compile_options={"math_mode": "safe"})
    tmpl = [("T", queries.dtype), ("HQ", HQ), ("HKV", HKV), ("D", D)]
    s = _kernels["datt_s"](inputs=[queries, keys, dims, mx.array([scale], dtype=mx.float32)], template=tmpl,
                           grid=(((KL + 31) // 32) * 8, ((L + 31) // 32) * 8, B * HQ), threadgroup=(8, 8, 1),
                           output_shapes=[(B, HQ, L, KL)], output_dtypes=[mx.float32])[0]
    if mask is not None:
        if isinstance(mask, str):
            mask = mx.arange(KL)[None, :] <= (mx.arange(L)[:, None] + KL - L)
        s = mx.where(mask, s, -mx.inf)
    p = _kernels["datt_m"](inputs=[s, dims], template=[("T", queries.dtype)], grid=(((L + 3) // 4) * 128, B * HQ, 1),
                           threadgroup=(128, 1, 1), output_shapes=[(B, HQ, L, KL)], output_dtypes=[mx.float32])[0]
    return _kernels["datt_o"](inputs=[p, values, dims], template=tmpl,
                              grid=(((D + 31) // 32) * 8, ((L + 31) // 32) * 8, B * HQ), threadgroup=(8, 8, 1),
                              output_shapes=[(B, HQ, L, D)], output_dtypes=[queries.dtype])[0]


# Per-sequence left shift of K / V rows (key re-alignment for batched left padding): out[b, h, j] = x[b, h, j + pad[b]],
# zeros past the end. A plain copy kernel (no index arrays); sizes are runtime arguments.
_SRC_SHIFT_ROWS = r"""
    const uint d = thread_position_in_grid.x;
    const uint j = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const int KL = int(dims[0]);
    const int H = int(dims[1]);
    const int D = int(dims[2]);
    if (int(d) >= D || int(j) >= KL) { return; }
    const int b = int(bh) / H;
    const int src = int(j) + int(pad[b]);
    out[(size_t(bh) * KL + j) * D + d] = (src < KL) ? x[(size_t(bh) * KL + src) * D + d] : T(0);
"""


def shift_rows_left(x, pad):
    """x [B, H, kL, D]; pad [B] int: out[b, :, j] = x[b, :, j + pad[b]] (zeros where j + pad[b] >= kL)."""
    B, H, KL, D = x.shape
    if "shift_rows" not in _kernels:
        _kernels["shift_rows"] = mx.fast.metal_kernel(name="mlx_exact_shift_rows", input_names=["x", "pad", "dims"],
                                                      output_names=["out"], source=_SRC_SHIFT_ROWS)
    return _kernels["shift_rows"](inputs=[x, pad.astype(mx.int32), mx.array([KL, H, D], dtype=mx.int32)],
                                  template=[("T", x.dtype)], grid=(D, KL, B * H), threadgroup=(min(D, 128), 1, 1),
                                  output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


# Per-sequence RIGHT shift of K / V rows into a longer buffer (absolute-position alignment for batched sliding-window
# caches): out[b, h, j] = x[b, h, j - r[b]] for 0 <= j - r[b] < kL, zeros elsewhere; out has KO >= kL rows.
_SRC_SHIFT_RIGHT = r"""
    const uint d = thread_position_in_grid.x;
    const uint j = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const int KL = int(dims[0]);
    const int KO = int(dims[1]);
    const int H = int(dims[2]);
    const int D = int(dims[3]);
    if (int(d) >= D || int(j) >= KO) { return; }
    const int b = int(bh) / H;
    const int src = int(j) - int(r[b]);
    out[(size_t(bh) * KO + j) * D + d] = (src >= 0 && src < KL) ? x[(size_t(bh) * KL + src) * D + d] : T(0);
"""


def shift_rows_right(x, r, out_len):
    """x [B, H, kL, D]; r [B] int in [0, out_len - kL]: out[b, :, j] = x[b, :, j - r[b]] (zeros elsewhere)."""
    B, H, KL, D = x.shape
    if "shift_right" not in _kernels:
        _kernels["shift_right"] = mx.fast.metal_kernel(name="mlx_exact_shift_rows_right", input_names=["x", "r", "dims"],
                                                       output_names=["out"], source=_SRC_SHIFT_RIGHT)
    return _kernels["shift_right"](inputs=[x, r.astype(mx.int32), mx.array([KL, out_len, H, D], dtype=mx.int32)],
                                   template=[("T", x.dtype)], grid=(D, out_len, B * H), threadgroup=(min(D, 128), 1, 1),
                                   output_shapes=[(B, H, out_len, D)], output_dtypes=[x.dtype])[0]
