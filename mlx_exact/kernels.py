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
    const device T* xr = x + size_t(m) * K_IN;
    const device uint32_t* wr = w + size_t(n) * (K_IN / 8);
    const device T* sr = scales + size_t(n) * (K_IN / GS);
    const device T* br = biases + size_t(n) * (K_IN / GS);
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
    const device T* xr = x + size_t(m) * K_IN;
    const device T* sr = scales + size_t(nn) * (K_IN / 64);
    const device T* br = biases + size_t(nn) * (K_IN / 64);
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
    const device T* xr = x + size_t(m) * K_IN;
    const device uint32_t* wr = w + (size_t(e) * N_OUT + n) * (K_IN / 8);
    const device T* sr = scales + (size_t(e) * N_OUT + n) * (K_IN / GS);
    const device T* br = biases + (size_t(e) * N_OUT + n) * (K_IN / GS);
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
    const device T* xr = x + size_t(m) * K_IN;
    const device uint32_t* wr = w + size_t(n) * (K_IN / PER_WORD);
    const device T* sr = scales + size_t(n) * (K_IN / GS);
    const device T* br = biases + size_t(n) * (K_IN / GS);
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
    const device T* xr = x + size_t(m) * K_IN;
    const device T* sr = scales + size_t(nn) * (K_IN / 64);
    const device T* br = biases + size_t(nn) * (K_IN / 64);
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
    const device T* xr = x + size_t(m) * K_IN;
    const device T* wr = w + size_t(n) * K_IN;
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
    const device T* xr = x + size_t(m) * K_IN;
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
