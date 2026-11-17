"""Replica of MLX 0.32.4's kernel choice for 4-bit `quantized_matmul` with transposed, non-batched weights (every
`nn.QuantizedLinear`) on GPUs of architecture generation >= 17 that are not 'd'-class (validated on Apple M6).
Source: mlx/backend/metal/quantized.cpp (v0.32.4): get_qmv_batch_limit, QuantizedMatmul::eval_gpu, qmm_splitk, qmm_nax.
"""
import math

CANONICAL = "qmm_t_nax_bm64"


def qmv_batch_limit(K: int, N: int) -> int:
    if K <= 2048 and N <= 2048:
        return 33
    if K <= 4096 and N <= 4096:
        return 25
    return 13


def stock_qmm_path(M: int, N: int, K: int, group_size: int = 64) -> str:
    """Name of the kernel MLX would run for x[M, K] @ dequant(w[N, K]).T."""
    if M < qmv_batch_limit(K, N):
        if K in (64, 128):
            return "qmv_quad"
        return "qmv" if M == 1 else "qmv_wide"
    tgs = math.ceil(N / 32) * math.ceil(M / 32)
    split_k = max(1, 512 // tgs)
    k_align = max(group_size, 32)
    split_k = min(split_k, K // k_align)
    while split_k > 1 and K % (split_k * k_align) != 0:
        split_k -= 1
    if split_k > 1:
        return f"qmm_t_splitk{split_k}"
    if K % 64 == 0:
        return "qmm_t_nax_bm32" if M <= 32 else "qmm_t_nax_bm64"
    return "qmm_t"


def dense_stock_path(M: int, N: int, K: int) -> str:
    """Kernel MLX 0.32.4 picks for a 2-D x[M, K] @ W.T (W [N, K], bf16/fp16) on gen >= 17 (matmul.cpp)."""
    if M == 1 or N == 1:
        return "gemv"
    if 2 <= M <= 14 and K % 4 == 0:
        return "gemv_wide"
    if N <= 64 or M <= 64:
        return "thin_nax"
    m = max(M, N)
    if K >= 3 * m or (m <= 1024 and K > 2 * m):
        return "splitk_nax"
    return "nax"
