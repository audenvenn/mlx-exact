"""The exact kernels reproduce MLX's canonical (long-prefill) 4-bit matmul bit for bit, for any number of rows."""
import numpy as np
import mlx.core as mx
import pytest

from mlx_exact.dispatch import stock_qmm_path
from mlx_exact.kernels import qmv_nax_exact, qmv_nax_exact_v6
from mlx_exact.nax_kernel import qmm_nax_exact_v4


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", [(2048, 1024), (1024, 3072), (4096, 4096), (1000, 512)])
def test_kernels_match_canonical_prefill(dtype, N, K):
    assert stock_qmm_path(512, N, K) == "qmm_t_nax_bm64"
    rng = np.random.default_rng(N + K)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    x = mx.array(rng.standard_normal((512, K)).astype(np.float32)).astype(dtype)
    wq, s, b = mx.quantize(W, group_size=64, bits=4)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)      # canonical path
    for M in (1, 7, 40):
        for fn in (qmv_nax_exact, qmv_nax_exact_v6, qmm_nax_exact_v4):
            y = fn(x[:M], wq, s, b)
            assert mx.array_equal(y, ref[:M]).item(), (fn.__name__, M)
    assert mx.array_equal(qmv_nax_exact(x[0], wq, s, b), ref[0]).item()
