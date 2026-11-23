"""The exact kernels reproduce MLX's canonical (long-prefill) 4-bit matmul bit for bit, for any number of rows."""
import numpy as np
import mlx.core as mx
from mlx_exact.testing import bits_equal
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
            assert bits_equal(y, ref[:M]).item(), (fn.__name__, M)
    assert bits_equal(qmv_nax_exact(x[0], wq, s, b), ref[0]).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_8bit_kernel_matches_canonical_prefill(dtype):
    from mlx_exact.kernels import qmv_exact_bits
    N, K = 2048, 1024
    rng = np.random.default_rng(88)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    x = mx.array(rng.standard_normal((512, K)).astype(np.float32)).astype(dtype)
    wq, s, b = mx.quantize(W, group_size=64, bits=8)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=8)
    assert bits_equal(qmv_exact_bits(x[:3], wq, s, b, bits=8), ref[:3]).item()
    from mlx_exact.kernels import qmv_exact_v6b
    assert bits_equal(qmv_exact_v6b(x[:3], wq, s, b, bits=8), ref[:3]).item()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", [(1024, 1024), (2048, 1024), (1024, 3072), (1000, 512), (6144, 2048)])
def test_v7_decode_kernel_matches_canonical_prefill(bits, dtype, N, K):
    from mlx_exact.kernels import qmv_exact_v7
    rng = np.random.default_rng(N + K + bits)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    x = mx.array(rng.standard_normal((512, K)).astype(np.float32)).astype(dtype)
    wq, s, b = mx.quantize(W, group_size=64, bits=bits)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
    for M in (1, 3, 16):
        assert bits_equal(qmv_exact_v7(x[:M], wq, s, b, 64, bits), ref[:M]).item(), M
    assert bits_equal(qmv_exact_v7(x[5], wq, s, b, 64, bits), ref[5]).item()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", [(1024, 1024), (1000, 512), (2048, 2048), (6144, 2048)])
def test_multirow_kernels_match_canonical_prefill(bits, dtype, N, K):
    from mlx_exact.kernels import qmv_exact_v7_rows
    from mlx_exact.nax_kernel import qmm_nax_exact_v4
    rng = np.random.default_rng(N + K + bits + 1)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    x = mx.array(rng.standard_normal((512, K)).astype(np.float32)).astype(dtype)
    wq, s, b = mx.quantize(W, group_size=64, bits=bits)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
    if K % 256 == 0:
        for M in (1, 2, 3, 4):
            assert bits_equal(qmv_exact_v7_rows(x[:M], wq, s, b, 64, bits), ref[:M]).item(), ("rows", M)
    for M in (1, 4, 5, 16, 17, 100):
        assert bits_equal(qmm_nax_exact_v4(x[:M], wq, s, b, 64, bits), ref[:M]).item(), ("v4", M)
