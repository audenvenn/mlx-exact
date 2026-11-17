"""Unquantized layers: the dense exact kernel equals MLX's canonical (regular accelerator) GEMM wherever the dispatch
replica predicts that path."""
import numpy as np
import mlx.core as mx
import pytest

from mlx_exact.dispatch import dense_stock_path
from mlx_exact.kernels import dense_exact


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", [(3584, 1024), (2048, 1024), (1024, 3584), (6144, 1024)])
def test_dense_kernel_matches_canonical(dtype, N, K):
    rng = np.random.default_rng(N + K)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    x = mx.array(rng.standard_normal((1300, K)).astype(np.float32)).astype(dtype)
    ours = dense_exact(x[:3], W)
    checked = 0
    for M in (65, 300, 1300):
        if dense_stock_path(M, N, K) == "nax":
            assert mx.array_equal((x[:M] @ W.T)[:3], ours).item(), M
            checked += 1
    assert checked >= 1


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", [(1024, 1024), (1000, 1024), (1024, 3584), (3584, 1024)])
def test_dense_v7_matches_canonical(dtype, N, K):
    from mlx_exact.kernels import dense_exact_v7
    rng = np.random.default_rng(N * 3 + K)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    Mc = 2048 if max(N, K) > 1024 else 1024
    assert dense_stock_path(Mc, N, K) == "nax"
    x = mx.array(rng.standard_normal((Mc, K)).astype(np.float32)).astype(dtype)
    ref = x @ W.T
    for M in (1, 2, 4):
        assert mx.array_equal(dense_exact_v7(x[:M], W), ref[:M]).item(), M
