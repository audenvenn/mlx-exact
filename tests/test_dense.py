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



@pytest.mark.parametrize("tile", [16, 128])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", [(1024, 1024), (1000, 1024), (1024, 3584), (2048, 768), (16, 1024)])
def test_dense_v4_matches_canonical(tile, dtype, N, K):
    from mlx_exact.kernels import dense_exact
    from mlx_exact.nax_kernel import dense_nax_exact_v4
    rng = np.random.default_rng(N * 5 + K)
    W = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(dtype)
    Mc = next((m for m in (1024, 2048, 4096) if dense_stock_path(m, N, K) == "nax"), None)
    x = mx.array(rng.standard_normal((Mc or 128, K)).astype(np.float32)).astype(dtype)
    ref = x @ W.T if Mc else dense_exact(x, W)        # N <= 64: stock never runs the regular GEMM
    for M in (1, 5, 17, 100):
        assert mx.array_equal(dense_nax_exact_v4(x[:M], W, tile), ref[:M]).item(), M


def test_unquantized_model_all_chunk_sizes_match_full_prefill():
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_exact
    cfg = dict(model_type="qwen3", hidden_size=1024, num_hidden_layers=2, intermediate_size=3072, num_attention_heads=8,
               num_key_value_heads=4, head_dim=128, rms_norm_eps=1e-6, vocab_size=4096, rope_theta=1e6,
               max_position_embeddings=4096, tie_word_embeddings=True)
    mx.random.seed(11)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    ids = np.random.default_rng(11).integers(0, cfg["vocab_size"], 200).tolist()
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        for c in (1, 3, 5, 17, 64, 130):
            cache = make_prompt_cache(model)
            for s in range(0, len(ids), c):
                out = model(mx.array(ids[s:s + c])[None], cache=cache)[0]
                for i in range(out.shape[0]):
                    assert mx.array_equal(out[i], full[s + i]).item(), (c, s + i)
    finally:
        mlx_exact.disable()
