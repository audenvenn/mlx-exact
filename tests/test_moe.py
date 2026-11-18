"""MoE expert layers: the expert-indexed exact kernel equals MLX's canonical sorted expert kernel per row, and a small
random-weight qwen3_moe model (no download) is schedule-invariant end to end."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_exact.kernels import gather_qmv_exact_v7, gather_qmv_nax_exact


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_expert_kernel_matches_canonical_gather(dtype):
    E, N, K, B = 16, 1024, 2048, 256
    rng = np.random.default_rng(8)
    W = mx.array((rng.standard_normal((E, N, K)) * 0.02).astype(np.float32)).astype(dtype)
    wq, s, b = mx.quantize(W, group_size=64, bits=4)
    x = mx.array(rng.standard_normal((B, 1, K)).astype(np.float32)).astype(dtype)
    idx = mx.array(np.sort(rng.integers(0, E, B)).astype(np.uint32))
    canon = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=4, sorted_indices=True)
    ours = gather_qmv_nax_exact(x.reshape(B, K), wq, s, b, idx)
    assert mx.array_equal(ours, canon.reshape(B, N)).item()


@pytest.mark.parametrize("bits", [4, 8])
def test_random_moe_model_decode_equals_prefill(bits):
    from mlx_lm.models import qwen3_moe
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_exact
    cfg = dict(model_type="qwen3_moe", hidden_size=512, num_hidden_layers=2, intermediate_size=1024,
               num_attention_heads=8, num_experts=16, num_experts_per_tok=4, decoder_sparse_step=1, mlp_only_layers=[],
               moe_intermediate_size=512, rms_norm_eps=1e-6, vocab_size=2048, num_key_value_heads=4, head_dim=64,
               rope_theta=1e6, tie_word_embeddings=False, max_position_embeddings=2048, norm_topk_prob=True)
    mx.random.seed(0)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=bits)
    ids = np.random.default_rng(1).integers(0, cfg["vocab_size"], 120).tolist()
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        cache = make_prompt_cache(model)
        for t in range(len(ids)):
            assert mx.array_equal(model(mx.array(ids[t:t + 1])[None], cache=cache)[0, -1], full[t]).item(), t
        cache = make_prompt_cache(model)
        for s0 in range(0, len(ids), 9):
            blk = model(mx.array(ids[s0:s0 + 9])[None], cache=cache)[0]
            assert mx.array_equal(blk, full[s0:s0 + blk.shape[0]]).item(), s0
    finally:
        mlx_exact.disable()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("E,N,K", [(16, 768, 2048), (16, 2048, 768), (8, 1000, 512)])
def test_expert_v7_kernel_matches_canonical_gather(bits, dtype, E, N, K):
    rng = np.random.default_rng(E + N + K + bits)
    W = mx.array((rng.standard_normal((E, N, K)) * 0.02).astype(np.float32)).astype(dtype)
    wq, s, b = mx.quantize(W, group_size=64, bits=bits)
    B = 128
    x = mx.array(rng.standard_normal((B, 1, K)).astype(np.float32)).astype(dtype)
    idx = mx.array(np.sort(rng.integers(0, E, B)).astype(np.uint32))
    canon = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=bits, sorted_indices=True)
    assert mx.array_equal(gather_qmv_exact_v7(x.reshape(B, K), wq, s, b, idx, 64, bits), canon.reshape(B, N)).item()
