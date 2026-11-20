"""MX-format (mxfp4) layers: MLX's own fp_qmm_t_nax arithmetic with a 16-row tile (fp_kernel.py). The dense kernel
reproduces stock's long-prefill rows for any row count, the expert kernel equals stock's sorted expert kernel, and a
tiny gpt-oss model with mxfp4 experts decodes / prefills in chunks bit-identically to one full prefill (no
download)."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_exact.fp_kernel import fp_gather_exact_v4, fp_qmm_exact_v4


def _q(rng, shape, dt):
    w = mx.array(rng.standard_normal(shape).astype(np.float32) * 0.05).astype(dt)
    return mx.quantize(w, group_size=32, bits=4, mode="mxfp4")


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("K,N", [(384, 320), (2880, 512)])
def test_dense_rows_equal_the_long_prefill_rows(dt, K, N):
    rng = np.random.default_rng(K + N)
    wq, sc = _q(rng, (N, K), dt)
    x = mx.array(rng.standard_normal((2048, K)).astype(np.float32)).astype(dt)
    ref = mx.quantized_matmul(x, wq, sc, None, transpose=True, group_size=32, bits=4, mode="mxfp4")
    assert mx.array_equal(fp_qmm_exact_v4(x, wq, sc), ref).item()
    for M in (1, 3, 16, 37):
        assert mx.array_equal(fp_qmm_exact_v4(x[:M], wq, sc), ref[:M]).item(), M


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
def test_expert_kernel_equals_stock_sorted_expert_kernel(dt):
    rng = np.random.default_rng(7)
    E, K, N, B = 8, 384, 320, 64
    wq, sc = _q(rng, (E, N, K), dt)
    idx = mx.array(np.repeat(np.arange(E), B // E).astype(np.uint32))
    x = mx.array(rng.standard_normal((B, 1, K)).astype(np.float32)).astype(dt)
    stock = mx.gather_qmm(x, wq, sc, None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4",
                          sorted_indices=True).reshape(B, N)
    assert mx.array_equal(fp_gather_exact_v4(x.reshape(B, K), wq, sc, idx), stock).item()


def test_tiny_gpt_oss_with_mxfp4_experts_is_bit_exact():
    import importlib
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.switch_layers import SwitchLinear
    import mlx_exact
    mod = importlib.import_module("mlx_lm.models.gpt_oss")
    cfg = dict(model_type="gpt_oss", hidden_size=256, num_hidden_layers=2, intermediate_size=512,
               num_attention_heads=4, num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6, vocab_size=1024,
               rope_theta=10000.0, max_position_embeddings=2048, tie_word_embeddings=False, num_local_experts=8,
               num_experts_per_tok=2, sliding_window=32)
    mx.random.seed(1)
    model = mod.Model(mod.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=4, mode="mxfp4", class_predicate=lambda p, m: isinstance(m, SwitchLinear))
    ids = np.random.default_rng(0).integers(0, 1024, 70).tolist()
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        for c in (1, 3, 17):
            cache = make_prompt_cache(model)
            for s0 in range(0, len(ids), c):
                out = model(mx.array(ids[s0:s0 + c])[None], cache=cache)[0]
                for i in range(out.shape[0]):
                    assert mx.array_equal(out[i], full[s0 + i]).item(), (c, s0 + i)
        assert any(k.startswith("gather_mxfp4") for k, v in mlx_exact.stats()["matmul"].items() if v)
    finally:
        mlx_exact.disable()
