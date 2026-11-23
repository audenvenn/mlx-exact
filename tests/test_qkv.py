"""Quantized KV cache: the attention arithmetic is the same for any number of queries and as accurate as stock decode."""
import numpy as np
import mlx.core as mx
from mlx_exact.testing import bits_equal
import mlx.nn as nn

from mlx_exact.kernels import qkv_attention_exact


def test_quantized_kv_attention_invariant_and_accurate():
    B, HQ, HKV, D, KL = 1, 8, 4, 128, 300
    rng = np.random.default_rng(3)
    q0 = rng.standard_normal((B, HQ, 64, D)).astype(np.float32)
    fresh = lambda: mx.array(q0).astype(mx.bfloat16)        # mlx-lm's stock function mutates its queries in place
    k = mx.array(rng.standard_normal((B, HKV, KL, D)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((B, HKV, KL, D)).astype(np.float32)).astype(mx.bfloat16)
    kq, vq = mx.quantize(k, group_size=64, bits=8), mx.quantize(v, group_size=64, bits=8)
    pre = qkv_attention_exact(fresh(), kq, vq, D ** -0.5, "causal", 64, 8)
    for t in (0, 17, 63):
        n = KL - 64 + t + 1
        one = qkv_attention_exact(fresh()[:, :, t:t + 1], (kq[0][:, :, :n], kq[1][:, :, :n], kq[2][:, :, :n]),
                                  (vq[0][:, :, :n], vq[1][:, :, :n], vq[2][:, :, :n]), D ** -0.5, None, 64, 8)
        assert bits_equal(one[:, :, 0], pre[:, :, t]).item(), t
    kn = np.array(mx.dequantize(*kq, group_size=64, bits=8).astype(mx.float32)).astype(np.float64)[0]
    vn = np.array(mx.dequantize(*vq, group_size=64, bits=8).astype(mx.float32)).astype(np.float64)[0]
    qn = q0.astype(np.float64)[0]
    out = np.array(pre.astype(mx.float32))[0, :, 63]
    for h in range(HQ):
        s = (qn[h, 63] * D ** -0.5) @ kn[h // 2].T
        e = np.exp(s - s.max()); e /= e.sum()
        assert np.abs(out[h] - e @ vn[h // 2]).max() < 2e-2


def test_random_model_with_quantized_kv_cache():
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import QuantizedKVCache
    import mlx_exact
    cfg = dict(model_type="qwen3", hidden_size=512, num_hidden_layers=2, intermediate_size=1024, num_attention_heads=8,
               num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6, vocab_size=2048, rope_theta=1e6,
               max_position_embeddings=2048, tie_word_embeddings=True)
    mx.random.seed(0)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    ids = mx.array(np.random.default_rng(2).integers(0, cfg["vocab_size"], 90))
    newc = lambda: [QuantizedKVCache(group_size=64, bits=8) for _ in model.layers]
    mlx_exact.enable()
    try:
        full = model(ids[None], cache=newc())[0]
        cache = newc()
        for t in range(ids.shape[0]):
            assert bits_equal(model(ids[t:t + 1][None], cache=cache)[0, -1], full[t]).item(), t
    finally:
        mlx_exact.disable()
