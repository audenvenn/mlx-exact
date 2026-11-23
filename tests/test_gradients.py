"""Gradients through routes that had none, and mx.matmul's stream argument (0.13.1).
- mxfp4 layers (bias-free QuantizedLinear / QuantizedSwitchLinear, e.g. gpt-oss) called Metal kernels directly
  since 0.9.0, and Gemma 2's attention used the lane softmax without a backward pass in 0.13.0: "VJP not implemented
  for CustomKernel".
- In 0.13.0 the raw-matmul patch rejected mx.matmul(a, b, stream=...).
Tiny random-weight models; no download."""
import importlib

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_exact.testing import bits_equal


def _bf(rng, *shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.linalg.norm(a - b) / (mx.linalg.norm(b) + 1e-12)).item())


def test_mxfp4_linear_gradient_matches_stock():
    import mlx_exact
    rng = np.random.default_rng(0)
    lin = nn.Linear(256, 512, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=32, bits=4, mode="mxfp4")
    x = _bf(rng, 3, 256)
    f = lambda xx: q(xx).astype(mx.float32).sum()  # noqa: E731
    g_stock = mx.grad(f)(x)
    mlx_exact.enable()
    try:
        g_exact = mx.grad(f)(x)
    finally:
        mlx_exact.disable()
    assert _rel(g_exact, g_stock) < 1e-2


def _tiny(arch, **extra):
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(1)
    base = dict(model_type=arch, hidden_size=256, num_hidden_layers=2, intermediate_size=512, num_attention_heads=4,
                num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6, vocab_size=1024, rope_theta=10000.0,
                max_position_embeddings=4096, tie_word_embeddings=True)
    model = mod.Model(mod.ModelArgs.from_dict(dict(base, **extra)))
    model.set_dtype(mx.bfloat16)
    return model


def _grads(model, exact):
    import mlx_exact
    ids = mx.array(np.random.default_rng(2).integers(0, 1024, (1, 24)))
    loss = lambda m: m(ids).astype(mx.float32).mean()  # noqa: E731
    if exact:
        mlx_exact.enable()
    try:
        _, g = nn.value_and_grad(model, loss)(model)
        mx.eval(g)
    finally:
        if exact:
            mlx_exact.disable()
    return g


def test_gemma2_attention_has_a_gradient():
    from mlx.utils import tree_flatten
    model = _tiny("gemma2", head_dim=256, num_attention_heads=8, num_key_value_heads=4, query_pre_attn_scalar=256)
    g_stock = dict(tree_flatten(_grads(model, False)))
    g_exact = dict(tree_flatten(_grads(model, True)))
    k = "model.layers.0.self_attn.q_proj.weight"
    assert k in g_exact and float(mx.abs(g_exact[k]).max().item()) > 0
    assert _rel(g_exact[k], g_stock[k]) < 5e-2          # different (canonical) forward rounding, same gradient


def test_gpt_oss_mxfp4_experts_have_a_gradient():
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import SwitchLinear
    model = _tiny("gpt_oss", num_local_experts=8, num_experts_per_tok=2, sliding_window=32)
    nn.quantize(model, group_size=32, bits=4, mode="mxfp4", class_predicate=lambda p, m: isinstance(m, SwitchLinear))
    g = dict(tree_flatten(_grads(model, True)))
    k = "model.layers.0.self_attn.q_proj.weight"
    assert k in g and float(mx.abs(g[k]).max().item()) > 0


def test_matmul_accepts_a_stream_in_exact_mode():
    import mlx_exact
    rng = np.random.default_rng(3)
    a, b = _bf(rng, 5, 256), _bf(rng, 256, 64)
    s = mx.new_stream(mx.gpu)
    mlx_exact.enable()
    try:
        y = mx.matmul(a, b, stream=s)
        ref = a @ b
        assert bits_equal(y, ref).item()
    finally:
        mlx_exact.disable()


def test_bits_equal_distinguishes_signed_zeros():
    a, b = mx.array([0.0, 1.0], mx.bfloat16), mx.array([-0.0, 1.0], mx.bfloat16)
    assert mx.array_equal(a, b).item() and not bits_equal(a, b).item()
