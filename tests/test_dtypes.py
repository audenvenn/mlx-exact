"""Unsupported activation dtypes fall back to stock MLX with a warning instead of silently computing a different
function (with fp32 activations MLX's canonical kernel runs at reduced, TF32-like precision by default)."""
import warnings

import numpy as np
import mlx.core as mx

import mlx_exact


def test_fp32_activations_fall_back_with_warning():
    rng = np.random.default_rng(0)
    w = mx.array((rng.standard_normal((256, 512)) * 0.02).astype(np.float32))
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    x = mx.array(rng.standard_normal((3, 512)).astype(np.float32))
    mlx_exact._warned.clear()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        y = mlx_exact.quantized_matmul(x, wq, s, b, group_size=64, bits=4)
    assert any("activations mlx.core.float32" in str(r.message) for r in rec)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    assert mx.array_equal(y, ref).item()
