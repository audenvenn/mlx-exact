"""End to end on a real model (skipped unless it is in the local Hugging Face cache): decode == prefill, bit for bit."""
import os

import mlx.core as mx
from mlx_exact.testing import bits_equal
import pytest

MODEL = os.environ.get("MLX_EXACT_TEST_MODEL", "mlx-community/Qwen3-0.6B-4bit")


def test_decode_equals_prefill():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        model, tok = load(MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model not available offline: {e}")
    import mlx_exact
    mlx_exact.enable()
    try:
        ids = tok.encode("Floating-point addition is not associative, so summation order matters. " * 8)[:80]
        full = model(mx.array(ids)[None])[0]
        cache = make_prompt_cache(model)
        for t in range(len(ids)):
            row = model(mx.array(ids[t:t + 1])[None], cache=cache)[0, -1]
            assert bits_equal(row, full[t]).item(), t
    finally:
        mlx_exact.disable()
