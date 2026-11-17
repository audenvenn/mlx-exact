"""Short query blocks and left-padded batches give the same rows as full prefill."""
import numpy as np
import mlx.core as mx
import pytest

from mlx_exact.attention import exact_sdpa


def _mk(rng, *s):
    return mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)


@pytest.mark.parametrize("D,HQ,HKV", [(128, 16, 8), (256, 8, 2)])
@pytest.mark.parametrize("T", [3, 12, 40, 300])
def test_decode_rows_equal_prefill_rows(D, HQ, HKV, T):
    rng = np.random.default_rng(T + D)
    q, k, v = _mk(rng, 1, HQ, T, D), _mk(rng, 1, HKV, T, D), _mk(rng, 1, HKV, T, D)
    full = exact_sdpa(q, k, v, None, D ** -0.5, "causal" if T > 1 else None)
    for t in range(T):
        one = exact_sdpa(q[:, :, t:t + 1], k[:, :, :t + 1], v[:, :, :t + 1], None, D ** -0.5, None)
        assert mx.array_equal(one[:, :, 0], full[:, :, t]).item(), t
    for c in (4, 9, 16):
        for s0 in range(0, T, c):
            e0 = min(T, s0 + c)
            blk = exact_sdpa(q[:, :, s0:e0], k[:, :, :e0], v[:, :, :e0], None, D ** -0.5,
                             "causal" if e0 - s0 > 1 else None)
            assert mx.array_equal(blk, full[:, :, s0:e0]).item(), (c, s0)


@pytest.mark.parametrize("n", [17, 30, 59])
def test_left_padding_is_realigned(n):
    rng = np.random.default_rng(n)
    D = 128
    q, k, v = _mk(rng, 1, 16, 1, D), _mk(rng, 1, 8, n, D), _mk(rng, 1, 8, n, D)
    solo = exact_sdpa(q, k, v, None, D ** -0.5, None)
    for p in (1, 7, 13, 29, 42):
        kp = mx.concatenate([_mk(rng, 1, 8, p, D), k], axis=2)
        vp = mx.concatenate([_mk(rng, 1, 8, p, D), v], axis=2)
        m = mx.array(np.arange(n + p) >= p)[None, None, None, :]
        assert mx.array_equal(exact_sdpa(q, kp, vp, None, D ** -0.5, m), solo).item(), p
