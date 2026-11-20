"""MLX's accelerator attention kernel called directly with per-sequence key offsets (attn_nax.py) gives exactly the
values of the padded / re-aligned path (0.7.1) for decode, short and long chunks and left-padded batches;
a left-padded batch through a real model equals each sequence alone. In-place cache reads, gradients and the
review regressions: test_attention_review.py."""
import os

import numpy as np
import mlx.core as mx
import pytest

import mlx_exact.attention as A
from mlx_exact.attention import exact_sdpa
from mlx_exact.attn_nax import attention_offsets, first_valid_key

# one case per head dim and dtype (the full cross product, fp32 included, is in the research probe; FINDINGS F34)
CASES = [(D, dt) for D, dt in [(128, mx.bfloat16), (64, mx.bfloat16), (96, mx.float16)]
         if D in A.OFFSETS_HEAD_DIMS and dt in A.OFFSETS_DTYPES]
HQ, HKV = 16, 8


def _mk(rng, *s, dt=mx.bfloat16):
    return mx.array(rng.standard_normal(s).astype(np.float32)).astype(dt)


def _reference(*args):
    """The 0.7.1 path: query padding to 9 rows, key re-alignment, stock attention kernel."""
    saved = A.OFFSETS_HEAD_DIMS
    A.OFFSETS_HEAD_DIMS = ()
    try:
        return exact_sdpa(*args)
    finally:
        A.OFFSETS_HEAD_DIMS = saved


def _eq(a, b, rows=None):
    if rows is not None:
        sel = mx.broadcast_to(rows[..., None], a.shape)
        a, b = mx.where(sel, a, 0), mx.where(sel, b, 0)
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("D,dt", CASES)
@pytest.mark.parametrize("kL", [1, 8, 9, 32, 33, 257])
def test_decode_matches_reference(D, dt, kL):
    rng = np.random.default_rng(kL)
    q, k, v = _mk(rng, 1, HQ, 1, D, dt=dt), _mk(rng, 1, HKV, kL, D, dt=dt), _mk(rng, 1, HKV, kL, D, dt=dt)
    ref = _reference(q, k, v, None, D ** -0.5, None)
    assert _eq(attention_offsets(q, k, v, D ** -0.5), ref)                         # 16-row tile, heads packed
    assert _eq(attention_offsets(q, k, v, D ** -0.5, pack=False, wm=4), ref)       # MLX's own 64-row tile
    assert _eq(exact_sdpa(q, k, v, None, D ** -0.5, None), ref)


@pytest.mark.parametrize("D,dt", CASES)
@pytest.mark.parametrize("L,kL", [(2, 2), (5, 40), (8, 100), (9, 9), (16, 60), (37, 87)])
def test_causal_chunks_match_reference(D, dt, L, kL):
    rng = np.random.default_rng(L * 1000 + kL)
    q, k, v = _mk(rng, 1, HQ, L, D, dt=dt), _mk(rng, 1, HKV, kL, D, dt=dt), _mk(rng, 1, HKV, kL, D, dt=dt)
    ref = _reference(q, k, v, None, D ** -0.5, "causal")
    assert _eq(attention_offsets(q, k, v, D ** -0.5, causal=True), ref)
    assert _eq(attention_offsets(q, k, v, D ** -0.5, causal=True, wm=4), ref)
    assert _eq(exact_sdpa(q, k, v, None, D ** -0.5, "causal"), ref)


@pytest.mark.parametrize("D,dt", CASES)
@pytest.mark.parametrize("L", [1, 4, 16])
def test_left_padded_batch_matches_reference_and_alone(D, dt, L):
    from mlx_lm.models.base import create_causal_mask
    rng = np.random.default_rng(L)
    B, kL, pads = 4, 100, [0, 3, 31, 90]
    q, k, v = _mk(rng, B, HQ, L, D, dt=dt), _mk(rng, B, HKV, kL, D, dt=dt), _mk(rng, B, HKV, kL, D, dt=dt)
    mask = create_causal_mask(L, offset=kL - L, left_padding=mx.array(pads))
    valid = mx.any(mask, axis=-1)          # rows of padding tokens have no valid key; their outputs are not used
    out = exact_sdpa(q, k, v, None, D ** -0.5, mask)
    assert A.STATS.get("sdpa_offsets", 0) > 0
    assert _eq(out, _reference(q, k, v, None, D ** -0.5, mask), valid)
    assert _eq(attention_offsets(q, k, v, D ** -0.5, mask, first_valid_key(mask, B, kL), pack=False, wm=4), out, valid)
    for b, p in enumerate(pads):
        if kL - p >= L:
            solo = _reference(q[b:b + 1], k[b:b + 1, :, p:], v[b:b + 1, :, p:], None, D ** -0.5,
                              "causal" if L > 1 else None)
            assert _eq(out[b:b + 1], solo, valid[b:b + 1]), b


def test_left_padded_batch_through_a_model_equals_each_sequence_alone():
    """Prefill of a left-padded batch (mlx-lm BatchKVCache) and three decode steps give, bit for bit, the logits of
    each sequence run alone (skipped unless the model is in the local Hugging Face cache)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from mlx_lm import load
        from mlx_lm.models.cache import BatchKVCache, make_prompt_cache
        model, tok = load(os.environ.get("MLX_EXACT_TEST_MODEL", "mlx-community/Qwen3-0.6B-4bit"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model not available offline: {e}")
    import mlx_exact
    ids = tok.encode("Floating-point addition is not associative, so the order of a sum matters. " * 4)
    prompts = [ids[:7], ids[:23], ids[:41]]
    n = max(len(p) for p in prompts)
    pads = [n - len(p) for p in prompts]
    mlx_exact.enable()
    try:
        solo_caches = [make_prompt_cache(model) for _ in prompts]
        solo = [model(mx.array(p)[None], cache=c)[0, -1] for p, c in zip(prompts, solo_caches)]
        caches = [BatchKVCache(pads) for _ in model.layers]
        batch = model(mx.array([[0] * d + p for p, d in zip(prompts, pads)]), cache=caches)[:, -1]
        for step in range(4):
            for b in range(len(prompts)):
                assert mx.array_equal(batch[b], solo[b]).item(), (step, b)
            nxt = [int(mx.argmax(s).item()) for s in solo]
            solo = [model(mx.array([[t]]), cache=c)[0, -1] for t, c in zip(nxt, solo_caches)]
            batch = model(mx.array(nxt)[:, None], cache=caches)[:, -1]
    finally:
        mlx_exact.disable()
