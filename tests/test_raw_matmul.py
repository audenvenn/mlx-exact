"""Raw matmuls (`a @ b`, `mx.matmul`) in exact mode (0.13.0): MoE router gates, tied heads, Gemma 2's attention.
A row's result must not depend on how many rows the call has (decode vs prefill). Tiny random-weight models; no
download."""
import importlib

import numpy as np
import mlx.core as mx
import pytest


def _bf(rng, *shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)


def test_raw_matmul_rows_do_not_depend_on_the_row_count():
    import mlx_exact
    rng = np.random.default_rng(0)
    a = _bf(rng, 2, 4, 70, 256)            # batched, attention-like: [B, H, rows, K] @ [B, H, K, N]
    b = _bf(rng, 2, 4, 256, 300)
    x = _bf(rng, 70, 1024)                 # 2-D weight applied raw: x @ w.T (router gate, tied head)
    w = _bf(rng, 64, 1024)
    mlx_exact.enable()
    try:
        full_b, full_w = a @ b, x @ w.T
        for m in (1, 2, 3, 8, 15):
            assert mx.array_equal(a[:, :, -m:] @ b, full_b[:, :, -m:]).item(), m
            assert mx.array_equal(mx.matmul(x[-m:], w.T), full_w[-m:]).item(), m
    finally:
        mlx_exact.disable()


def _sweep_check(model, ids):
    from mlx_lm.models.cache import make_prompt_cache
    full = model(mx.array(ids)[None])[0]
    out = {}
    for c in (1, 3, 17):
        cache, same = make_prompt_cache(model), 0
        for s in range(0, len(ids), c):
            o = model(mx.array(ids[s:s + c])[None], cache=cache)[0]
            same += sum(int(mx.array_equal(o[i], full[s + i]).item()) for i in range(o.shape[0]))
        out[c] = same
    return out


BASE = dict(hidden_size=256, num_hidden_layers=2, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
            head_dim=64, rms_norm_eps=1e-6, vocab_size=1024, rope_theta=10000.0, max_position_embeddings=8192,
            tie_word_embeddings=True)


def _model(arch, **extra):
    mod = importlib.import_module(f"mlx_lm.models.{arch}")
    mx.random.seed(1)
    model = mod.Model(mod.ModelArgs.from_dict(dict(BASE, model_type=arch, **extra)))
    model.set_dtype(mx.bfloat16)
    return model


@pytest.mark.parametrize("arch,extra", [
    ("gemma2", dict(head_dim=256, num_attention_heads=8, num_key_value_heads=4, query_pre_attn_scalar=256)),
    ("seed_oss", dict(tie_word_embeddings=True)),
])
def test_raw_matmul_models_decode_and_chunks_equal_full_prefill(arch, extra):
    import mlx_exact
    model = _model(arch, **extra)
    ids = np.random.default_rng(0).integers(0, 1024, 70).tolist()
    mlx_exact.enable()
    try:
        res = _sweep_check(model, ids)
    finally:
        mlx_exact.disable()
    assert res == {1: 70, 3: 70, 17: 70}, res


def test_gemma2_decode_past_4096_keys_equals_prefill():
    """MLX's softmax switches kernels above 4,096 keys; Gemma 2's attention uses mlx-exact's softmax instead."""
    from mlx_lm.models.cache import make_prompt_cache
    import mlx_exact
    model = _model("gemma2", head_dim=256, num_attention_heads=8, num_key_value_heads=4, query_pre_attn_scalar=256)
    ids = np.random.default_rng(0).integers(0, 1024, 4200).tolist()
    pos = list(range(4080, 4200, 8))
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None], cache=make_prompt_cache(model))[0]
        ref = {p: full[p] for p in pos}
        cache = make_prompt_cache(model)
        model(mx.array(ids[:4080])[None], cache=cache)
        got = {}
        for p in range(4080, 4200):
            o = model(mx.array([[ids[p]]]), cache=cache)[0, -1]
            if p in ref:
                got[p] = o
    finally:
        mlx_exact.disable()
    for p in pos:
        assert mx.array_equal(got[p], ref[p]).item(), p
