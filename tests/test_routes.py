"""Every route of the quantized matmul (v7 for one row, v7_rows for 2-4, v4 for more, MLX's canonical kernel for long
prefills) must produce the same logits: decode and chunked prefill with chunk sizes 2-64 vs one full prefill, on small
random-weight models (no download), 4- and 8-bit."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_exact


@pytest.mark.parametrize("bits", [4, 8])
def test_all_chunk_sizes_match_full_prefill(bits):
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import make_prompt_cache
    cfg = dict(model_type="qwen3", hidden_size=1024, num_hidden_layers=2, intermediate_size=3072, num_attention_heads=8,
               num_key_value_heads=4, head_dim=128, rms_norm_eps=1e-6, vocab_size=4096, rope_theta=1e6,
               max_position_embeddings=4096, tie_word_embeddings=True)
    mx.random.seed(bits)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=bits)
    ids = np.random.default_rng(bits).integers(0, cfg["vocab_size"], 140).tolist()
    mlx_exact.enable()
    try:
        full = model(mx.array(ids)[None])[0]
        for c in (1, 2, 3, 4, 5, 17, 64):
            cache = make_prompt_cache(model)
            for s in range(0, len(ids), c):
                out = model(mx.array(ids[s:s + c])[None], cache=cache)[0]
                for i in range(out.shape[0]):
                    assert mx.array_equal(out[i], full[s + i]).item(), (bits, c, s + i)
    finally:
        mlx_exact.disable()
