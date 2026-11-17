"""RL: with LoRA adapters, the sampler's log-probabilities (decode with a KV cache) equal the trainer's (one full
forward), bit for bit, and gradients flow through the exact routes. Small random-weight model; no download."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn


def test_sampler_equals_trainer_and_gradients_flow():
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    import mlx_exact
    cfg = dict(model_type="qwen3", hidden_size=512, num_hidden_layers=2, intermediate_size=1024, num_attention_heads=8,
               num_key_value_heads=4, head_dim=64, rms_norm_eps=1e-6, vocab_size=2048, rope_theta=1e6,
               max_position_embeddings=2048, tie_word_embeddings=True)
    mx.random.seed(0)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 8, "scale": 20.0, "dropout": 0.0,
                                     "keys": ["self_attn.q_proj", "self_attn.v_proj"]})
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m.lora_b = mx.random.normal(m.lora_b.shape) * 0.02
    prompt = mx.array(np.random.default_rng(0).integers(0, cfg["vocab_size"], 20))
    mlx_exact.enable()
    try:
        cache = make_prompt_cache(model)
        lg = model(prompt[None], cache=cache)[0, -1]
        toks, s_lp = [], []
        for _ in range(24):
            lp = lg.astype(mx.float32) - mx.logsumexp(lg.astype(mx.float32))
            t = int(mx.argmax(lp).item())
            toks.append(t); s_lp.append(lp[t])
            lg = model(mx.array([t])[None], cache=cache)[0, -1]
        full = mx.concatenate([prompt, mx.array(toks)])

        def trainer_lp(m):
            lgf = m(full[None])[0][len(prompt) - 1:-1].astype(mx.float32)
            lp = lgf - mx.logsumexp(lgf, axis=-1, keepdims=True)
            return mx.take_along_axis(lp, mx.array(toks)[:, None], axis=-1)[:, 0]
        assert mx.array_equal(trainer_lp(model), mx.stack(s_lp)).item()
        loss, grads = nn.value_and_grad(model, lambda m: -mx.mean(trainer_lp(m)))(model)
        gn = sum(float(mx.sum(g.astype(mx.float32) ** 2).item()) for _, g in tree_flatten(grads))
        assert np.isfinite(float(loss.item())) and gn > 0
    finally:
        mlx_exact.disable()
