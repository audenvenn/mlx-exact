"""Architecture coverage: are tiny random-weight models of other mlx-lm architectures schedule-invariant with mlx-exact?
For each architecture, bf16 and 4-bit: decode and chunks of 3 / 17 vs one full prefill at every position (stock vs
mlx-exact), and exact-mode values vs stock values. Sliding windows are set below the prompt length. No download.
Usage: python experiments/architecture_sweep.py
"""
import importlib

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

import mlx_exact

BASE = dict(hidden_size=256, num_hidden_layers=2, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
            head_dim=64, rms_norm_eps=1e-6, vocab_size=1024, rope_theta=10000.0, max_position_embeddings=2048,
            tie_word_embeddings=True)
CFG = {
    "llama": {}, "qwen2": {}, "phi3": {}, "olmo2": {}, "internlm3": {}, "smollm3": {},
    "hunyuan_v1_dense": {}, "seed_oss": dict(tie_word_embeddings=False), "starcoder2": {},
    "gemma2": dict(query_pre_attn_scalar=64),
    "gemma3_text": dict(query_pre_attn_scalar=64, sliding_window=32, sliding_window_pattern=2),
    "granite": dict(logits_scaling=1.0, attention_multiplier=0.125, embedding_multiplier=1.0, residual_multiplier=1.0,
                    attention_bias=False, mlp_bias=False),
    "glm4": dict(attention_bias=False, partial_rotary_factor=0.5),
    "cohere2": dict(layer_norm_eps=1e-5, sliding_window=32, sliding_window_pattern=2, logit_scale=1.0),
    "exaone4": dict(rope_scaling=None, sliding_window=32, sliding_window_pattern="LG"),
    "gpt_oss": dict(num_local_experts=8, num_experts_per_tok=2, sliding_window=32),
    "deepseek_v3": dict(num_key_value_heads=4, moe_intermediate_size=128, n_routed_experts=8, num_experts_per_tok=2,
                        n_shared_experts=1, kv_lora_rank=64, q_lora_rank=128, qk_rope_head_dim=32, v_head_dim=64,
                        qk_nope_head_dim=32, first_k_dense_replace=1, n_group=1, topk_group=1,
                        routed_scaling_factor=1.0, rope_scaling=None),
    "lfm2": dict(norm_eps=1e-5, conv_bias=False, conv_L_cache=3, block_dim=256, block_multiple_of=256,
                 block_ffn_dim_multiplier=1.0, block_auto_adjust_ff_dim=False, full_attn_idxs=[1]),
}



def check(model, ids):
    full = model(mx.array(ids)[None])[0]
    out = {}
    for c in (1, 3, 17):
        cache = make_prompt_cache(model)
        same = 0
        for s in range(0, len(ids), c):
            o = model(mx.array(ids[s:s + c])[None], cache=cache)[0]
            same += sum(int(mx.array_equal(o[i], full[s + i]).item()) for i in range(o.shape[0]))
        out[c] = same
    return out, full.astype(mx.float32)


if __name__ == "__main__":
    ids = np.random.default_rng(0).integers(0, 1024, 70).tolist()
    print(f"positions bit-identical to full prefill, of {len(ids)} (decode / chunks of 3 / chunks of 17)")
    for arch, extra in CFG.items():
        for mode in ("bf16", "q4"):
            try:
                mod = importlib.import_module(f"mlx_lm.models.{arch}")
                mx.random.seed(1)
                model = mod.Model(mod.ModelArgs.from_dict(dict(BASE, model_type=arch, **extra)))
                model.set_dtype(mx.bfloat16)
                if mode == "q4":
                    nn.quantize(model, group_size=64, bits=4,
                                class_predicate=lambda p, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0)
                stock, s_full = check(model, ids)
                mlx_exact.enable()
                try:
                    exact, e_full = check(model, ids)
                finally:
                    mlx_exact.disable()
                dv = float(mx.max(mx.abs(e_full - s_full)).item())
                ok = all(v == len(ids) for v in exact.values())
                print(f"{arch:17s} {mode:4s} stock {list(stock.values())}  mlx-exact {list(exact.values())}  "
                      f"{'bit-exact' if ok else 'not bit-exact'}  (max |exact - stock| logit {dv:.3g})", flush=True)
            except Exception as e:
                mlx_exact.disable()
                print(f"{arch:17s} {mode:4s} skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)
