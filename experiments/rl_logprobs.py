"""Are RL rollouts on-policy? The log-probabilities of sampled tokens as the sampler computed them (decode with a KV
cache) vs as the trainer computes them (one full forward over prompt + completion, the quantity GRPO/PPO
differentiate), for a LoRA-adapted 4-bit model.
Usage: python experiments/rl_logprobs.py [--model mlx-community/Qwen3-0.6B-4bit]
"""
import argparse

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

import mlx_exact

PROMPTS = ["Explain why the sum of two odd numbers is always even.",
           "Write two sentences about a lighthouse in a storm."]


def logprobs(logits):
    lg = logits.astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def compare(model, prompts, tag, n=64, temp=0.8):
    for i, prompt in enumerate(prompts):
        model.eval()                                           # sampler: decode with a KV cache
        cache = make_prompt_cache(model)
        lg = model(prompt[None], cache=cache)[0, -1]
        key, toks, s_lp = mx.random.key(100 + i), [], []
        for _ in range(n):
            key, sub = mx.random.split(key)
            lp = logprobs(lg)
            t = int(mx.random.categorical(lp / temp, key=sub).item())
            toks.append(t)
            s_lp.append(lp[t])
            lg = model(mx.array([t])[None], cache=cache)[0, -1]
        model.train()                                          # trainer: one full forward
        full = mx.concatenate([prompt, mx.array(toks)])
        t_lp = mx.take_along_axis(logprobs(model(full[None])[0][len(prompt) - 1:-1]), mx.array(toks)[:, None], axis=-1)[:, 0]
        s_lp = mx.stack(s_lp)
        d = mx.abs(s_lp - t_lp)
        print(f"[{tag}] prompt {i}: sampler == trainer log-prob on {int(mx.sum(s_lp == t_lp).item())}/{n} sampled tokens; "
              f"mean |Δ| {float(mx.mean(d).item()):.3g} nats, max |Δ| {float(mx.max(d).item()):.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    a = ap.parse_args()
    model, tok = load(a.model)
    model.freeze()
    mx.random.seed(0)        # before the LoRA layers: MLX seeds its default generator from the clock, and lora_a is random
    linear_to_lora_layers(model, 8, {"rank": 8, "scale": 20.0, "dropout": 0.0,
                                     "keys": ["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"]})
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m.lora_b = mx.random.normal(m.lora_b.shape) * 0.02     # a non-trivial adapter
    prompts = [mx.array(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                                enable_thinking=False)) for p in PROMPTS]
    compare(model, prompts, "stock")
    mlx_exact.enable()
    compare(model, prompts, "mlx-exact")
    mlx_exact.disable()
