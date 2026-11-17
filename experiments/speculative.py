"""Greedy speculative decoding should reproduce plain greedy decoding exactly: the draft model only proposes tokens and
the target model decides. Does it?
Usage: python experiments/speculative.py [--target mlx-community/Qwen3-1.7B-4bit] [--draft mlx-community/Qwen3-0.6B-4bit]
"""
import argparse

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, speculative_generate_step

import mlx_exact

PROMPTS = [
    "Explain step by step why the sum of the first n odd numbers is n squared.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that returns the longest palindromic substring of a string, with comments.",
    "List the main causes of the French Revolution and briefly explain each one.",
    "What is the difference between a process and a thread in an operating system?",
]


def compare(target, draft, tok, n, k, tag):
    for i, p in enumerate(PROMPTS):
        x = mx.array(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                             enable_thinking=False))
        plain = [int(t) for (t, _), _ in zip(generate_step(x, target, max_tokens=n), range(n))]
        spec = [int(t) for (t, _, _), _ in zip(speculative_generate_step(x, target, draft, num_draft_tokens=k,
                                                                          max_tokens=n), range(n))]
        div = next((j for j, (u, v) in enumerate(zip(plain, spec)) if u != v), None)
        same = div is None and len(plain) == len(spec)
        print(f"[{tag}] prompt {i}: speculative (k={k}) vs plain greedy, {n} tokens: "
              + ("identical" if same else f"diverge at token {div}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--draft", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--k", type=int, default=2, help="draft tokens per step")
    a = ap.parse_args()
    target, tok = load(a.target)
    draft, _ = load(a.draft)
    compare(target, draft, tok, a.tokens, a.k, "stock")
    mlx_exact.enable()
    compare(target, draft, tok, a.tokens, a.k, "mlx-exact")
    mlx_exact.disable()
