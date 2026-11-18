"""Schedule noise on a benchmark: do greedy GSM8K answers change with how the computation is scheduled?
GSM8K test (first N problems; downloaded from Hugging Face at run time), greedy, thinking off, 512-token cap.
Stock schedules: S1 whole prompt in one prefill, S2 prefill chunks of 64, S3 chunks of 16, S4 batches of 4, S5 batches
of 8 (mlx-lm BatchGenerator, left padding). mlx-exact: S1, S3, S5. Prints accuracy per schedule, answer flips vs S1,
the number of problems whose answer changes under some stock schedule, and whether the mlx-exact schedules agree.
About 45 minutes for N = 300 with Qwen3-1.7B-4bit on an M6.
Usage: python experiments/gsm8k_schedule_noise.py [--model mlx-community/Qwen3-1.7B-4bit] [--n 300] [--out summary.json]
"""
import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from mlx_lm import load
from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.sample_utils import make_sampler

import mlx_exact

MAX_TOK = 512


def _norm(x):
    v = float(x)
    return int(v) if v == int(v) else v


def number(s):
    """Final answer: the last \\\\boxed{...}, else the first number after the last '####', else the last number."""
    s = s.replace(",", "").replace("$", "")
    boxed = re.findall(r"\\\\boxed\\{([^}]*)\\}", s)
    if boxed:
        nums = re.findall(r"-?\\d+(?:\\.\\d+)?", boxed[-1])
        if nums:
            return _norm(nums[0])
    if "####" in s:
        nums = re.findall(r"-?\\d+(?:\\.\\d+)?", s[s.rindex("####"):])
        if nums:
            return _norm(nums[0])
    nums = re.findall(r"-?\\d+(?:\\.\\d+)?", s)
    return _norm(nums[-1]) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    qs = t.column("question").to_pylist()[:args.n]
    golds = [number(a) for a in t.column("answer").to_pylist()[:args.n]]
    model, tok = load(args.model)
    eos = set(tok.eos_token_ids)
    prompts = []
    for q in qs:
        msgs = [{"role": "user", "content": q + "\nSolve the problem step by step and put the final answer after '####'."}]
        try:
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True))

    def solo(p, step):
        out = []
        for (tk, _), _ in zip(generate_step(mx.array(p), model, max_tokens=MAX_TOK, prefill_step_size=step), range(MAX_TOK)):
            if int(tk) in eos:
                break
            out.append(int(tk))
        return out

    def batched(ps, bs):
        res = []
        for i in range(0, len(ps), bs):
            chunk = ps[i:i + bs]
            gen = BatchGenerator(model, stop_tokens=[[e] for e in eos], sampler=make_sampler(temp=0.0),
                                 completion_batch_size=len(chunk), prefill_batch_size=len(chunk), max_tokens=MAX_TOK)
            uids = gen.insert([list(p) for p in chunk], [MAX_TOK] * len(chunk))
            out = {u: [] for u in uids}
            while responses := gen.next_generated():
                for r in responses:
                    if r.finish_reason != "stop" and int(r.token) not in eos:
                        out[r.uid].append(int(r.token))
            gen.close()
            res += [out[u][:MAX_TOK] for u in uids]
        return res

    run = {"S1": lambda: [solo(p, 2048) for p in prompts], "S2": lambda: [solo(p, 64) for p in prompts],
           "S3": lambda: [solo(p, 16) for p in prompts], "S4": lambda: batched(prompts, 4), "S5": lambda: batched(prompts, 8)}
    ans, secs = {}, {}
    for mode, scheds in (("stock", ("S1", "S2", "S3", "S4", "S5")), ("mlx-exact", ("S1", "S3", "S5"))):
        if mode == "mlx-exact":
            mlx_exact.enable()
        try:
            for s in scheds:
                t0 = time.perf_counter()
                ans[(mode, s)] = [number(tok.decode(x)) for x in run[s]()]
                secs[(mode, s)] = time.perf_counter() - t0
                acc = sum(a == g for a, g in zip(ans[(mode, s)], golds)) / len(golds)
                print(f"[{mode}] {s}: accuracy {acc:.3f} ({secs[(mode, s)]:.0f} s)", flush=True)
        finally:
            if mode == "mlx-exact":
                mlx_exact.disable()
    n = len(golds)
    flips = {s: sum(a != b for a, b in zip(ans[("stock", "S1")], ans[("stock", s)])) for s in ("S2", "S3", "S4", "S5")}
    unstable = sum(1 for i in range(n) if len({ans[("stock", s)][i] for s in ("S1", "S2", "S3", "S4", "S5")}) > 1)
    exact_flips = sum(1 for i in range(n) if len({ans[("mlx-exact", s)][i] for s in ("S1", "S3", "S5")}) > 1)
    print(f"stock: answers changed vs S1 {flips}; problems whose answer changes under some schedule: {unstable}/{n}")
    print(f"mlx-exact: problems whose answer changes between S1, S3, S5: {exact_flips}/{n}")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "n": n, "flips_vs_S1": flips, "unstable": unstable,
                                              "exact_unstable": exact_flips,
                                              "seconds": {f"{m}-{s}": v for (m, s), v in secs.items()}}, indent=1))


if __name__ == "__main__":
    main()
