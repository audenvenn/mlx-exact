"""Schedule noise on a benchmark: do greedy GSM8K answers change with how the computation is scheduled?
Qwen3-1.7B-4bit, thinking off, greedy, first N test problems (GSM8K is downloaded from Hugging Face at run time).
Schedules: A one-shot prompt prefill; B prefill in chunks of 16 tokens; C batched generation (8 per batch, left
padding). Stock MLX vs mlx-exact. Reports how often the output text and the extracted answer differ between schedules,
and the accuracy of each. About 20 minutes for N = 100 on an M6.
Usage: python experiments/gsm8k_schedule_noise.py [--n 100] [--out summary.json]
"""
import argparse
import json
import re
import time
from pathlib import Path

N = 100
MAX_TOK = 512


def _norm(x):
    v = float(x)
    return int(v) if v == int(v) else v


def number(s):
    """Final answer: the last \\boxed{...}, else the first number after the last '####', else the last number."""
    s = s.replace(",", "").replace("$", "")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", s)
    if boxed:
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed[-1])
        if nums:
            return _norm(nums[0])
    if "####" in s:
        nums = re.findall(r"-?\d+(?:\.\d+)?", s[s.rindex("####"):])
        if nums:
            return _norm(nums[0])
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return _norm(nums[-1]) if nums else None


def main():
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of GSM8K test problems (from the start)")
    ap.add_argument("--out", default="", help="optional JSON file for the summary")
    args = ap.parse_args()
    N = args.n
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx_exact

    t = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    qs, golds = t.column("question").to_pylist()[:N], [number(a) for a in t.column("answer").to_pylist()[:N]]
    model, tok = load("mlx-community/Qwen3-1.7B-4bit")
    eos = set(tok.eos_token_ids)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nSolve the problem step by step and put the "
                                         "final answer after '####'."}], add_generation_prompt=True, enable_thinking=False)
               for q in qs]

    def solo(p, step):
        out = []
        for (tk, _), _ in zip(generate_step(mx.array(p), model, max_tokens=MAX_TOK, prefill_step_size=step), range(MAX_TOK)):
            tk = int(tk)
            if tk in eos:
                break
            out.append(tk)
        return out

    def batched(ps, bs=8):
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

    results = {}
    if True:
        for mode in ("stock", "exact"):
            if mode == "exact":
                mlx_exact.enable()
            try:
                t0 = time.perf_counter()
                A = [solo(p, 2048) for p in prompts]
                B = [solo(p, 16) for p in prompts]
                C = batched(prompts)
                results[mode] = {"A": A, "B": B, "C": C}
                print(f"[{mode}] done in {time.perf_counter() - t0:.0f} s", flush=True)
            finally:
                if mode == "exact":
                    mlx_exact.disable()
    summary = {}
    for mode, r in results.items():
        txt = {k: [tok.decode(v) for v in r[k]] for k in r}
        ans = {k: [number(s) for s in txt[k]] for k in r}
        acc = {k: sum(a == g for a, g in zip(ans[k], golds)) / N for k in r}
        diff = {pair: sum(r[pair[0]][i] != r[pair[1]][i] for i in range(N)) for pair in (("A", "B"), ("A", "C"))}
        adiff = {pair: sum(ans[pair[0]][i] != ans[pair[1]][i] for i in range(N)) for pair in (("A", "B"), ("A", "C"))}
        summary[mode] = {"accuracy": acc, "outputs_differ": {f"{a}-{b}": v for (a, b), v in diff.items()},
                         "answers_differ": {f"{a}-{b}": v for (a, b), v in adiff.items()}}
        print(f"[{mode}] accuracy A {acc['A']:.2f} B {acc['B']:.2f} C {acc['C']:.2f} | outputs differ A-B {diff[('A', 'B')]}/{N}, "
              f"A-C {diff[('A', 'C')]}/{N} | answers differ A-B {adiff[('A', 'B')]}/{N}, A-C {adiff[('A', 'C')]}/{N}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"n": N, "summary": summary}, indent=1))



if __name__ == "__main__":
    main()
