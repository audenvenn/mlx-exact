"""Does a request's greedy output depend on what else an mlx_lm.server is doing? mlx_lm.server batches concurrent
requests (continuous batching). This starts a server (stock, or with mlx-exact enabled), sends N GSM8K problems via
/v1/completions (temperature 0): first each alone, then all at once, and counts identical outputs.
Usage: python experiments/server_identity.py [--model mlx-community/Qwen3-0.6B-4bit] [--n 16] [--exact] [--port 18093]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--exact", action="store_true", help="start the server with mlx-exact enabled")
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    def once(prompt):
        req = urllib.request.Request(url + "/v1/completions", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": args.model, "prompt": prompt, "temperature": 0.0,
                                                      "max_tokens": args.max_tokens}).encode())
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["text"]

    def post(prompt):
        for k in range(20):                   # simultaneous connects can overflow the server's listen backlog
            try:
                return once(prompt)
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.05 * (k + 1))
        return once(prompt)

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    t = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    prompts = [f"Question: {q}\nAnswer step by step.\n" for q in t.column("question").to_pylist()[:args.n]]

    boot = ("import sys; sys.argv = ['mlx_lm.server'] + sys.argv[1:]; "
            + (f"sys.path.insert(0, {ROOT!r}); import mlx_exact; mlx_exact.enable(); " if args.exact else "")
            + "from mlx_lm.server import main; main()")
    server = subprocess.Popen([sys.executable, "-c", boot, "--model", args.model, "--port", str(args.port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2)
                break
            except Exception:  # noqa: BLE001
                if server.poll() is not None:
                    sys.exit("server exited")
                time.sleep(0.5)
        post("warm up")
        alone = [post(p) for p in prompts]
        together = [None] * len(prompts)
        threads = [threading.Thread(target=lambda i=i: together.__setitem__(i, post(prompts[i])))
                   for i in range(len(prompts))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        server.terminate()
        server.wait(timeout=30)
    same = sum(a == b for a, b in zip(alone, together))
    print(f"{'mlx-exact' if args.exact else 'stock'} server, {args.model}: {same}/{len(prompts)} requests give the same "
          f"output alone and with {len(prompts) - 1} others in flight")


if __name__ == "__main__":
    main()
