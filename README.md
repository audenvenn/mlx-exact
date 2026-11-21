# mlx-exact

**Bit-exact, schedule-invariant inference for [MLX](https://github.com/ml-explore/mlx) / mlx-lm on Apple M5/M6 GPUs.**

In stock MLX, a token's logits depend on how the computation was scheduled. Full prefill, token-by-token decode, the prefill chunk size, the batch layout and speculative verification each select different GPU kernels, and on M5/M6 different hardware units. These kernels round differently. Consequences:
- greedy decoding with a KV cache is not reproducible by recomputation;
- greedy speculative decoding changes the output;
- a request's output depends on which other requests share its batch;
- RL rollouts are not on-policy.

`mlx_exact.enable()` makes every schedule compute the same function, bit for bit. Long prefills keep MLX's neural-accelerator speed.

## Install
Requires an Apple M5/M6 Mac (validated on M6), Python ≥ 3.10, and MLX 0.32.4 exactly, which is installed as a dependency. mlx-lm comes with the `lm` extra; validated with 0.32.2.

```bash
pip install "mlx-exact[lm] @ git+https://github.com/audenvenn/mlx-exact"
```

To run the tests and the reproduction scripts, use a clone instead: `git clone https://github.com/audenvenn/mlx-exact && cd mlx-exact && pip install -e ".[test]"`.

## Usage
```python
import mlx_exact
from mlx_lm import load, generate

mlx_exact.enable()                     # before or after load()
model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit")
print(generate(model, tokenizer, "Hello", max_tokens=50))
print(mlx_exact.stats())               # which kernels stock MLX would have used, and the routes taken
```

## Results (Apple M6, MLX 0.32.4, mlx-lm 0.32.2)

**Why it matters, on a benchmark:** GSM8K, first 300 test problems, greedy decoding. In stock MLX, five common ways of running the same prompts give different final answers:
- the whole prompt prefilled at once;
- the prompt prefilled in chunks of 64 or of 16 tokens;
- batches of 4 or of 8.

| Model | Stock accuracy across the 5 schedules | Problems whose answer changes under some schedule | mlx-exact |
|---|---|---|---|
| Qwen3-0.6B-4bit | 45.7–48.7% | 154 / 300 (51%) | same answers under every schedule |
| Qwen3-0.6B-8bit | 62.3–65.3% | 90 / 300 (30%) | same answers under every schedule |
| Qwen3-1.7B-4bit | 77.7–80.0% | 59 / 300 (20%) | same answers under every schedule |
| Qwen3.5-0.8B-4bit | 41.0–43.7% | 179 / 300 (60%) | same answers under every schedule |

Aggregate accuracy moves by up to 3 points (not significant: the flips go both ways), but per-problem answers depend heavily on the schedule. That noise affects any paired comparison on the same problems. A short write-up of this result: [Same prompt, different answer: MLX's output depends on how the work is scheduled](https://audenvenn.github.io/2026/11/20/mlx-output-depends-on-schedule.html).

With mlx-exact the answers are identical under every schedule. Measured with 0.7.0, the cost was 0.86–0.93× of the stock wall time for single sequences, and 0.35–0.72× for batches of 8. 0.8.0 makes batched exact decode about 2× faster (0.89× of stock in the throughput test below).


| Test | Stock MLX | With mlx-exact |
|---|---|---|
| Qwen3-0.6B-4bit, 360-token prompt: decode / chunks of 4–100 vs full prefill | 0 / 360 positions bit-identical | 360 / 360 |
| Same at 2048 tokens (every 8th position + last 64) | 0 / 312 (except chunks of 512) | 312 / 312 |
| Qwen3.5-0.8B-4bit (hybrid gated delta + head dim 256), same tests | 0 identical | all identical |
| Greedy with KV cache vs recomputing the sequence each step | diverges at token 30 (Qwen3) / 2 (Qwen3.5) | identical |
| Greedy speculative decoding (Qwen3-1.7B + 0.6B draft, k = 2, 4) vs plain, 5 prompts × 200 tokens | 1 / 5 identical | 5 / 5 identical |
| Batched generation (`BatchGenerator`, left padding) vs each prompt alone, 4 prompts | 1 / 4 identical | 4 / 4 identical |
| Batched decode throughput (`BatchGenerator`, Qwen3-0.6B-4bit, 8 sequences) | 1× | 0.89× (0.7.x: 0.35–0.45×) |
| Serving: `mlx_lm.server` (continuous batching), 16 GSM8K requests sent at once vs each alone, Qwen3-0.6B-4bit, 200 tokens | 7–8 / 16 identical (varies with arrival timing) | 16 / 16 identical |
| MoE (mlx-lm `qwen3_moe`, 16 experts, top-4, random weights, 4-bit): decode / chunks of 4, 9, 64 vs full prefill | 0–12 / 200 positions | 200 / 200 |
| MoE greedy with KV cache vs recompute | diverges at token 4 | identical |
| Qwen3-0.6B-8bit: decode / chunks vs full prefill | 0 / 360 positions | 360 / 360 |
| Qwen3.5-0.8B unquantized (bf16): decode / chunks of 4, 9, 64 vs full prefill | 0 / 360 positions | 360 / 360 |
| gpt-oss-20b (MXFP4-Q8: mxfp4 experts, sinks, sliding windows): decode / chunks of 7 vs full prefill, 200 tokens | 0 / 200 positions | 200 / 200 (decode 0.77× of stock) |
| Quantized KV cache (8-bit, Qwen3-0.6B-4bit): decode / chunks of 4, 9, 64 vs full prefill | 0 / 300 positions | 300 / 300 |
| RL with LoRA (Qwen3-0.6B-4bit): sampler vs trainer log-probs of 64 sampled tokens, before and after an optimizer step | 0 / 64 identical (mean \|Δ\| ≈ 0.05 nats/token) | 64 / 64 |
| Decode throughput (end to end, mlx-lm `generate_step`) | 1× | 0.93× (Qwen3-0.6B-4bit, Qwen3-1.7B-4bit); 0.97× (Qwen3-0.6B-8bit); 0.94× (Qwen3.5-0.8B-4bit); 0.98× (Qwen3.5-0.8B bf16) |
| Speculative decoding throughput (Qwen3-1.7B-4bit + 0.6B-4bit draft) | 1× | 0.78× (k = 2), 0.83× (k = 4) |
| Chunked prefill throughput (Qwen3-1.7B-4bit, chunks of 16 / 64 / 128 tokens) | 1× | **1.78× / 1.38× / 1.43×** (faster than stock); one-shot prefill 1.00× |
| Chunked prefill throughput, unquantized (Qwen3.5-0.8B bf16, chunks of 16 / 64 / 128 / 512) | 1× | 0.78× / 0.73× / 0.94× / 0.98× |

## On-policy RL
In stock MLX, the policy that samples (decode with a KV cache) and the policy that is trained (one forward over prompt + completion) compute different functions. Every sampled token's log-probability differs, by about 0.05 nats on average in our test, so on-policy methods silently become off-policy.

With `mlx_exact.enable()` active for both sampling and training, the two are bit-identical. The exact routes are differentiable: they are `mx.custom_function`s whose backward passes use stock MLX ops (gradients need not be bit-exact). LoRA adapters (`mlx_lm.tuner.lora.LoRALinear`) are computed with a fixed-order fp32 kernel.

## How it works
MLX picks the kernel for a 4-bit matmul from the matrix shape:
- `qmv` for one row;
- `qmv_wide` for small batches;
- split-K kernels with 2–16 partitions for medium sizes;
- the accelerator kernel for large sizes.

These differ in two ways. Matrix-matrix kernels round each dequantized weight to bf16/fp16 before multiplying, and vector kernels do not. And each kernel sums in a different order. Attention similarly switches kernels with the number of queries, and its fixed KV blocks make left padding change the summation order.

mlx-exact takes the arithmetic MLX already uses for long prefills as **canonical**:
- **Matmuls**: unquantized (bf16/fp16) and 4-/8-bit quantized layers, MoE experts included. Every call that MLX would send elsewhere is rerouted to a kernel that reproduces it bit for bit. The accelerator's accumulation order (K in chunks of 16, specific fp32 pair/quad sums) was reconstructed with crafted inputs and validated against the hardware.
- **Attention.**
  - For head dims 64, 96 and 128 (bf16/fp16), mlx-exact calls MLX's own fused attention kernel directly (vendored, MIT; `mlx_exact.attn_nax`), for any number of queries.
    - Each sequence's keys are read from its first valid key, so a row of a left-padded batch sees the same keys, in the same 32-key blocks, as the sequence alone.
    - The KV cache is read in place: no query padding, no key copies.
  - For head dims 192/256 with several KV heads (since 0.12.1), mlx-exact composes MLX's unfused attention itself. The two matmuls run on MLX's accelerator kernels, with at least 16 query rows. In between, a softmax of its own sums each row in an order that does not depend on where the keys sit (mod 32) or on how many there are. MLX's own softmax does depend on it (see [docs/numerics.md](docs/numerics.md)).
  - For other head dims, query blocks are padded or split so that the same attention kernel always runs, and batched sequences are re-aligned so that their keys start at index 0.

For calls with 4 or more rows, mlx-exact runs MLX's own accelerator implementation (vendored, MIT; `mlx_exact.nax_kernel`) with a 16-row tile, which is exact by construction and agrees bit for bit with the emulation kernels. mxfp4 layers use MLX's own mxfp4 accelerator implementation the same way, for every row count (`mlx_exact.fp_kernel`).

The full analysis is in [docs/numerics.md](docs/numerics.md): the kernel dispatch, the accelerator's measured accumulation order, the consequences in stock MLX, the method, and the results. The scripts in [`experiments/`](experiments) reproduce the main measurements (stock MLX vs mlx-exact, same run).

## Limitations
- **Validated environment:** MLX 0.32.4 on Apple M6 (GPU architecture generation 18). `enable()` refuses other environments unless `force=True`, because the kernel-dispatch replica is version- and chip-specific.
- **Weights and activations:** unquantized bf16/fp16; 4- and 8-bit affine quantization with group size 64; and since 0.9.0 mxfp4 (MX format, group size 32, e.g. gpt-oss's experts). Activations bf16 or fp16. Other formats (nvfp4, mxfp8) and fp32 activations fall back to stock MLX with a warning, and exactness is then not guaranteed.
- **Not yet supported:** DoRA and LoRA on MoE expert layers (`LoRASwitchLinear`), and attention sinks combined with a quantized KV cache.
- **Attention computed outside mlx-lm's `scaled_dot_product_attention`** is not rerouted:
  - Gemma 2 (`gemma2`) computes attention itself with raw matmuls, for its logit softcapping. A tiny head-dim-64 Gemma 2 is bit-exact in the architecture sweep. At head dim 256 (Gemma 2 2B and 9B), decode and chunks of 3 are not: 32 and 43 of 70 positions match the full prefill; chunks of 17 match at all 70.
  - In mlx-lm 0.32.2, `deepseek_v41`, `plamo2` and `muse_spark` call `mx.fast.scaled_dot_product_attention` directly.
- **Architectures** (tiny random-weight models, bf16 and 4-bit, `experiments/architecture_sweep.py`): decode and chunked prefill are bit-identical to full prefill for llama, qwen2, phi3, olmo2, internlm3, smollm3, hunyuan_v1_dense, seed_oss (untied head), starcoder2, gemma2 (head dim 64 only; see below), granite, glm4, lfm2 and the sliding-window families gemma3_text, cohere2, exaone4 and gpt_oss, as well as qwen3, qwen3_5 and qwen3_moe above. Real gpt-oss: with `mlx-community/gpt-oss-20b-MXFP4-Q8` (mxfp4 experts; attention sinks; 128-token sliding windows), decode and prefill chunks are bit-identical to one full prefill at all 200 positions of a 200-token prompt (stock: 0 / 200), for a single sequence.
- **Sliding-window attention** (rotating KV cache: Gemma 3, Cohere 2, EXAONE 4, gpt-oss, …): bit-exact for a single sequence since 0.6.0. The window is put back in temporal order and aligned to the attention kernel's 32-key blocks, with at most 31 extra masked keys per step.
  - Batched sliding windows are bit-exact since 0.10.0 for head dims 64/96/128 (bf16/fp16) and for head dims 192/256 with one KV head. Attention sinks such as gpt-oss's are covered since 0.11.0.
  - **Head dims 192/256 with several KV heads** (Gemma 3 4B and larger) are bit-exact for sliding windows, alone and batched, since 0.12.1.
    - 0.6.0–0.12.0 were not, once the window had moved past the start of the sequence: MLX's softmax on this path depends on where the window's keys sit modulo 512 or more, and mlx-exact aligned them only modulo 32.
    - With `mlx-community/gemma-3-4b-it-qat-4bit` and a 1,500-token prompt, all of these are bit-identical:
      - one-chunk prefill vs chunks of 512 (109 of 109 checked rows; stock: 1);
      - token-by-token decode after a 1,400-token prefill (100 of 100; 0.12.0 and stock: 0);
      - each sequence of a left-padded batch vs alone (11 of 11 each; 0.12.0: 0 for the padded one);
      - greedy generation with prefill steps of 2,048 and 512 (64 of 64 tokens; stock: 45).
  - In a batch, the shared ring buffer is put back in temporal order, and each sequence is shifted to its absolute positions modulo 32.
  - Not covered:
    - caches with a kept prefix (`keep > 0`, e.g. mlx-lm's `--max-kv-size`);
    - the head-dim case below.
- **Head dims 192/256 with a single KV head** (e.g. Gemma 3 1B): MLX computes attention for these head dims with an unfused path whose internal kernels depend on the query and key counts, and with one KV head the results differ between schedules. Since 0.7.0 mlx-exact uses its own fixed-order attention there (fp32 scores and softmax; accuracy at least stock's). It is bit-exact, but slower: prefill attention for those layers takes about 3× (512 tokens) to 12× (2,048 tokens) as long as MLX's, and decode attention about 2–2.5× (since 0.8.1; 0.7.0–0.8.0: 7× to 36× on the same measurement). Models with two or more KV heads keep MLX's attention.
- **Unquantized MoE expert layers** (`SwitchLinear` → `gather_mm`) are not rerouted; quantized experts are.
- **Boolean attention masks:** the bitwise guarantee covers causal masks with left padding, including mlx-lm's `BatchKVCache`, whose recorded padding mlx-exact uses as each sequence's key origin, and sliding windows on a rotating cache (single sequence). For other boolean masks passed without such a cache, the values are correct, but the attention kernel's key-block origin is derived from the mask and can differ between schedules.
- **Rows with no valid key** (e.g. the padding tokens of a left-padded prefill) get unspecified finite values. Their outputs only feed other padding positions. On mlx-exact's own attention routes they get no gradient. Calls left on stock attention keep stock's gradient, which spreads such a row over all keys.
- **Transforms:** gradients flow through every exact route (`mx.custom_function`). `mx.vmap` over the exact kernels is not supported; batch explicitly. Attention sinks (gpt-oss) on head dims 64/96/128 run on mlx-exact's kernel, whose backward pass includes the sinks (since 0.11.0). With other head dims they run on stock MLX attention, whose gradient MLX 0.32.4 does not support with sinks.
- **Weights applied with raw matmuls** (`x @ w.T` instead of a linear or embedding layer) are not rerouted, so those products stay schedule-dependent. In mlx-lm 0.32.2 this applies to:
  - the MoE router gates of DeepSeek-V2/V3, GLM-4-MoE, dots1, EXAONE-MoE, MiMo-V2-Flash, Nemotron-H and Bailing-MoE-V3;
  - DeepSeek-V3's MLA projections;
  - the tied output layer of Llama 4, MiniCPM and Seed-OSS when `tie_word_embeddings` is set (Seed-OSS with an untied head is bit-exact).
- **DeepSeek-V3-style MLA** (as implemented in mlx-lm): decode uses a different, algebraically equivalent formulation than prefill, and the rotary scores are passed as an additive mask. Such layers run on stock attention: values correct, not bit-exact. Additive (non-boolean) masks in general are passed to stock attention with a warning.
- **MoE expert layers** (`QuantizedSwitchLinear` → `gather_qmm`, 4- and 8-bit) are supported. MLX's own sorted expert kernel equals the exact kernel per expert, bit for bit. On a decode-like step with Qwen3-30B-A3B shapes (8 rows, 128 experts, 4-bit), the exact expert kernel takes 110 µs vs stock's 125 µs (kernel time; no large MoE model was run end to end).
- **Speed:** end-to-end decode runs at 0.93–0.98× of stock on the models above (Apple M6), speculative decoding at 0.78–0.83×, and chunked prefill (chunks of 16–128 tokens) at 1.4–1.8× of stock: for those sizes MLX's own accelerator kernel with a 16-row tile, which mlx-exact runs, is faster than the kernels stock MLX picks. One-shot prefills run at stock speed. A decode step is a chain of dependent matmuls, so each kernel's latency matters more than its throughput. The decode kernels split every output row over many lanes, and keep the canonical summation order with a per-row chain of fp32 additions ([docs/numerics.md](docs/numerics.md), section 4). Unquantized models prefill at 0.73–0.98× of stock (Qwen3.5-0.8B bf16, chunks of 16–512) through MLX's own dense accelerator GEMM loop (vendored) with a 16- or 128-row tile. Remaining slow paths:
  - MoE prefill chunks too small for MLX's sorted expert kernel (they use the per-row expert kernel);
  - decode for head dims 192/256 with several KV heads: Gemma 3 4B (`gemma-3-4b-it-qat-4bit`, 1,500-token context) decodes at 0.75× of stock (42.5 vs 57.0 tok/s), because each attention call pads its query to 16 rows. Its prefill runs at 0.96×.
- **Accuracy:** the canonical arithmetic is that of MLX's prefill. Its per-matmul error is slightly higher than stock decode's because of the bf16 weight rounding, which is negligible next to 4-bit quantization error.

## Tests
`pytest tests/` checks:
- the kernels against MLX's prefill and MoE kernels;
- attention against prefill rows, and left-padding re-alignment;
- MLX's attention kernel with per-sequence key offsets against the padded / re-aligned path; a left-padded batch through `Qwen3-0.6B-4bit` (if cached) against each sequence alone;
- the fixed-order attention kernels against the previous (one thread per output) kernels, bit for bit (`tests/test_fixed_order.py`);
- mxfp4: the dense and expert kernels against stock's long-prefill and sorted expert kernels, and a tiny random gpt-oss with mxfp4 experts end to end (`tests/test_mxfp4.py`);
- batched sliding windows: a left-padded batch through sliding-window models (gpt-oss with sinks, and head dim 256 with one or two KV heads, included), every step's logits against each sequence alone (`tests/test_batch_sliding.py`);
- prompt chunks of 1,024 or more tokens for head dim 256 with several KV heads: whole, chunked and token-by-token prefills of a 1,100-token prompt, and a long left-padded batch against each sequence alone (`tests/test_long_d256.py`);
- attention sinks on the offsets kernel against stock MLX's sink kernel, and their gradient (`tests/test_sinks.py`);
- regression tests from an independent pre-release review (`tests/test_attention_review.py`):
  - unusual masks, layouts and shapes;
  - rows without a valid key, and a left-padded batch through a head-dim-256, single-KV-head model;
  - gradients through every attention route against stock attention's;
- the dense kernel against MLX's canonical GEMM;
- decode == prefill end to end on small random-weight MoE models, 4- and 8-bit (no download);
- with LoRA, sampler log-probs == trainer log-probs, and gradients flow, on a small random-weight model (no download);
- decode == prefill on `mlx-community/Qwen3-0.6B-4bit`, if it is in the local Hugging Face cache.
- the quantized KV cache: invariance, accuracy against an fp64 reference, and decode == prefill on a small random-weight model;
- unsupported activation dtypes fall back to stock MLX with a warning.
- every quantized route (1, 2–4 and more rows; 4- and 8-bit) against MLX's canonical kernel, and decode plus chunk sizes 2–64 against full prefill on small random-weight models.

## Related work
- **Batch-invariant kernels.** Thinking Machines' [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025) traced output differences in served models to kernels whose reduction order depends on the batch. It fixed them with batch-invariant kernels ([batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)). The same idea is behind [SGLang's deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) and [vLLM's batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) on CUDA.
- **On MLX.** Two projects bring batch-invariant kernels to MLX:
  - [mlx-batch-invariant](https://github.com/waseemh/mlx-batch-invariant) (2026): batch-invariant matmul, 4-/8-bit quantized matmul, RMSNorm and attention. Its attention gives the same bits in prefill and decode, and chunked prefill matches one-shot prefill. Its README reports identical logits at batch sizes 1–64 for Qwen2.5, SmolLM2 and GPT-2 models on an Apple M4.
  - [mlx-deterministic](https://github.com/ProbioticFarmer/mlx-deterministic) (2025): batch-invariant RMSNorm, matmul, softmax and attention operations, with custom Metal kernels for bitwise results.
- **Scheduling and specification (CUDA).**
  - LLM-42 ([arXiv 2601.17768](https://arxiv.org/abs/2601.17768)) and CoRun ([arXiv 2608.14376](https://arxiv.org/abs/2608.14376)) get determinism from scheduling instead of new kernels: verify-and-rollback, or isolated prefill with fixed-shape decode.
  - Vosti ([arXiv 2609.38981](https://arxiv.org/abs/2609.38981)) specifies deterministic inference as bitwise-identical logits across batch composition, chunking, prefill/decode execution and KV-cache reuse, and verifies an engine against it. mlx-exact targets the same property on MLX.
- **What mlx-exact does differently.**
  - The batch-invariant projects define one new reduction order and use it for every batch size. mlx-exact instead keeps the arithmetic MLX already uses for long prefills (see *How it works*) and reproduces it in every other schedule.
    - Long prefills therefore keep stock MLX's speed and, for most layers, its results on the M5/M6 neural accelerators.
    - The exceptions are head dims 192/256. With one KV head they use mlx-exact's own fixed-order attention. With several KV heads they use MLX's unfused composition with mlx-exact's own softmax for every query count, where stock MLX uses a vector kernel for decode and a fused kernel from 1,024 queries.
  - It is built for mlx-lm's batched serving: left-padded batches with array masks, `BatchGenerator` and `mlx_lm.server`.
  - It also covers attention sinks, mxfp4 weights, sliding windows, speculative decoding, quantized KV caches, MoE experts and gradients for on-policy RL.

## Disclosure
This code and the underlying analysis were developed with AI assistance (Anthropic's Claude) and verified by the tests and measurements described above. 0.8.0 also had an independent pre-release code review by another AI model (OpenAI's Codex); each reported bug was confirmed with a regression test and fixed, and the remaining points (limits of scope) are documented above.

## Changes
- **0.12.1:** head dims 192/256 with several KV heads (Gemma 3 4B and larger) are bit-exact with sliding windows, decode and batches.
  - On real Gemma 3 4B, 0.12.0 (and every version since 0.6.0) was not bit-exact for decode once the window had slid, or for padded sequences in batches. The tiny-model tests had passed.
  - Cause: MLX's softmax on the unfused attention path sums each row in a tree over 128-key groups. Its bits depend on where the keys sit modulo 512 or more, while mlx-exact aligned them modulo 32. The differences are rare (a few hundred in 26 million outputs), but each one changes every later token's logits.
  - mlx-exact now composes this path itself: MLX's accelerator matmuls (at least 16 query rows), with a softmax in between whose sum depends only on the keys' positions modulo 32.
  - With `mlx-community/gemma-3-4b-it-qat-4bit` and a 1,500-token prompt, chunked prefill, token-by-token decode, padded batches and greedy generation all match the full prefill bit for bit. Speed: prefill 0.96×, decode 0.75× of stock.
  - **New tests** (`tests/test_long_d256.py`):
    - the softmax's invariance;
    - a window's decode, shifted and prefill layouts at real sizes (window 1024, head dim 256, 8 / 4 heads): bit-identical, where stock MLX's attention gives 54 mismatching outputs in the same test.
- **0.12.0:** head dims 192/256 with several KV heads (Gemma 3 4B and larger).
  - **Fix (since 0.1.0):** in exact mode, prompt chunks of 1,024 or more tokens with an array mask raised NotImplementedError.
    - This hit a single prompt over 1,024 tokens on a sliding-window layer (mlx-lm prefills in 2,048-token steps), and long left-padded batches.
    - Such chunks are now split into blocks of at most 1,008 rows on MLX's unfused attention path, which decode and shorter chunks use.
  - **Fix (since 0.6.0):** when a causal chunk of 1,024 or more tokens on a sliding-window layer was split, each block re-derived its key alignment from the cache and could shift its keys by up to 31 positions. Values were correct, but the bits could differ from other schedules.
  - **Batched sliding windows** for these head dims: each sequence is placed at its absolute positions modulo 32, as on the offsets route. **Correction:** this is not enough on MLX's unfused attention, and real Gemma 3 4B decode and padded batches are not bit-exact (see *Limitations*). The tiny-model tests pass.
  - **New tests:**
    - `tests/test_long_d256.py`: a 1,100-token prompt prefilled whole, in chunks of 300, or token by token gives bit-identical logits; a left-padded batch of 1,100- and 1,040-token prompts matches each sequence alone.
    - A Gemma 3 4B-like case (head dim 256, two KV heads) in `tests/test_batch_sliding.py`, which fails with 0.11.0.
  - **README:** a related-work section (batch-invariant kernels on CUDA and MLX; LLM-42, CoRun, Vosti).
- **0.11.0:** attention sinks (gpt-oss) run on the exact offsets kernel for head dims 64/96/128. The kernel uses its own sink path, one logit per query head, so query heads are not packed into one tile.
  - Batched and served gpt-oss (batched sliding windows with sinks) is bit-exact.
  - Results are bit-identical to stock MLX's sink kernel (`tests/test_sinks.py`). A tiny gpt-oss with mxfp4 experts gives every step's logits bit-identical to each sequence alone in a left-padded batch (`tests/test_batch_sliding.py`; with 0.10.0 it did not).
  - The backward pass includes the sinks and gives them a gradient.
- **0.10.0:** batched generation with sliding windows (mlx-lm `BatchRotatingKVCache`, used by `BatchGenerator` and `mlx_lm.server` for Gemma 3, Cohere 2, EXAONE 4 and similar) is bit-exact for head dims 64/96/128 and for head dims 192/256 with one KV head.
  - Every sequence's window is put back in temporal order and shifted to its absolute positions modulo 32. The shift comes from the per-sequence offsets that RoPE also uses, so each sequence sees the 32-key blocks of its own full prefill.
  - New tests (`tests/test_batch_sliding.py`): a left-padded batch through tiny Gemma 3 / Cohere 2 / EXAONE 4 models and a head-dim-256, one-KV-head Gemma 3, prefilled whole or in chunks of 16, then 40 decode steps. Every step's logits are bit-identical to each sequence alone; with 0.9.0 they were not.
- **0.9.0:** mxfp4 (MX-format) layers are exact, e.g. the experts of gpt-oss.
  - mlx-exact runs MLX's own mxfp4 accelerator implementation (vendored) with a 16-row tile for any row count, plus an expert-indexed variant for MoE layers.
  - Both are bit-identical to stock's long-prefill and sorted expert kernels.
  - With `mlx-community/gpt-oss-20b-MXFP4-Q8`, decode and chunked prefill match one full prefill at every position of a 200-token prompt (stock: none), at 0.77× of stock decode speed.
  - New tests: `tests/test_mxfp4.py`, including a tiny random gpt-oss with mxfp4 experts.
- **0.8.1:** faster fixed-order attention for head dims 192/256 with one KV head (e.g. Gemma 3 1B / 270M). Prefill attention is 2.4–3× faster: 0.95 vs 2.32 ms at 512 tokens and 11.8 vs 35.4 ms at 2,048 tokens, against stock MLX's 0.32 / 0.97 ms.
  - Every output is still the same sequential fp32 chain. Each thread now computes a 4 × 4 block of outputs from tiles staged in threadgroup memory, and one SIMD group handles each softmax row; its sum stays sequential.
  - Results are bit-identical to 0.8.0, checked against the previous kernels in `tests/test_fixed_order.py`.
- **0.8.0:** exact attention for head dims 64, 96 and 128 (bf16/fp16) calls MLX's own fused attention kernel directly, with per-sequence key offsets: no query padding, no key copies, and the KV cache is read in place.
  - Batched exact decode is 2× faster: Qwen3-0.6B-4bit, 8 sequences, 1,125 generated tok/s vs 566 in 0.7.1 (stock 1,260; now 0.89× of stock). One attention call (one layer of a batched decode step, 300-key cache) takes 41 µs, vs 405 µs in 0.7.1; stock's own, non-exact kernel takes 81 µs. Single-sequence decode: 0.95× of stock (Qwen3-0.6B-4bit, 318 vs 336 tok/s; 0.93× in 0.7.x).
  - Results are unchanged: bit-identical to 0.7.1 in 828 kernel checks and end to end.
  - Fixes, found in an independent pre-release review and confirmed with new regression tests (`tests/test_attention_review.py`):
    - **NaN in left-padded batches** (since 0.7.0) for head dims 192/256 with one KV head (e.g. Gemma 3 1B). The fixed-order attention gave NaN for rows with no valid key, i.e. the padding tokens. In the next layer, those positions' values were NaN, and 0 × NaN reached every real row. Such rows now give zeros.
    - **Backward passes through exact attention.** Since 0.7.0 the fixed-order attention had none, in 0.7.1 the key re-alignment for batched or sliding-window masks had none, and the quantized-KV attention never had one. Every attention route now has a gradient (for the queries only with a quantized cache), checked against stock attention's.
    - **Unrestricted attention** (`mask=None` with several queries) was made causal on the padded fallback path. That path is used for head dims other than 64/96/128, for fp32 and for attention sinks.
    - **Unusual shapes:** with more queries than keys, the output was wrong or empty; such calls now run on stock attention. Masks broadcast along the key axis failed with a rotating cache.
- **0.7.1:** batched exact decode re-aligns each sequence's keys with a plain shifted-copy kernel instead of index gathers: about 27% faster (Qwen3-0.6B-4bit, 8 sequences: 445 → 566 generated tok/s; stock 1,260). Results are unchanged.
- **0.7.0:** fixed-order attention for head dims 192/256 with a single KV head (bit-exact where MLX's own attention is not). Exact decode with a quantized KV cache no longer compiles a new kernel at every step: 5.0 ms per step vs stock's 3.9 ms on Qwen3-0.6B-4bit with an 8-bit cache (0.3.0–0.6.0: about 50 ms).
- **0.6.0:** sliding-window attention is bit-exact for single sequences (rotating KV caches: temporal order plus 32-key block alignment). The architecture sweep now passes for 16 mlx-lm architectures in bf16 and 4-bit; documented limits: single-KV-head models at head dim 192/256, unquantized MoE experts, batched sliding windows.
- **0.5.3:** fixes wrong attention values in exact mode for sliding-window masks and additive masks (present since 0.1.0). Re-aligning keys for batched left padding shifted every query row by the last row's first valid key, which drops valid keys of earlier rows under a sliding window. The shift is now the earliest valid key over all rows and heads, which never changes a row's values, and additive masks are no longer shifted. Causal and left-padding masks, and therefore the results above, were not affected.

## License
MIT (see `LICENSE`). `mlx_exact/_vendor/mlx_v0_32_4/` contains unmodified headers from MLX v0.32.4, © Apple Inc., MIT (see `NOTICE`).
