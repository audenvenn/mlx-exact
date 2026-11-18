# How mlx-exact works: the numerics of MLX on Apple M5/M6

This note documents why stock MLX produces schedule-dependent results, what the GPU kernels actually compute, and how mlx-exact makes every schedule compute the same function. All measurements are on an Apple M6 (GPU architecture generation 18) with MLX 0.32.4 and mlx-lm 0.32.2. The experiments in [`experiments/`](../experiments) reproduce the central claims.

## 1. One operation, many kernels
MLX picks a kernel for each operation from its shapes. For the matmuls of a language model this happens on every call, and the choice depends on how many rows (tokens) are processed together.

**4-bit and 8-bit matmul** (`x @ dequant(W).T`, the layout of `nn.QuantizedLinear`; MLX 0.32.4, `mlx/backend/metal/quantized.cpp`):

| Rows M | Kernel | Runs on |
|---|---|---|
| 1 | `qmv` | GPU cores (SIMD) |
| 2 … limit − 1 | `qmv_wide` | GPU cores (SIMD) |
| ≥ limit, while split_k > 1 | `qmm_t_splitk` (K split into 2–16 partitions) | GPU cores (simdgroup matrix) |
| ≥ limit, split_k = 1 | `qmm_t_nax` (32-row tiles up to 32 rows, else 64-row tiles) | neural accelerators |

- **limit** (on this GPU class) is 33 when K, N ≤ 2048; 25 when K, N ≤ 4096; 13 otherwise.
- **split_k** = max(1, 512 / (⌈N/32⌉·⌈M/32⌉)) (integer division), capped at K/64 and reduced until split_k·64 divides K.

So, depending on prompt length, chunk size or batch size, one layer can run `qmv`, `qmv_wide`, split-K with several partition counts, or either accelerator tile size. For example, in Qwen3-0.6B the k/v/o/down projections use split-K up to 256 rows and the accelerator only beyond.

- **Dense bf16/fp16 matmul** (`matmul.cpp`): `gemv` (1 row), `gemv_wide` (2–14 rows), a thin accelerator kernel (≤ 64 rows or ≤ 64 columns), split-K accelerator (K ≥ 3·max(M, N), or max(M, N) ≤ 1024 and K > 2·max(M, N)), otherwise the regular accelerator GEMM.
- **MoE experts** (`gather_qmm`, as mlx-lm calls it): `gather_qmv` per (token, expert) pair, or `gather_qmm_rhs` when the pairs are sorted by expert (≥ 16 pairs, ≥ 4 per expert on average).
- **Attention** (`mx.fast.scaled_dot_product_attention`):
  - ≤ 8 queries use vector kernels. With grouped-query attention and ≥ 4096 keys (≥ 1024 keys on any model for Max/Ultra-class GPUs), they use a 2-pass variant whose per-block partial outputs are stored in the activation type (bf16).
  - ≥ 9 queries use the accelerator kernel.
  - For head dims 192/256, MLX instead runs an unfused path (separate matmuls) for 9–1023 queries and a fused head-dim-split kernel from 1024.

## 2. The kernels compute different arithmetic
Different kernels are not only different summation orders.

- **Weight rounding.** Matrix-matrix kernels (`qmm`, split-K, accelerator) dequantize each weight as `s·q + b` in float and **round it to the activation type** (bf16/fp16) in threadgroup memory before multiplying. The vector kernels (`qmv`) never round weights; they compute `s·Σ(q·x) + b·Σx` in float.
  - For random bf16 data, the two disagree in about 40% of output elements.
  - The prefill path is slightly less accurate: relative RMS error 1.9–2.2e-3 vs 1.66e-3 for decode on real Qwen3-1.7B activations, against an fp64 reference.
- **The accelerator's accumulation order.** On the M6, the matrix operation that MLX's accelerator kernels use (`mpp::tensor_ops::matmul2d`, descriptor 16×32×16, relaxed precision) has a fixed internal order, determined with crafted inputs:
  - **Probe:** a dot product holds +2²⁴ at position p, −2²⁴ at q and a single 1 at m, so the exact result is 1. In fp32, 2²⁴ + 1 rounds to 2²⁴ while −2²⁴ + 1 is exact, so the 1 is lost exactly when the summation combines it with +2²⁴ before −2²⁴ has joined either of them. Scanning m over all positions, for several (p, q) and also with 2³⁰ (where −2³⁰ + 1 is inexact too), reveals the order.
  - **Result:** K is processed in sequential chunks of 16. Within a chunk, products form 4 quads at offsets {2r, 2r+1, 2r+8, 2r+9}. Each quad is summed in fp32 as (p[2r] + p[2r+1]) + (p[2r+8] + p[2r+9]), and quads are added to one fp32 accumulator in order r = 0..3.
  - **Validation:** on random data (4-bit weights, K = 4096, fp16, 512 rows; bf16 output is too coarse to tell orders apart), a scalar emulation of this order reproduces the hardware in 2048 / 2048 outputs. An exact dot product misses 8, and other pairings of the four products miss 1–3. MLX's own accelerator implementation, run with a 16-row tile, agrees with it on 7 matrix shapes.
  - The 32-row-tile instantiation of the same kernel follows a different order (no candidate matched), so it is not canonical.
- **The vector kernel's order** (`gemv`): each of 32 lanes sums 4 consecutive elements per 128-element block in order, then a butterfly `simd_sum` (xor 16, 8, 4, 2, 1).
- **Attention.**
  - The vector kernels use `fast::exp`; the accelerator kernel uses `fast::exp2` on log2-scaled scores.
  - Their reductions differ.
  - The accelerator kernel's KV blocks and 16-key fragments are aligned to key index 0. So **left padding in a batch shifts a sequence's keys across block boundaries** and changes its result (37–144 of 2048 outputs for a 59-token sequence).
- **fp32 matmuls** with 2 or more rows run on the accelerator at reduced, TF32-like precision by default (`MLX_ENABLE_TF32=1`; MLX's precision guide documents it, and `matmul.cpp` / `quantized.cpp` gate the accelerator path for fp32 on it).
  - Measured max relative error vs fp64: 4e-4 – 9e-4 for 2–300 rows, against 3e-7 – 7e-7 for one row (`gemv`).
  - So in stock MLX an fp32 LoRA adapter is computed at full precision during decode and at reduced precision during prefill and training.

## 3. Consequences in stock MLX
| Test | Result |
|---|---|
| Qwen3-0.6B-4bit, 360-token prompt: logits from decode or chunked prefill (4–100) vs full prefill | 0 / 360 positions bit-identical; max \|Δlogit\| up to 2.0 |
| Greedy decoding with KV cache vs recomputing the sequence each step | diverges at generated token 30 (Qwen3-0.6B-4bit) and token 2 (Qwen3.5-0.8B-4bit) |
| Greedy speculative decoding (Qwen3-1.7B-4bit + 0.6B-4bit draft, 200 tokens) vs plain greedy | output changes on 4 / 5 prompts; first change at tokens 1–82 |
| Batched generation (mlx-lm `BatchGenerator`, left padding) vs each prompt alone | 3 / 4 prompts change |
| RL with LoRA (Qwen3-0.6B-4bit, random rank-8 adapter): sampler (decode) vs trainer (full forward) log-probs of 64 sampled tokens | 0 / 64 identical for each prompt; mean \|Δ\| 0.04–0.06 nats, max up to 0.26 (depends on the adapter) |
| GSM8K (100 problems, Qwen3-1.7B-4bit, greedy): one-shot prefill vs prefill in chunks of 16 vs batches of 8 | 67–71% of solutions and 14–18% of final answers change; accuracy 75% / 80% / 75% (McNemar p = 0.18: noise) |

## 4. Method: one canonical arithmetic
mlx-exact takes the arithmetic MLX uses for **long prefills** as canonical, so long prefills keep full accelerator speed, and reroutes every call that would leave it.

- **Matmuls.** A Python replica of MLX's dispatch decides per call whether stock would run the canonical kernel (`qmm_t_nax` with 64-row tiles; the regular accelerator GEMM; `gather_qmm_rhs`). Otherwise the call runs an exact kernel:
  - **v0:** one thread per output, a scalar emulation of the accelerator order, with weights rounded exactly as the loader rounds them;
  - **v6/v6b:** coalesced 32-row tiles with prefetch, the same arithmetic;
  - **v7** (decode, 4- and 8-bit, and unquantized): each output row split over many lanes, see *Speed* below;
  - **v7_rows** (2–4 rows, e.g. speculative verification): v7 that dequantizes each weight once for all rows;
  - **v4** (4 or more rows): MLX's own accelerator implementation (vendored) run with a 16-row tile, exact by construction;
  - **dense v4** (unquantized, 5 or more rows): MLX's own dense accelerator GEMM loop (`gemm_loop`, vendored) with a 16-row tile, or with stock's own 128-row tile above 64 rows. The K order of an output element does not depend on the tile height, so this is the canonical arithmetic for every row count;
  - expert-indexed variants for MoE (v0 and v7 designs; each row reads its own expert's weights), and dense variants without dequantization.
- **Attention.**
  - Query blocks below the accelerator threshold are padded to 9 queries, or 16 for head dims 192/256. The real queries go last under a causal, bottom-aligned mask. If the context is shorter than the padding, a square problem with zero keys (always masked) is used instead.
  - Blocks of ≥ 1024 queries for head dims 192/256 are split below MLX's fused-kernel threshold.
  - With a batch mask, each sequence's keys are first re-aligned to start at index 0.
  - With a sliding window (rotating KV cache), the window is put in temporal order and r = (absolute position of its first key) mod 32 masked zero keys are prepended, so each key sits at the same position within the kernel's 32-key blocks as in a full prefill.
  - Masked keys contribute exact zeros, so this reproduces the canonical rows bit for bit.
- **Head dims 192/256 with a single KV head.** MLX's unfused attention for these head dims changes its internal matmul kernels with the query and key counts, so it is not schedule-invariant there. mlx-exact uses its own fixed-order attention instead: fp32 scores, a sequential per-row softmax, and a sequential fp32 output sum. Masked keys contribute exact zeros, so the result depends only on the order of the valid keys.
- **Paths that mlx-lm composes from several MLX ops** (quantized KV-cache attention, LoRA adapters) have no single MLX kernel to take as canonical, so mlx-exact defines the arithmetic itself: fixed-order fp32 kernels that are identical for every query or row count. Checked against fp64, the quantized-KV attention is as accurate as stock decode, and the LoRA kernel is accurate to fp32 rounding.
- **Gradients.** Every exact route is an `mx.custom_function` whose backward pass uses stock MLX ops. Gradients need not be bit-exact; only the forward must match, e.g. the sampler's and the trainer's log-probabilities.
- **Speed.** A decode step is a chain of dependent matmuls, so a kernel's *latency* matters more than its throughput.
  - With one thread per output row, every lane runs a K-long serial loop, and a 1024-output layer occupies only 32 SIMD groups. In a dependent chain this was 2.3–3.4× slower than stock per call on 0.6B-sized layers, although back-to-back independent calls were only 1.2–1.5× slower (the GPU overlaps them).
  - **v7** keeps the canonical order but splits each row. Per threadgroup of 32 rows and per 256-weight K tile, 8 producer SIMD groups (lane = 16-weight chunk) compute every chunk's four quad sums into a double buffer in threadgroup memory. Meanwhile a chain SIMD group (lane = row) adds the previous tile's quad sums into each row's fp32 accumulator in canonical order, with one barrier per tile.
  - The pair sums use `fma(a, b, c·d)`. Products of bf16/fp16 values are exact in fp32, so this equals the canonical `round(a·b + c·d)` bit for bit.
  - Register prefetch and integer bf16 rounding were tried and did not help.
  - **Several rows** (speculative verification, small batches, chunked prefill). Per-row kernels repeat the dequantization for every row, so their cost grows linearly with the row count.
    - v7_rows shares it across 2–4 rows, keeping the double buffer of quad sums small: 8 output rows per threadgroup.
    - From 4 rows, v4 is best. MLX's accelerator kernel with a 16-row tile costs almost the same for 1–16 rows. For 16–128 rows it is faster than the `qmv_wide` and split-K kernels that stock MLX itself picks, so exact chunked prefill is 1.4–1.8× faster than stock.

## 5. Results with mlx-exact
| Test | mlx-exact |
|---|---|
| decode / chunked prefill (4–512) vs full prefill: Qwen3-0.6B 4-bit and 8-bit; Qwen3.5-0.8B hybrid 4-bit and unquantized bf16; random MoE (4-bit); up to 2048 tokens | every position bit-identical |
| greedy with cache vs recompute; greedy speculative vs plain (5 / 5); batched vs alone (4 / 4) | identical |
| quantized KV cache (8-bit) | 300 / 300 positions identical |
| RL with LoRA: sampler vs trainer log-probs, before and after an optimizer step | 64 / 64 identical |
| decode throughput (mlx-lm `generate_step`) | 0.93× of stock (Qwen3-0.6B-4bit: 306 vs 330 tok/s; Qwen3-1.7B-4bit: 126 vs 136), 0.97× (Qwen3-0.6B-8bit), 0.94× (Qwen3.5-0.8B-4bit), 0.98× (Qwen3.5-0.8B bf16). Before v7: 0.68–0.79× |
| speculative decoding (Qwen3-1.7B-4bit + 0.6B-4bit draft) | 0.78× of stock (k = 2), 0.83× (k = 4); before multi-row kernels 0.63–0.70× |
| chunked prefill (Qwen3-1.7B-4bit, 512 tokens) | 1.78× / 1.38× / 1.43× of stock for chunks of 16 / 64 / 128; 1.00× in one chunk |
| MoE decode-like step (8 rows, 128 experts, 4-bit, Qwen3-30B-A3B shapes; kernel time) | 110 µs vs stock 125 µs (previous exact kernel 136 µs) |
| chunked prefill, unquantized (Qwen3.5-0.8B bf16, 512 tokens) | 0.78× / 0.73× / 0.94× / 0.98× of stock for chunks of 16 / 64 / 128 / 512 (before dense v4: 0.13–0.44×) |

## 6. Pitfalls found along the way
- mlx-lm's `quantized_scaled_dot_product_attention` scales its `queries` argument with `*=`, which updates the caller's array object in place. Inside a model this is harmless, but a test that reuses the same query array across calls silently feeds pre-scaled queries to later calls (it cost us a false alarm).
- `mx.fast.metal_kernel` passes very small inputs in Metal's `constant` address space, so kernels should declare input-derived pointers with `auto`.
- `mx.fast.metal_kernel` appends the generated `template <...>` line directly after the `header` string, so a header whose last line is a `//` comment must end with a newline, or the comment swallows the template line.
- Every distinct value of a `metal_kernel` template parameter compiles a new kernel variant (about 0.2 s for a large header). Sizes that vary from call to call, such as the row count of a chunk or batch, should be runtime inputs.
- An attention mask is not necessarily left padding. Re-aligning keys must shift by the earliest valid key over all query rows, which preserves every row's values; shifting by the last row's first valid key silently corrupts sliding-window layers (a bug in mlx-exact 0.1.0–0.5.2, found with the architecture sweep).
- Agreement between two schedules shows only consistency. Every self-defined route must also be checked against an independent reference (here fp64 and MLX's own canonical kernels).
- MLX seeds its default random generator from the clock (`mlx/random.h`), so anything random created before `mx.random.seed(...)`, such as LoRA's `lora_a` in `linear_to_lora_layers`, differs between runs. Seed first if the numbers should repeat.

## 7. Reproducing
From a clone, `pip install -e ".[lm]"`, then run the scripts in `experiments/`. Each runs the same workload in stock MLX and with mlx-exact in one process. The models (`mlx-community/Qwen3-0.6B-4bit`, `Qwen3-1.7B-4bit`) are downloaded from Hugging Face on first use. Output on an Apple M6, MLX 0.32.4, mlx-lm 0.32.2:

**`order_scan.py`** (under a second): the survival patterns for `gemv` and the accelerator GEMM (e.g. with −B at 127, only positions 118, 119 and 126 survive on the accelerator, the last quad), then:
```
Accelerator kernel vs reconstructed order: 0 / 2048 outputs differ (an exact dot product: 5 / 2048).
```

**`schedule_invariance.py`** (Qwen3-0.6B-4bit, 360-token prompt; 13 s):
```
[stock] decode (1 token per step)  vs full prefill: 0/360 positions bit-identical, max |Δlogit| 1.72
[stock] chunks of 4 / 9 / 16 / 64 / 100: 0/360 each, max |Δlogit| 1.56 / 1.33 / 1.78 / 2.03 / 1.62
[stock] greedy, KV cache vs recomputing the sequence (64 tokens): diverge at generated token 30
[mlx-exact] decode and every chunk size: 360/360 positions bit-identical, max |Δlogit| 0
[mlx-exact] greedy, KV cache vs recomputing the sequence (64 tokens): identical
```

**`speculative.py`** (Qwen3-1.7B-4bit target, Qwen3-0.6B-4bit draft, 5 prompts, 200 tokens; about 40 s; the same with `--k 4`):
```
[stock] prompts 0-4: diverge at tokens 1, 41, 82, (identical), 80
[mlx-exact] prompts 0-4: identical
```

**`rl_logprobs.py`** (Qwen3-0.6B-4bit with a random rank-8 LoRA adapter, 2 prompts, 64 sampled tokens each; 2 s):
```
[stock] prompt 0: sampler == trainer log-prob on 0/64 sampled tokens; mean |Δ| 0.0367 nats, max |Δ| 0.144
[stock] prompt 1: sampler == trainer log-prob on 0/64 sampled tokens; mean |Δ| 0.0431 nats, max |Δ| 0.179
[mlx-exact] prompts 0, 1: sampler == trainer log-prob on 64/64 sampled tokens; mean |Δ| 0 nats
```
(Some lines are condensed here; the scripts print one line per case.)

## 8. Scope
Validated on one chip generation and one MLX version. The dispatch replica encodes MLX 0.32.4's thresholds, so `mlx_exact.enable()` refuses other environments unless forced. The same method applies elsewhere, but the replica and the canonical kernels must be re-derived and re-validated for each version and chip.

---
*Written with AI assistance (Anthropic's Claude); every number above comes from the measurements and scripts described here.*
