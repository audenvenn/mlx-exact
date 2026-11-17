# mlx-exact

**Bit-exact, schedule-invariant inference for [MLX](https://github.com/ml-explore/mlx) / mlx-lm on Apple M5/M6 GPUs.**

In stock MLX, a token's logits depend on how the computation was scheduled. Full prefill, token-by-token decode, the prefill chunk size, the batch layout and speculative verification each select different GPU kernels, and on M5/M6 different hardware units. These kernels round differently. Consequences:
- greedy decoding with a KV cache is not reproducible by recomputation;
- greedy speculative decoding changes the output;
- a request's output depends on which other requests share its batch;
- RL rollouts are not on-policy.

`mlx_exact.enable()` makes every schedule compute the same function, bit for bit. Long prefills keep MLX's neural-accelerator speed.

```python
import mlx_exact
from mlx_lm import load, generate

mlx_exact.enable()                     # before or after load()
model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit")
print(generate(model, tokenizer, "Hello", max_tokens=50))
print(mlx_exact.stats())               # which kernels stock MLX would have used, and the routes taken
```

## Results (Apple M6, MLX 0.32.4, mlx-lm 0.32.2)

| Test | Stock MLX | With mlx-exact |
|---|---|---|
| Qwen3-0.6B-4bit, 360-token prompt: decode / chunks of 4–100 vs full prefill | 0 / 360 positions bit-identical | 360 / 360 |
| Same at 2048 tokens (every 8th position + last 64) | 0 / 312 (except chunks of 512) | 312 / 312 |
| Qwen3.5-0.8B-4bit (hybrid gated delta + head dim 256), same tests | 0 identical | all identical |
| Greedy with KV cache vs recomputing the sequence each step | diverges at token 30 (Qwen3) / 2 (Qwen3.5) | identical |
| Greedy speculative decoding (Qwen3-1.7B + 0.6B draft, k = 2, 4) vs plain, 5 prompts × 200 tokens | 1 / 5 identical | 5 / 5 identical |
| Batched generation (`BatchGenerator`, left padding) vs each prompt alone, 4 prompts | 1 / 4 identical | 4 / 4 identical |
| MoE (mlx-lm `qwen3_moe`, 16 experts, top-4, random weights, 4-bit): decode / chunks of 4, 9, 64 vs full prefill | 0–12 / 200 positions | 200 / 200 |
| MoE greedy with KV cache vs recompute | diverges at token 4 | identical |
| Qwen3-0.6B-8bit: decode / chunks vs full prefill | 0 / 360 positions | 360 / 360 |
| Qwen3.5-0.8B unquantized (bf16): decode / chunks of 4, 9, 64 vs full prefill | 0 / 360 positions | 360 / 360 |
| Quantized KV cache (8-bit, Qwen3-0.6B-4bit): decode / chunks of 4, 9, 64 vs full prefill | 0 / 300 positions | 300 / 300 |
| RL with LoRA (Qwen3-0.6B-4bit): sampler vs trainer log-probs of 64 sampled tokens, before and after an optimizer step | 0 / 64 identical (mean \|Δ\| ≈ 0.05 nats/token) | 64 / 64 |
| Decode throughput (end to end, mlx-lm `generate_step`) | 1× | 0.93× (Qwen3-0.6B-4bit, Qwen3-1.7B-4bit); 0.97× (Qwen3-0.6B-8bit); 0.94× (Qwen3.5-0.8B-4bit); 0.98× (Qwen3.5-0.8B bf16) |
| Speculative decoding throughput (Qwen3-1.7B-4bit + 0.6B-4bit draft) | 1× | 0.78× (k = 2), 0.83× (k = 4) |
| Chunked prefill throughput (Qwen3-1.7B-4bit, chunks of 16 / 64 / 128 tokens) | 1× | **1.78× / 1.38× / 1.43×** (faster than stock); one-shot prefill 1.00× |

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
  - Query blocks are padded or split so that the same attention kernel always runs.
  - Batched sequences are re-aligned so that their keys start at index 0.

For calls with 4 or more rows, mlx-exact runs MLX's own accelerator implementation (vendored, MIT; `mlx_exact.nax_kernel`) with a 16-row tile, which is exact by construction and agrees bit for bit with the emulation kernels.

The full analysis is in [docs/numerics.md](docs/numerics.md): the kernel dispatch, the accelerator's measured accumulation order, the consequences in stock MLX, the method, and the results. The scripts in [`experiments/`](experiments) reproduce the main measurements (stock MLX vs mlx-exact, same run).

## Limitations
- **Validated environment:** MLX 0.32.4 on Apple M6 (GPU architecture generation 18). `enable()` refuses other environments unless `force=True`, because the kernel-dispatch replica is version- and chip-specific.
- **Weights and activations:** unquantized bf16/fp16, or 4- and 8-bit affine quantization with group size 64; bf16 or fp16 activations. Other formats, and fp32 activations, fall back to stock MLX with a warning, and exactness is then not guaranteed.
- **Not yet supported:** DoRA and LoRA on MoE expert layers (`LoRASwitchLinear`), 8-bit MoE experts, attention sinks combined with a quantized KV cache, and models that call `mx.fast.scaled_dot_product_attention` directly instead of mlx-lm's `scaled_dot_product_attention`.
- **MoE expert layers** (`QuantizedSwitchLinear` → `gather_qmm`) are supported. MLX's own sorted expert kernel equals the exact kernel per expert, bit for bit.
- **Speed:** end-to-end decode runs at 0.93–0.98× of stock on the models above (Apple M6), speculative decoding at 0.78–0.83×, and chunked prefill (chunks of 16–128 tokens) at 1.4–1.8× of stock: for those sizes MLX's own accelerator kernel with a 16-row tile, which mlx-exact runs, is faster than the kernels stock MLX picks. One-shot prefills run at stock speed. A decode step is a chain of dependent matmuls, so each kernel's latency matters more than its throughput. The decode kernels split every output row over many lanes, and keep the canonical summation order with a per-row chain of fp32 additions ([docs/numerics.md](docs/numerics.md), section 4). Remaining slow paths: MoE expert layers (one thread per output), and unquantized layers with more than 4 rows.
- **Accuracy:** the canonical arithmetic is that of MLX's prefill. Its per-matmul error is slightly higher than stock decode's because of the bf16 weight rounding, which is negligible next to 4-bit quantization error.

## Tests
`pytest tests/` checks:
- the kernels against MLX's prefill and MoE kernels;
- attention against prefill rows, and left-padding re-alignment;
- the dense kernel against MLX's canonical GEMM;
- decode == prefill end to end on a small random-weight MoE model (no download);
- with LoRA, sampler log-probs == trainer log-probs, and gradients flow, on a small random-weight model (no download);
- decode == prefill on `mlx-community/Qwen3-0.6B-4bit`, if it is in the local Hugging Face cache.
- the quantized KV cache: invariance, accuracy against an fp64 reference, and decode == prefill on a small random-weight model;
- unsupported activation dtypes fall back to stock MLX with a warning.
- every quantized route (1, 2–4 and more rows; 4- and 8-bit) against MLX's canonical kernel, and decode plus chunk sizes 2–64 against full prefill on small random-weight models.

## Disclosure
This code and the underlying analysis were developed with AI assistance (Anthropic's Claude) and verified by the tests and measurements described above.

## License
MIT (see `LICENSE`). `mlx_exact/_vendor/mlx_v0_32_4/` contains unmodified headers from MLX v0.32.4, © Apple Inc., MIT (see `NOTICE`).
