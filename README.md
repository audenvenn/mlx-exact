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
| Decode throughput (end to end) | 1× | 0.61× (Qwen3-0.6B, 167 vs 272 tok/s); 0.70–0.78× (Qwen3-1.7B, 92–103 vs 132–134 tok/s) |

## How it works
MLX picks the kernel for a 4-bit matmul from the matrix shape:
- `qmv` for one row;
- `qmv_wide` for small batches;
- split-K kernels with 2–16 partitions for medium sizes;
- the accelerator kernel for large sizes.

These differ in two ways. Matrix-matrix kernels round each dequantized weight to bf16/fp16 before multiplying, and vector kernels do not. And each kernel sums in a different order. Attention similarly switches kernels with the number of queries, and its fixed KV blocks make left padding change the summation order.

mlx-exact takes the arithmetic MLX already uses for long prefills as **canonical**:
- **4-bit matmuls.** Every call that MLX would send elsewhere is rerouted to a kernel that reproduces it bit for bit. The accelerator's accumulation order (K in chunks of 16, specific fp32 pair/quad sums) was reconstructed with crafted inputs and validated against the hardware.
- **Attention.**
  - Query blocks are padded or split so that the same attention kernel always runs.
  - Batched sequences are re-aligned so that their keys start at index 0.

An alternative kernel (`mlx_exact.nax_kernel`) runs MLX's own accelerator implementation (vendored, MIT) with a 16-row tile; the two kernels agree bit for bit.

## Limitations
- **Validated environment:** MLX 0.32.4 on Apple M6 (GPU architecture generation 18). `enable()` refuses other environments unless `force=True`, because the kernel-dispatch replica is version- and chip-specific.
- **Quantization:** 4-bit affine with group size 64 only. Other quantizations fall back to stock MLX with a warning, and exactness is then not guaranteed.
- **Not yet supported:** MoE layers (`gather_qmm`), dense (unquantized) models, quantized KV caches, and models that call `mx.fast.scaled_dot_product_attention` directly instead of mlx-lm's `scaled_dot_product_attention`.
- **Speed:** decode is slower than stock. For matrices of ≥ 2048 rows the exact kernel runs at 0.82–0.92× of stock speed (kernel time, 4-bit, up to 152k × 1k and 14k × 4k); narrower matrices run at about 0.5×. End-to-end decode is 0.61–0.78× on the models above, and larger models (wider layers) should do better. Long prefills run at stock speed.
- **Accuracy:** the canonical arithmetic is that of MLX's prefill. Its per-matmul error is slightly higher than stock decode's because of the bf16 weight rounding, which is negligible next to 4-bit quantization error.

## Tests
`pytest tests/` checks the kernels against MLX's prefill kernel, attention against prefill rows, left-padding re-alignment, and (if `mlx-community/Qwen3-0.6B-4bit` is in the local Hugging Face cache) decode == prefill end to end.

## Disclosure
This code and the underlying analysis were developed with AI assistance (Anthropic's Claude) and verified by the tests and measurements described above.

## License
MIT (see `LICENSE`). `mlx_exact/_vendor/mlx_v0_32_4/` contains unmodified headers from MLX v0.32.4, © Apple Inc., MIT (see `NOTICE`).
