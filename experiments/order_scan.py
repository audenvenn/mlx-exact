"""Map the summation order of MLX's matmul kernels on an Apple M5/M6 GPU with crafted inputs, then validate the
reconstructed order of the neural-accelerator kernel bit for bit on random data.

Probe: one dot product of length 128 holds +2^24 at position p, -2^24 at q and a single 1 at position m (exact result
1). In fp32 the 1 is lost if it is added to a partial sum that holds exactly one of the large terms; scanning m over all
positions (one output column per m) maps the order. Outputs are 0/1, exact in bf16.
Usage: python experiments/order_scan.py
"""
import numpy as np
import mlx.core as mx

K = N = 128


def scan(p, q, M):
    h = 2.0 ** 12
    x = np.ones(K, np.float32); x[p] = x[q] = h
    W = np.zeros((N, K), np.float32); W[:, p] = h; W[:, q] = -h
    for j in range(N):
        if j not in (p, q):
            W[j, j] = 1.0
    y = mx.array(np.tile(x, (M, 1))).astype(mx.bfloat16) @ mx.array(W).astype(mx.bfloat16).T
    row = np.array(y[0].astype(mx.float32))
    return "".join("." if j in (p, q) else ("1" if row[j] == 1 else "0") for j in range(N))


def emulate(P):
    """Reconstructed accelerator order: chunks of 16; quads {2r,2r+1,2r+8,2r+9} = (pair)+(pair) in fp32; in order."""
    acc = np.zeros(P.shape[0], np.float32)
    for c in range(0, P.shape[1], 16):
        for r in range(4):
            a, b, cc, d = (P[:, c + o] for o in (2 * r, 2 * r + 1, 2 * r + 8, 2 * r + 9))
            acc = (acc + ((a + b) + (cc + d))).astype(np.float32)
    return acc


if __name__ == "__main__":
    print("Survival pattern of the small term (1 = survives, 0 = lost), bf16, +B at p, -B at q:")
    for p, q in [(0, 127), (127, 0), (5, 70), (31, 32)]:
        for M, name in ((1, "gemv (decode, 1 row)"), (128, "accelerator GEMM (128 rows)")):
            print(f"  p={p:3d} q={q:3d} {name:28s} {scan(p, q, M)}")
    # Validation on random fp16 data (bf16 output is too coarse to tell orders apart): 4-bit weights, K = 4096, 512
    # rows -> the canonical accelerator kernel (64-row tiles), whose loader rounds each dequantized weight to fp16.
    rng = np.random.default_rng(0)
    Nv, Kv = 2048, 4096
    W = mx.array((rng.standard_normal((Nv, Kv)) * 0.02).astype(np.float32)).astype(mx.float16)
    x = mx.array(rng.standard_normal((512, Kv)).astype(np.float32)).astype(mx.float16)
    wq, s, b = mx.quantize(W, group_size=64, bits=4)
    gpu = np.array(mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)[0].astype(mx.float32))
    w = np.array(wq).astype(np.uint64)
    qv = ((w[..., None] >> (4 * np.arange(8, dtype=np.uint64))) & 0xF).reshape(Nv, -1).astype(np.float32)
    deq = (np.repeat(np.array(s.astype(mx.float32)), 64, 1) * qv + np.repeat(np.array(b.astype(mx.float32)), 64, 1))
    wr = deq.astype(np.float32).astype(np.float16).astype(np.float32)      # weights rounded to the activation type
    P = (wr * np.array(x[0].astype(mx.float32))[None, :]).astype(np.float32)
    emu = emulate(P).astype(np.float16).astype(np.float32)
    exact_dot = (wr.astype(np.float64) @ np.array(x[0].astype(mx.float32)).astype(np.float64)).astype(np.float16).astype(np.float32)
    print(f"Accelerator kernel vs reconstructed order: {int(np.sum(gpu != emu))} / {Nv} outputs differ "
          f"(an exact dot product: {int(np.sum(gpu != exact_dot))} / {Nv}).")
