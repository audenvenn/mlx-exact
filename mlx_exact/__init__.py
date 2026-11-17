"""mlx-exact: bit-exact, schedule-invariant inference for MLX / mlx-lm on Apple M5/M6 GPUs.

With `mlx_exact.enable()`, the logits for a token no longer depend on how the computation was scheduled: full prefill,
token-by-token decode, any prefill chunk size, batching with left padding and speculative verification all give
bit-identical results. The exact routes are differentiable (custom VJPs), so an RL trainer's forward pass computes
exactly the log-probabilities its sampler computed (LoRA adapters included). Long prefills keep MLX's neural-accelerator speed; decode is slower than stock (see README).

Supported: MLX 0.32.4 on architecture generation >= 17 (validated on Apple M6), unquantized bf16/fp16 layers and 4- and
8-bit affine quantization with group
size 64 (dense and MoE expert layers), bf16/fp16 activations, mlx-lm models that use
`mlx_lm.models.base.scaled_dot_product_attention`.
Unsupported layers fall back to stock MLX with a one-time warning (exactness is then not guaranteed).
"""
import sys
import warnings

import mlx.core as mx
import mlx.nn as nn

from .attention import STATS as _ATT_STATS
from .attention import exact_sdpa
from .dispatch import CANONICAL, dense_stock_path, stock_qmm_path
from .kernels import (dense_exact, fp32_seq_matmul, gather_qmv_nax_exact, qmv_exact_bits, qmv_exact_v6b,
                      qmv_nax_exact, qmv_nax_exact_v6)

__version__ = "0.3.1"
VALIDATED_MLX = "0.32.4"
_STATS = {}
_orig = {}
_warned = set()


def _check_env(force):
    problems = []
    if mx.__version__ != VALIDATED_MLX:
        problems.append(f"MLX {mx.__version__} (validated: {VALIDATED_MLX}; the kernel-dispatch replica is version-specific)")
    try:
        arch = mx.device_info().get("architecture", "")
    except Exception:
        arch = mx.metal.device_info().get("architecture", "")
    gen = int("".join(c for c in arch.split("_g")[-1] if c.isdigit()) or 0) if "_g" in arch else 0
    if gen < 17 or arch.endswith("d"):
        problems.append(f"GPU architecture {arch!r} (validated: generation >= 17, not 'd', e.g. applegpu_g18g = M6)")
    if problems and not force:
        raise RuntimeError("mlx_exact: unvalidated environment: " + "; ".join(problems) + ". Pass force=True to try anyway.")
    for p in problems:
        warnings.warn("mlx_exact: " + p)


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        warnings.warn("mlx_exact: " + msg)


def quantized_matmul(x, w, scales, biases, group_size=64, bits=4, mode="affine"):
    """Drop-in for mx.quantized_matmul(..., transpose=True) with the canonical (long-prefill) arithmetic."""
    K, N = x.shape[-1], w.shape[0]
    M = x.size // K
    if (bits not in (4, 8) or group_size != 64 or mode != "affine" or K % 64 != 0 or biases is None
            or x.dtype not in (mx.bfloat16, mx.float16)):
        # fp32 activations: MLX's canonical kernel then runs at reduced (TF32-like) precision by default
        # (MLX_ENABLE_TF32=1), which the exact kernels do not emulate
        _warn_once(("q", bits, group_size, mode, str(x.dtype)), f"unsupported quantized layer (bits={bits}, "
                   f"group={group_size}, mode={mode}, K={K}, activations {x.dtype}); using stock MLX for these layers, "
                   "exactness not guaranteed")
        _STATS["unsupported"] = _STATS.get("unsupported", 0) + 1
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
    path = stock_qmm_path(M, N, K, group_size)
    _STATS[path] = _STATS.get(path, 0) + 1
    if path == CANONICAL:
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    if bits == 8:
        k8 = qmv_exact_v6b if K % 128 == 0 else qmv_exact_bits
        return k8(x.reshape(M, K), w, scales, biases, group_size, 8).reshape(*x.shape[:-1], N)
    # v6 (coalesced 32-row tiles) is faster for wide layers (N >= 4096); v0 (one thread per output) for narrower ones.
    # Both are exact (interleaved benchmark on M6, FINDINGS F18).
    kern = qmv_nax_exact_v6 if (K % 256 == 0 and N >= 4096) else qmv_nax_exact
    return kern(x.reshape(M, K), w, scales, biases, group_size, bits).reshape(*x.shape[:-1], N)


def dense_matmul(x, w):
    """Drop-in for x @ w.T (unquantized bf16/fp16 Linear) with the canonical (regular accelerator GEMM) arithmetic."""
    K, N = x.shape[-1], w.shape[0]
    M = x.size // K
    path = dense_stock_path(M, N, K)
    _STATS["dense_" + path] = _STATS.get("dense_" + path, 0) + 1
    if path == "nax" or K % 16 != 0 or x.dtype != w.dtype or x.dtype not in (mx.bfloat16, mx.float16):
        if path != "nax":
            _warn_once(("d", str(x.dtype), str(w.dtype)), f"unsupported dense layer (x {x.dtype}, w {w.dtype}, K={K}); "
                       "using stock MLX, exactness not guaranteed")
        return (x.reshape(M, K) @ w.T).reshape(*x.shape[:-1], N)
    return dense_exact(x, w)


def gather_quantized_matmul(x, w, scales, biases, indices, sorted_indices=False, group_size=64, bits=4,
                            mode="affine"):
    """Drop-in for mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True, ...) (MoE experts) with
    the canonical arithmetic. MLX's own canonical path (sorted rows, one row per pair, >= 16 pairs, >= 4 per expert)
    is kept; every other call runs the expert-indexed exact kernel."""
    E, N, K, M = w.shape[0], w.shape[1], x.shape[-1], x.shape[-2]
    batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
    B = 1
    for d in batch:
        B *= d
    if (bits != 4 or group_size != 64 or mode != "affine" or K % 64 != 0 or biases is None
            or x.dtype not in (mx.bfloat16, mx.float16)):
        _warn_once(("g", bits, group_size, mode, str(x.dtype)), f"unsupported expert quantization (bits={bits}, "
                   f"group={group_size}, mode={mode}, activations {x.dtype}); using stock MLX, exactness not guaranteed")
        _STATS["gather_unsupported"] = _STATS.get("gather_unsupported", 0) + 1
        return mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True, group_size=group_size,
                             bits=bits, mode=mode, sorted_indices=sorted_indices)
    if M == 1 and sorted_indices and B >= 16 and B // E >= 4:
        _STATS["gather_canonical"] = _STATS.get("gather_canonical", 0) + 1
        return mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True, group_size=group_size,
                             bits=bits, sorted_indices=sorted_indices)
    _STATS["gather_rerouted"] = _STATS.get("gather_rerouted", 0) + 1
    xr = mx.broadcast_to(x, (*batch, M, K)).reshape(-1, K)
    ir = mx.broadcast_to(mx.expand_dims(indices, -1), (*batch, M)).reshape(-1)
    return gather_qmv_nax_exact(xr, w, scales, biases, ir, group_size, bits).reshape(*batch, M, N)


# ---- differentiable wrappers: forward = exact routes; backward = stock ops (gradients need not be bit-exact, only the
# forward must match, e.g. the sampler's and the trainer's log-probabilities in RL) ----------------------------------
_fns = {}


def _sum_to(g, shape):
    """Reduce a broadcast gradient back to `shape`."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, d in enumerate(shape):
        if d == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _qmm_fn(group_size, bits, mode):
    key = ("q", group_size, bits, mode)
    if key not in _fns:
        @mx.custom_function
        def f(x, w, scales, biases):
            return quantized_matmul(x, w, scales, biases, group_size, bits, mode)

        @f.vjp
        def f_vjp(primals, cot, out):
            x, w, scales, biases = primals
            dx = mx.quantized_matmul(cot, w, scales, biases, transpose=False, group_size=group_size, bits=bits, mode=mode)
            return dx.astype(x.dtype), mx.zeros_like(w), mx.zeros_like(scales), mx.zeros_like(biases)
        _fns[key] = f
    return _fns[key]


def _gather_fn(group_size, bits, mode, sorted_indices):
    key = ("g", group_size, bits, mode, sorted_indices)
    if key not in _fns:
        @mx.custom_function
        def f(x, w, scales, biases, indices):
            return gather_quantized_matmul(x, w, scales, biases, indices, sorted_indices, group_size, bits, mode)

        @f.vjp
        def f_vjp(primals, cot, out):
            x, w, scales, biases, indices = primals
            dx = mx.gather_qmm(cot, w, scales, biases, rhs_indices=indices, transpose=False, group_size=group_size,
                               bits=bits, mode=mode)
            return (_sum_to(dx, x.shape).astype(x.dtype), mx.zeros_like(w), mx.zeros_like(scales),
                    mx.zeros_like(biases), mx.zeros_like(indices))
        _fns[key] = f
    return _fns[key]


@mx.custom_function
def _dense_fn(x, w):
    return dense_matmul(x, w)


@_dense_fn.vjp
def _dense_vjp(primals, cot, out):
    x, w = primals
    c2 = cot.reshape(-1, cot.shape[-1])
    return (cot @ w).astype(x.dtype), (c2.T @ x.reshape(-1, x.shape[-1])).astype(w.dtype)


@mx.custom_function
def _lora_fn(x, w):
    return fp32_seq_matmul(x, w)


@_lora_fn.vjp
def _lora_vjp(primals, cot, out):
    x, w = primals
    c2 = cot.reshape(-1, cot.shape[-1])
    return (cot @ w.T).astype(x.dtype), (x.reshape(-1, x.shape[-1]).astype(mx.float32).T @ c2).astype(w.dtype)


def enable(force=False):
    """Patch MLX / mlx-lm for exact inference. Call before or after loading a model."""
    if _orig:
        return
    _check_env(force)
    import mlx_lm.models.base as base
    _orig["l"] = nn.Linear.__call__
    _orig["e"] = nn.Embedding.as_linear
    _orig["ql"] = nn.QuantizedLinear.__call__
    _orig["qe"] = nn.QuantizedEmbedding.as_linear
    _orig["base"] = base.scaled_dot_product_attention

    def ql_call(self, x):
        b = self.get("biases")
        if b is None:
            y = quantized_matmul(x, self["weight"], self["scales"], None, self.group_size, self.bits,
                                 getattr(self, "mode", "affine"))
        else:
            y = _qmm_fn(self.group_size, self.bits, getattr(self, "mode", "affine"))(x, self["weight"], self["scales"], b)
        return y + self["bias"] if "bias" in self else y

    def qe_as_linear(self, x):
        b = self.get("biases")
        if b is None:
            return quantized_matmul(x, self["weight"], self["scales"], None, self.group_size, self.bits,
                                    getattr(self, "mode", "affine"))
        return _qmm_fn(self.group_size, self.bits, getattr(self, "mode", "affine"))(x, self["weight"], self["scales"], b)

    def l_call(self, x):
        y = _dense_fn(x, self["weight"])
        return y + self["bias"] if "bias" in self else y

    nn.Linear.__call__ = l_call
    nn.Embedding.as_linear = lambda self, x: _dense_fn(x, self["weight"])
    nn.QuantizedLinear.__call__ = ql_call
    nn.QuantizedEmbedding.as_linear = qe_as_linear
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    _orig["qsl"] = QuantizedSwitchLinear.__call__

    def qsl_call(self, x, indices, sorted_indices=False):
        if getattr(self, "global_scale", None) is not None:
            _warn_once("gs", "expert layers with a global scale are not supported; using stock MLX")
            return _orig["qsl"](self, x, indices, sorted_indices)
        b = self.get("biases")
        if b is None:
            return _orig["qsl"](self, x, indices, sorted_indices)
        y = _gather_fn(self.group_size, self.bits, getattr(self, "mode", "affine"), bool(sorted_indices))(
            x, self["weight"], self["scales"], b, indices)
        return y + mx.expand_dims(self["bias"][indices], -2) if "bias" in self else y

    QuantizedSwitchLinear.__call__ = qsl_call
    try:
        from mlx_lm.tuner.lora import LoRALinear
        _orig["lora"] = LoRALinear.__call__

        def lora_call(self, x):
            y = self.linear(x)
            z = _lora_fn(_lora_fn(self.dropout(x), self.lora_a), self.lora_b)
            return y + (self.scale * z).astype(x.dtype)

        LoRALinear.__call__ = lora_call
    except ImportError:
        pass
    base.scaled_dot_product_attention = exact_sdpa
    _orig["mods"] = []
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and getattr(mod, "scaled_dot_product_attention", None) is _orig["base"]:
            mod.scaled_dot_product_attention = exact_sdpa
            _orig["mods"].append(name)


def disable():
    """Undo enable()."""
    if not _orig:
        return
    import mlx_lm.models.base as base
    nn.Linear.__call__ = _orig["l"]
    nn.Embedding.as_linear = _orig["e"]
    nn.QuantizedLinear.__call__ = _orig["ql"]
    nn.QuantizedEmbedding.as_linear = _orig["qe"]
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    QuantizedSwitchLinear.__call__ = _orig["qsl"]
    if "lora" in _orig:
        from mlx_lm.tuner.lora import LoRALinear
        LoRALinear.__call__ = _orig["lora"]
    base.scaled_dot_product_attention = _orig["base"]
    for name in _orig["mods"]:
        sys.modules[name].scaled_dot_product_attention = _orig["base"]
    for name, mod in list(sys.modules.items()):           # modules imported while enabled
        if name.startswith("mlx_lm.models.") and getattr(mod, "scaled_dot_product_attention", None) is exact_sdpa:
            mod.scaled_dot_product_attention = _orig["base"]
    _orig.clear()


def stats():
    """Counts of matmul calls by the kernel stock MLX would have used, and attention routes taken."""
    return {"matmul": dict(_STATS), "attention": dict(_ATT_STATS)}
