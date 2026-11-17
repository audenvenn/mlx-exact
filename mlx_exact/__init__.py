"""mlx-exact: bit-exact, schedule-invariant inference for MLX / mlx-lm on Apple M5/M6 GPUs.

With `mlx_exact.enable()`, the logits for a token no longer depend on how the computation was scheduled: full prefill,
token-by-token decode, any prefill chunk size, batching with left padding and speculative verification all give
bit-identical results. Long prefills keep MLX's neural-accelerator speed; decode is slower than stock (see README).

Supported: MLX 0.32.4 on architecture generation >= 17 (validated on Apple M6), 4-bit affine quantization with group
size 64, bf16/fp16 activations, mlx-lm models that use `mlx_lm.models.base.scaled_dot_product_attention`.
Unsupported layers fall back to stock MLX with a one-time warning (exactness is then not guaranteed).
"""
import sys
import warnings

import mlx.core as mx
import mlx.nn as nn

from .attention import STATS as _ATT_STATS
from .attention import exact_sdpa
from .dispatch import CANONICAL, stock_qmm_path
from .kernels import qmv_nax_exact, qmv_nax_exact_v6

__version__ = "0.1.0"
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
    if bits != 4 or group_size != 64 or mode != "affine" or K % 64 != 0 or biases is None:
        _warn_once(("q", bits, group_size, mode), f"unsupported quantization (bits={bits}, group={group_size}, "
                   f"mode={mode}, K={K}); using stock MLX for these layers, exactness not guaranteed")
        _STATS["unsupported"] = _STATS.get("unsupported", 0) + 1
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode)
    path = stock_qmm_path(M, N, K, group_size)
    _STATS[path] = _STATS.get(path, 0) + 1
    if path == CANONICAL:
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    # v6 (coalesced 32-row tiles) is faster for wide layers; v0 (one thread per output) for narrow ones. Both are exact.
    kern = qmv_nax_exact_v6 if (K % 256 == 0 and N >= 2048) else qmv_nax_exact
    return kern(x.reshape(M, K), w, scales, biases, group_size, bits).reshape(*x.shape[:-1], N)


def enable(force=False):
    """Patch MLX / mlx-lm for exact inference. Call before or after loading a model."""
    if _orig:
        return
    _check_env(force)
    import mlx_lm.models.base as base
    _orig["ql"] = nn.QuantizedLinear.__call__
    _orig["qe"] = nn.QuantizedEmbedding.as_linear
    _orig["base"] = base.scaled_dot_product_attention

    def ql_call(self, x):
        y = quantized_matmul(x, self["weight"], self["scales"], self.get("biases"), self.group_size, self.bits,
                             getattr(self, "mode", "affine"))
        return y + self["bias"] if "bias" in self else y

    def qe_as_linear(self, x):
        return quantized_matmul(x, self["weight"], self["scales"], self.get("biases"), self.group_size, self.bits,
                                getattr(self, "mode", "affine"))

    nn.QuantizedLinear.__call__ = ql_call
    nn.QuantizedEmbedding.as_linear = qe_as_linear
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
    nn.QuantizedLinear.__call__ = _orig["ql"]
    nn.QuantizedEmbedding.as_linear = _orig["qe"]
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
