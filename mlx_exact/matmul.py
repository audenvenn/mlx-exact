"""Raw matmuls (`a @ b`, `mx.matmul`) while exact mode is on (0.13.0).

Models also apply weights with raw matmuls (MoE router gates, tied output layers) or compute attention with them
(Gemma 2, for its logit softcapping). MLX picks the kernel for these from the shape: gemv for one row, gemv_wide for
2-14 rows, accelerator kernels from about 16 rows, and for unbatched products split-K kernels whose partition count
depends on the shape. So the same product rounds differently in decode and prefill. Here every such call gets one
arithmetic per shape class:
- a 2-D right operand (a weight applied as x @ w.T, or any matrix): mlx-exact's dense route, the arithmetic of
  unquantized Linear layers;
- batched products (both operands at least 3-D): MLX's accelerator kernels, with at least 16 rows and 16 columns
  (padding rows that copy the first row, and zero columns, both discarded). MLX never splits batched products along K.
Other calls (vectors, mixed or non-bf16/fp16 dtypes) run stock. mlx-exact's own code uses ORIG_MATMUL.
"""
import mlx.core as mx

ORIG_MATMUL = mx.matmul                    # captured at import, before enable() patches anything
ORIG_ARRAY_MATMUL = mx.array.__matmul__
_DTYPES = (mx.bfloat16, mx.float16)
_MIN = 16
STATS = {}


def exact_matmul(a, b, *, stream=None):
    if stream is not None:                    # mx.matmul's optional stream / device argument (0.13.1)
        with mx.stream(stream):
            return exact_matmul(a, b)
    if not (isinstance(a, mx.array) and isinstance(b, mx.array)) or a.dtype != b.dtype or a.dtype not in _DTYPES:
        return ORIG_MATMUL(a, b)
    if b.ndim == 2 and a.ndim >= 2:
        from . import _dense_fn                       # the dense Linear route (custom VJP), imported at call time
        STATS["raw_dense"] = STATS.get("raw_dense", 0) + 1
        return _dense_fn(a, b.T)
    if a.ndim >= 3 and b.ndim >= 3:
        M, N = a.shape[-2], b.shape[-1]
        pr, pc = max(0, _MIN - M), max(0, _MIN - N)
        if pr:
            a = mx.concatenate([mx.repeat(a[..., :1, :], pr, axis=-2), a], axis=-2)
        if pc:
            b = mx.concatenate([b, mx.zeros(tuple(b.shape[:-1]) + (pc,), b.dtype)], axis=-1)
        STATS["raw_batched"] = STATS.get("raw_batched", 0) + 1
        out = ORIG_MATMUL(a, b)
        if pr:
            out = out[..., pr:, :]
        if pc:
            out = out[..., :N]
        return out
    STATS["raw_stock"] = STATS.get("raw_stock", 0) + 1
    return ORIG_MATMUL(a, b)


def _array_matmul(self, other):
    return exact_matmul(self, other)


def install():
    mx.matmul = exact_matmul
    mx.array.__matmul__ = _array_matmul


def uninstall():
    mx.matmul = ORIG_MATMUL
    mx.array.__matmul__ = ORIG_ARRAY_MATMUL


def gemma2_attention_call(self, x, mask=None, cache=None):
    """mlx-lm 0.32.2's Gemma 2 attention (raw matmuls with logit softcapping) with mlx-exact's softmax. Its matmuls go
    through exact_matmul (the `@` patch). MLX's own softmax changes its reduction with the row length (a different
    kernel above 4,096 keys), so decode and prefill rows differed beyond 4,096 tokens even with exact matmuls."""
    B, L, D = x.shape
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)
    queries = queries * self.scale
    if self.repeats > 1:
        queries = queries.reshape(B, self.n_kv_heads, self.repeats, L, self.head_dim)
        keys = mx.expand_dims(keys, 2)
        values = mx.expand_dims(values, 2)
    scores = queries @ keys.swapaxes(-1, -2)
    scores = mx.tanh(scores / self.attn_logit_softcapping)
    scores *= self.attn_logit_softcapping
    if mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))
        else:
            scores = scores + mask
    scores = _softmax_lanes_fn(scores) if scores.dtype in _DTYPES else mx.softmax(scores, precise=True, axis=-1)
    output = scores @ values
    if self.repeats > 1:
        output = output.reshape(B, self.n_heads, L, self.head_dim)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


@mx.custom_function
def _softmax_lanes_fn(x):
    from .kernels import softmax_lanes
    return softmax_lanes(x)


@_softmax_lanes_fn.vjp
def _softmax_lanes_vjp(primals, cot, out):
    """Softmax backward (0.13.1; Gemma 2's attention had no gradient in 0.13.0): dx = p * (dy - sum(dy * p))."""
    p, c = out.astype(mx.float32), cot.astype(mx.float32)
    return (p * (c - mx.sum(c * p, axis=-1, keepdims=True))).astype(primals[0].dtype)
