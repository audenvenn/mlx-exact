"""Schedule-invariant attention for MLX on Apple M5/M6: every query row is computed with the arithmetic MLX uses for
long prefills, whatever the number of queries in the call, the context length or the batch layout.
- MLX switches kernels by query count (vector kernels for <= 8 queries; for head dims 192/256 an unfused path that runs
  gemv-like matmuls for <= 12 rows, and a fused kernel for >= 1024 queries). Short query blocks are padded (9 rows, or
  16 for head dims 192/256); long blocks for head dims 192/256 are split below 1024 rows.
- MLX's attention kernels align KV blocks and 16-key fragments to key index 0, so batched left padding changes the
  summation order; each sequence's keys are re-aligned to start at index 0 first.
"""
import mlx.core as mx

from .kernels import dense_attention_exact, qkv_attention_exact, shift_rows_left, shift_rows_right
from .attn_nax import _dummy_mask, attention_offsets, first_valid_key

STATS = {}
_WARNED = set()


def _warn_once_attn(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        import warnings
        warnings.warn("mlx_exact: " + msg)


def reference_attention(q, k, v, scale, mask):
    """Differentiable attention from plain MLX ops (fp32 scores and softmax), any query count, grouped-query heads.
    mask: None, "causal" (bottom-right aligned), boolean or additive, broadcastable to [B, Hq, L, kL]. Rows without any
    valid key (boolean masks) give zeros, so they receive no gradient. Used only for backward passes: MLX's own
    attention gradient needs its full-attention kernel in the forward pass (more than 8 queries), and mlx-exact's
    kernels have none."""
    B, Hq, L, D = q.shape
    Hkv, kL = k.shape[1], k.shape[2]
    g = Hq // Hkv
    qf = (q.astype(mx.float32) * scale).reshape(B, Hkv, g, L, D)
    s = qf @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)                  # [B, Hkv, g, L, kL]
    if isinstance(mask, str):
        m = mx.arange(kL)[None, :] <= (mx.arange(L)[:, None] + kL - L)            # [L, kL]
    elif mask is not None:
        m = mask if mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
        # [Bm, Hm, Lm, Km] -> [Bm, Hkv | 1, g | 1, Lm, Km]; broadcasts against the scores
        m = m.reshape(m.shape[0], Hkv, g, m.shape[2], m.shape[3]) if m.shape[1] == Hq and Hq > 1 else m[:, :, None]
    else:
        m = None
    valid = None
    if m is not None and m.dtype == mx.bool_:
        s = mx.where(m, s, mx.finfo(mx.float32).min)
        valid = mx.any(mx.broadcast_to(m, s.shape), axis=-1, keepdims=True)
    elif m is not None:
        s = s + m.astype(mx.float32)
    o = mx.softmax(s, axis=-1) @ v.astype(mx.float32)[:, :, None]
    if valid is not None:
        o = mx.where(valid, o, 0.0)
    return o.reshape(B, Hq, L, v.shape[-1]).astype(q.dtype)


_GRAD_FNS = {}


def _differentiable(route, forward, scale, mask_kind, n_aux=0):
    """mx.custom_function around an exact attention forward(q, k, v, m, *aux); backward = reference_attention's
    gradient. mask_kind: "none", "causal" or "array" (m is then the mask; otherwise a placeholder)."""
    key = (route, float(scale), mask_kind, n_aux)
    if key not in _GRAD_FNS:
        @mx.custom_function
        def f(q, k, v, m, *aux):
            return forward(q, k, v, m, *aux)

        @f.vjp
        def f_vjp(primals, cot, out):
            q, k, v, m = primals[:4]
            mm = m if mask_kind == "array" else ("causal" if mask_kind == "causal" else None)
            _, g = mx.vjp(lambda a, b, c: reference_attention(a, b, c, scale, mm), [q, k, v], [cot])
            return (g[0], g[1], g[2]) + tuple(mx.zeros_like(x) for x in primals[3:])
        _GRAD_FNS[key] = f
    return _GRAD_FNS[key]


def _mask_kind(mask):
    return "none" if mask is None else ("causal" if isinstance(mask, str) else "array")


def _qkv_differentiable(scale, kind, group_size, bits):
    """Quantized-KV attention with a backward pass for the queries (reference attention on the dequantized cache);
    the cache itself gets zero gradients."""
    key = ("qkv", float(scale), kind, group_size, bits)
    if key not in _GRAD_FNS:
        def _m(m):
            return m if kind == "array" else ("causal" if kind == "causal" else None)

        @mx.custom_function
        def f(q, kq, ks, kb, vq, vs, vb, m):
            return qkv_attention_exact(q, (kq, ks, kb), (vq, vs, vb), scale, _m(m), group_size, bits)

        @f.vjp
        def f_vjp(primals, cot, out):
            q, kq, ks, kb, vq, vs, vb, m = primals
            k = mx.dequantize(kq, ks, kb, group_size=group_size, bits=bits)
            v = mx.dequantize(vq, vs, vb, group_size=group_size, bits=bits)
            _, g = mx.vjp(lambda a: reference_attention(a, k, v, scale, _m(m)), [q], [cot])
            return (g[0],) + tuple(mx.zeros_like(x) for x in primals[1:])
        _GRAD_FNS[key] = f
    return _GRAD_FNS[key]


@mx.custom_function
def _shift_left(x, pad):
    return shift_rows_left(x, pad)


@_shift_left.vjp
def _shift_left_vjp(primals, cot, out):
    """out[b, :, j] = x[b, :, j + pad_b]  =>  dx[b, :, i] = cot[b, :, i - pad_b] for i >= pad_b, else 0."""
    x, pad = primals
    B, kL = x.shape[0], x.shape[2]
    src = mx.arange(kL).reshape(1, 1, kL, 1) - pad.reshape(B, 1, 1, 1)
    g = mx.take_along_axis(cot, mx.broadcast_to(mx.maximum(src, 0), cot.shape), axis=2)
    return mx.where(src >= 0, g, 0).astype(x.dtype), mx.zeros_like(pad)


@mx.custom_function
def _shift_right(x, r, pad_to):
    return shift_rows_right(x, r, pad_to.shape[0])


@_shift_right.vjp
def _shift_right_vjp(primals, cot, out):
    """out[b, :, j] = x[b, :, j - r_b]  =>  dx[b, :, i] = cot[b, :, i + r_b]."""
    x, r, pad_to = primals
    B, kL = x.shape[0], x.shape[2]
    src = mx.arange(kL).reshape(1, 1, kL, 1) + r.reshape(B, 1, 1, 1)
    g = mx.take_along_axis(cot, mx.broadcast_to(src, (B, cot.shape[1], kL, cot.shape[3])), axis=2)
    return g.astype(x.dtype), mx.zeros_like(r), mx.zeros_like(pad_to)


def _is_batch_rotating(cache, keys):
    """mlx-lm BatchRotatingKVCache (batched sliding-window layers): one ring buffer shared by the batch, per-sequence
    offsets (absolute positions) and left padding."""
    return (cache is not None and hasattr(cache, "max_size") and hasattr(cache, "left_padding")
            and hasattr(cache, "_idx") and not hasattr(cache, "bits") and isinstance(getattr(cache, "offset", None), mx.array)
            and cache.offset.ndim == 1 and cache.offset.shape[0] == keys.shape[0])


def _batch_rotating_temporal(keys, values, mask, cache, L):
    """Temporal order: at decode the cache returns its ring in buffer order, oldest entry at the next write slot."""
    kL = keys.shape[2]
    if L == 1:
        i = int(cache._idx) % kL
        if i:
            keys = mx.concatenate([keys[:, :, i:], keys[:, :, :i]], axis=2)
            values = mx.concatenate([values[:, :, i:], values[:, :, :i]], axis=2)
            if mask is not None and not isinstance(mask, str):
                mask = mx.concatenate([mask[..., i:], mask[..., :i]], axis=-1)
    return keys, values, mask


def _batch_rotating_absolute(keys, values, mask, cache, L):
    """Temporal order, then shift every sequence right by (absolute position of its first key) mod 32 so that each key
    sits at an index congruent to its absolute position: the 32-key blocks of a full prefill, per sequence (the batched
    form of _rotating_to_absolute). The added rows are zero keys, always masked."""
    keys, values, mask = _batch_rotating_temporal(keys, values, mask, cache, L)
    B, H, kL, D = keys.shape
    r = mx.remainder(cache.offset.astype(mx.int32) - kL, KV_BLOCK)
    KO = kL + KV_BLOCK - 1
    pad_to = mx.zeros((KO,), mx.int32)
    k2 = _shift_right(keys, r, pad_to)
    v2 = _shift_right(values, r, pad_to)
    m = mask if mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
    m = mx.broadcast_to(m, (B,) + tuple(m.shape[1:3]) + (kL,))
    src = mx.arange(KO).reshape(1, 1, 1, KO) - r.reshape(B, 1, 1, 1)
    ok = (src >= 0) & (src < kL)
    idx = mx.broadcast_to(mx.clip(src, 0, kL - 1), (B,) + tuple(m.shape[1:3]) + (KO,))
    m2 = mx.take_along_axis(m, idx, axis=-1) & mx.broadcast_to(ok, idx.shape)
    return k2, v2, m2


def _align_left_padding(keys, values, mask):
    """Batched left padding shifts a sequence's keys relative to the attention kernel's fixed KV blocks and 16-key
    fragments (aligned to key index 0), which changes the summation order (FINDINGS F14). Re-align every sequence so its
    first valid key sits at index 0: shift K, V and the mask columns left, filling the tail with zeros / masked entries.
    The shift is the smallest first-valid key index over all query rows and heads (rows with no valid key excluded).
    Every row's valid keys lie at or after it, so shifting keys and mask together never changes any row's result: for
    left padding it equals the padding, for sliding-window and other masks it is at most the earliest valid key.
    Boolean masks only. On device, no host sync."""
    B, Hk, kL, Dk = keys.shape
    m = mask
    if m.ndim < 4:
        m = m.reshape((1,) * (4 - m.ndim) + m.shape)
    mb4 = m if m.shape[0] == B else mx.broadcast_to(m, (B,) + tuple(m.shape[1:]))
    has = mx.any(mb4, axis=-1)                                                   # (B, Hm, Lq): row has a valid key
    fv = mx.where(has, mx.argmax(mb4.astype(mx.int32), axis=-1), kL)             # first valid key per row
    pad = mx.min(fv.reshape(B, -1), axis=-1)
    pad = mx.where(pad >= kL, 0, pad).reshape(B, 1, 1, 1)
    j = mx.arange(kL).reshape(1, 1, 1, kL)
    src = j + pad
    valid = src < kL
    # shifted copies of K and V (per-sequence offset, zeros past the end): a plain copy kernel, no index arrays
    k2 = _shift_left(keys, pad.reshape(B).astype(mx.int32))
    v2 = _shift_left(values, pad.reshape(B).astype(mx.int32))
    mb = mx.broadcast_to(m, (B, m.shape[1], m.shape[2], kL))
    idx_m = mx.broadcast_to(mx.minimum(src, kL - 1), mb.shape)
    m2 = mx.logical_and(mx.take_along_axis(mb, idx_m, axis=-1), mx.broadcast_to(valid, mb.shape))
    return k2.astype(keys.dtype), v2.astype(values.dtype), m2


KV_BLOCK = 32       # key block of MLX's fused accelerator attention kernel (sdpa_full_self_attention_nax: bk = 32)

# MLX's own accelerator attention kernel with per-sequence key offsets (attn_nax.py): no query padding, no K/V copies.
# Enabled for the head dims and dtypes verified bit-exact against the padded / re-aligned path (FINDINGS F34). fp32 is
# bit-exact too while MLX_ENABLE_TF32 is on, but fp32 models are outside the exact matmul routes, so it keeps the old
# path.
OFFSETS_HEAD_DIMS = (64, 96, 128)
OFFSETS_DTYPES = (mx.bfloat16, mx.float16)
_ZEROS = {}
CONTIGUOUS_CACHES = ()      # set by enable(): mlx-lm cache classes that store contiguous K/V


def _sequence_origins(cache, mask, B, kL):
    """Index of each sequence's first key, i.e. where its keys start when it runs alone. mlx-lm's BatchKVCache records
    it (left_padding: sequence b's tokens are at [left_padding[b], kL)), which is exact for any mask that hides the
    padding, as mlx-lm's own masks do; the earliest key the mask shows is used if it comes first. Without such a
    cache it is derived from the mask: the earliest valid key over all rows, which equals the left padding for causal
    masks with left padding (for other masks the values are right, but the block origin can vary between schedules)."""
    fv = first_valid_key(mask, B, kL)
    lp = getattr(cache, "left_padding", None)
    if isinstance(lp, mx.array) and not hasattr(cache, "max_size") and lp.ndim == 1 and lp.shape[0] == B:
        # a mask that shows keys before the recorded padding (not mlx-lm's own) moves the origin back to them
        return mx.minimum(lp.astype(mx.int32), fv)
    return fv


def _zeros(B):
    if B not in _ZEROS:
        _ZEROS[B] = mx.zeros((B,), mx.int32)
    return _ZEROS[B]




def _is_rotating(cache, keys):
    """mlx-lm RotatingKVCache (sliding-window layers), single sequence, no kept prefix."""
    return (cache is not None and hasattr(cache, "max_size") and hasattr(cache, "_idx") and not hasattr(cache, "bits")
            and getattr(cache, "keep", 0) == 0 and keys.shape[0] == 1)


def _rotating_to_absolute(keys, values, mask, cache, L):
    """A full prefill from position 0 has key position p at index p, so the attention kernel's 32-key blocks hold
    absolute positions [32j, 32j + 32). A rotating cache returns its window starting at another position (chunked
    prefill) and, once the ring buffer has wrapped, in rotated order (decode). Put the window in temporal order and
    prepend (position of its first key) mod 32 zero keys, always masked, so every key sits at an index congruent to its
    absolute position: the same block contents as in the full prefill. Masked keys contribute exact zeros."""
    B, H, kL, Dk = keys.shape
    if L == 1 and cache.offset > kL:
        i = int(cache._idx) % kL                               # next write slot = oldest entry of the wrapped buffer
        if i:
            keys = mx.concatenate([keys[:, :, i:], keys[:, :, :i]], axis=2)
            values = mx.concatenate([values[:, :, i:], values[:, :, :i]], axis=2)
            if mask is not None and not isinstance(mask, str):  # an array mask is in buffer order too (max_size > window)
                mask = mx.concatenate([mask[..., i:], mask[..., :i]], axis=-1)
    r = (int(cache.offset) - kL) % KV_BLOCK
    if r == 0:
        return keys, values, mask
    keys = mx.concatenate([mx.zeros((B, H, r, Dk), keys.dtype), keys], axis=2)
    values = mx.concatenate([mx.zeros((B, values.shape[1], r, values.shape[3]), values.dtype), values], axis=2)
    if mask is None:
        m = mx.ones((1, 1, L, kL), mx.bool_)
    elif isinstance(mask, str):                                # "causal", bottom-right aligned over the kL keys
        m = (mx.arange(kL)[None, :] <= (mx.arange(L)[:, None] + kL - L)).reshape(1, 1, L, kL)
    else:
        m = mask if mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + mask.shape)
    m = mx.concatenate([mx.zeros((*m.shape[:-1], r), mx.bool_), m], axis=-1)
    return keys, values, m


def exact_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """queries [B, Hq, L, D]. For L <= 8 pad to 9 rows (copies of the first row placed BEFORE the real rows) and use
    a causal mask, so MLX runs the full-attention kernel; return the last L rows."""
    if hasattr(cache, "bits"):
        # quantized KV cache: mlx-lm's own composition (two quantized matmuls + precise softmax); its arithmetic is
        # defined here by fixed-order kernels, the same for every query count
        if sinks is not None:
            raise NotImplementedError("exact mode: attention sinks with a quantized KV cache")
        STATS["quantized_kv"] = STATS.get("quantized_kv", 0) + 1
        kind = _mask_kind(mask)
        return _qkv_differentiable(scale, kind, cache.group_size, cache.bits)(
            queries, *keys, *values, mask if kind == "array" else _dummy_mask())
    L = queries.shape[2]
    D = queries.shape[3]
    if (mask is not None and not isinstance(mask, str) and keys.shape[2] > 1
            and (mask.ndim == 0 or mask.shape[-1] == 1)):
        # a mask broadcast along the keys: give it its full key axis before rotating / re-aligning its columns
        m4 = mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape)) if mask.ndim < 4 else mask
        mask = mx.broadcast_to(m4, tuple(m4.shape[:-1]) + (keys.shape[2],))
    if (D in (192, 256) and keys.shape[1] == 1 and sinks is None
            and (mask is None or isinstance(mask, str) or mask.dtype == mx.bool_)):
        # For these head dims MLX attends with an unfused path whose internal kernels depend on the query and key
        # counts; with a single KV head that is not consistent across schedules (FINDINGS F30). mlx-exact's own
        # fixed-order attention depends only on the order of the valid keys (masked keys contribute exact zeros), so it
        # needs no padding or re-alignment, only temporal order for a rotating cache. Slower prefill than MLX's.
        if _is_rotating(cache, keys):
            keys, values, mask = _rotating_to_absolute(keys, values, mask, cache, L)
        elif _is_batch_rotating(cache, keys):
            # fixed-order arithmetic depends only on the order of the valid keys: temporal order is enough
            keys, values, mask = _batch_rotating_temporal(keys, values, mask, cache, L)
            STATS["sdpa_batch_rotating_temporal"] = STATS.get("sdpa_batch_rotating_temporal", 0) + 1
        STATS["sdpa_fixed_order"] = STATS.get("sdpa_fixed_order", 0) + 1
        kind = _mask_kind(mask)
        fwd = lambda q, k, v, mm: dense_attention_exact(q, k, v, scale, mm if kind == "array" else
                                                        ("causal" if kind == "causal" else None))
        return _differentiable("fixed_order", fwd, scale, kind)(queries, keys, values,
                                                                mask if kind == "array" else _dummy_mask())
    if L > keys.shape[2]:
        # more queries than keys (not a decoder's self-attention): stock attention, whose arithmetic this shape defines
        STATS["sdpa_more_queries_stock"] = STATS.get("sdpa_more_queries_stock", 0) + 1
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask, sinks=sinks)
    rotating = _is_rotating(cache, keys) and (mask is None or isinstance(mask, str) or mask.dtype == mx.bool_)
    if rotating:
        keys, values, mask = _rotating_to_absolute(keys, values, mask, cache, L)
        STATS["sdpa_rotating_aligned"] = STATS.get("sdpa_rotating_aligned", 0) + 1
    arr = mask is not None and not isinstance(mask, str)
    if (D in OFFSETS_HEAD_DIMS and sinks is None and values.shape[-1] == D and queries.dtype in OFFSETS_DTYPES
            and keys.dtype == queries.dtype and values.dtype == queries.dtype and (not arr or mask.dtype == mx.bool_)
            and (not isinstance(mask, str) or (mask == "causal" and L <= keys.shape[2]))):
        if not arr and L >= 9:
            # stock attention already runs the canonical kernel for these
            STATS["sdpa_full"] = STATS.get("sdpa_full", 0) + 1
            return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)
        B = queries.shape[0]
        brot = arr and not rotating and _is_batch_rotating(cache, keys)
        if brot:
            # batched sliding window: per-sequence absolute-position alignment (copies; sliding layers only)
            keys, values, mask = _batch_rotating_absolute(keys, values, mask, cache, L)
            STATS["sdpa_batch_rotating_aligned"] = STATS.get("sdpa_batch_rotating_aligned", 0) + 1
        kL = keys.shape[2]
        # a rotating cache's keys were placed at their absolute positions mod 32 above: no further offset
        pads = _sequence_origins(cache, mask, B, kL) if arr and not rotating and not brot else _zeros(B)
        m = mask if arr else _dummy_mask()
        # while exact mode is on, mlx-lm's attention caches store contiguous K/V (see enable()), so the slices they
        # return have a contiguous last dimension: read them in place. Anything else is made contiguous first.
        inplace = isinstance(cache, CONTIGUOUS_CACHES) and not brot
        STATS["sdpa_offsets"] = STATS.get("sdpa_offsets", 0) + 1
        kind = _mask_kind(mask)
        fwd = lambda q, k, v, mm, pp: attention_offsets(q, k, v, scale, mm if kind == "array" else None, pp,
                                                         kind == "causal", kv_inplace=inplace)
        return _differentiable("offsets_inplace" if inplace else "offsets", fwd, scale, kind, 1)(
            queries, keys, values, m, pads)
    if mask is not None and not isinstance(mask, str) and not rotating:
        if mask.dtype != mx.bool_:
            # additive (float) masks, e.g. DeepSeek-V3's MLA passes rotary score terms as the mask: no re-alignment or
            # padding; stock attention keeps the values correct, but exactness is not guaranteed
            _warn_once_attn("additive", "additive (non-boolean) attention masks are passed to stock attention; "
                            "exactness is not guaranteed for these layers")
            STATS["sdpa_additive_mask_stock"] = STATS.get("sdpa_additive_mask_stock", 0) + 1
            return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask, sinks=sinks)
        keys, values, mask = _align_left_padding(keys, values, mask)
        STATS["sdpa_left_aligned"] = STATS.get("sdpa_left_aligned", 0) + 1
    # Head dims 192/256 take MLX's UNFUSED path for 9 <= qL < 1024 (separate dense matmuls; M <= 12 runs a gemv-like
    # kernel, M >= 16 the accelerator), so pad those to 16 query rows. (qL >= 1024 switches to a fused kernel and very
    # long contexts can reach split-K in P@V: not yet canonical for these head dims; see FINDINGS F12.)
    P = 16 if D in (192, 256) else 9
    if D in (192, 256) and L >= 1024:
        # MLX switches these head dims to a fused head-dim-split kernel at >= 1024 queries: split the query block into
        # sub-blocks of <= 1008 rows (each with its keys up to its last position; causal, bottom-aligned) so that every
        # row goes through the same unfused path as decode and short chunks.
        if mask is not None and not (isinstance(mask, str) and mask == "causal"):
            raise NotImplementedError("exact mode: array masks with >= 1024 queries for head dim 192/256")
        outs = []
        kL = keys.shape[2]
        causal = mask is not None
        for s0 in range(0, L, 1008):
            e0 = min(L, s0 + 1008)
            kv_end = kL - (L - e0) if causal else kL          # unrestricted attention keeps every key
            outs.append(exact_sdpa(queries[:, :, s0:e0], keys[:, :, :kv_end], values[:, :, :kv_end], cache, scale,
                                   "causal" if causal else None, sinks))
        STATS["sdpa_split_1008"] = STATS.get("sdpa_split_1008", 0) + 1
        return mx.concatenate(outs, axis=2)
    if L >= P:
        STATS["sdpa_full"] = STATS.get("sdpa_full", 0) + 1
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask, sinks=sinks)
    arr = mask is not None and not isinstance(mask, str)
    if arr and mask.dtype != mx.bool_:
        raise NotImplementedError("exact mode: only boolean array masks are supported")
    kL = keys.shape[2]
    if kL >= P:
        # real rows last; causal is bottom-right aligned, so row i of P sees keys <= kL - P + i. With an array mask the
        # padded rows reuse the first real row's mask (their outputs are discarded); real rows keep their own masks.
        pad = P - L
        qp = mx.concatenate([mx.repeat(queries[:, :, :1], pad, axis=2), queries], axis=2)
        if arr:
            mp = mask if mask.shape[-2] == 1 else mx.concatenate([mx.repeat(mask[..., :1, :], pad, axis=-2), mask], axis=-2)
            STATS["sdpa_padded_arraymask"] = STATS.get("sdpa_padded_arraymask", 0) + 1
            out = mx.fast.scaled_dot_product_attention(qp, keys, values, scale=scale, mask=mp, sinks=sinks)
            return out[:, :, pad:]
        STATS["sdpa_padded"] = STATS.get("sdpa_padded", 0) + 1
        # causal, or a single query (sees every key either way); unrestricted attention for several queries
        pm = "causal" if (mask is not None or L == 1) else None
        out = mx.fast.scaled_dot_product_attention(qp, keys, values, scale=scale, mask=pm, sinks=sinks)
        return out[:, :, pad:]
    # Short context (decode of the first P-1 positions): a query block longer than the key block would leave MLX's fused
    # kernel. Build a square P x P causal problem instead: keys/values padded with zero rows at the end (always masked,
    # like the masked future keys in prefill), real queries at their true positions kL-L .. kL-1.
    zk = mx.zeros((*keys.shape[:2], P - kL, keys.shape[3]), keys.dtype)
    zv = mx.zeros((*values.shape[:2], P - kL, values.shape[3]), values.dtype)
    kp, vp = mx.concatenate([keys, zk], axis=2), mx.concatenate([values, zv], axis=2)
    before = kL - L
    parts = []
    if before > 0:
        parts.append(mx.repeat(queries[:, :, :1], before, axis=2))
    parts.append(queries)
    after = P - kL
    parts.append(mx.repeat(queries[:, :, -1:], after, axis=2))
    qp = mx.concatenate(parts, axis=2)
    if arr:
        # extend the mask: padded query rows copy neighbouring real rows; padded keys are always masked
        m = mask if mask.shape[-2] != 1 else mx.repeat(mask, L, axis=-2)
        rows = []
        if before > 0:
            rows.append(mx.repeat(m[..., :1, :], before, axis=-2))
        rows.append(m)
        rows.append(mx.repeat(m[..., -1:, :], after, axis=-2))
        m = mx.concatenate(rows, axis=-2)
        m = mx.concatenate([m, mx.zeros((*m.shape[:-1], P - kL), mx.bool_)], axis=-1)
        STATS["sdpa_padded_square_arraymask"] = STATS.get("sdpa_padded_square_arraymask", 0) + 1
        out = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=scale, mask=m, sinks=sinks)
        return out[:, :, before:before + L]
    STATS["sdpa_padded_square"] = STATS.get("sdpa_padded_square", 0) + 1
    if mask is None and L > 1:
        # unrestricted attention for several queries: every real key visible, the zero keys masked
        sq = mx.concatenate([mx.ones((1, 1, P, kL), mx.bool_), mx.zeros((1, 1, P, P - kL), mx.bool_)], axis=-1)
        out = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=scale, mask=sq, sinks=sinks)
    else:
        out = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=scale, mask="causal", sinks=sinks)
    return out[:, :, before:before + L]
