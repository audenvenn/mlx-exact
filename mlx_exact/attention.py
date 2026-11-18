"""Schedule-invariant attention for MLX on Apple M5/M6: every query row is computed with the arithmetic MLX uses for
long prefills, whatever the number of queries in the call, the context length or the batch layout.
- MLX switches kernels by query count (vector kernels for <= 8 queries; for head dims 192/256 an unfused path that runs
  gemv-like matmuls for <= 12 rows, and a fused kernel for >= 1024 queries). Short query blocks are padded (9 rows, or
  16 for head dims 192/256); long blocks for head dims 192/256 are split below 1024 rows.
- MLX's attention kernels align KV blocks and 16-key fragments to key index 0, so batched left padding changes the
  summation order; each sequence's keys are re-aligned to start at index 0 first.
"""
import mlx.core as mx

from .kernels import qkv_attention_exact

STATS = {}
_WARNED = set()


def _warn_once_attn(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        import warnings
        warnings.warn("mlx_exact: " + msg)


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
    idx_k = mx.broadcast_to(mx.minimum(src, kL - 1).reshape(B, 1, kL, 1), (B, Hk, kL, Dk))
    k2 = mx.where(mx.broadcast_to(valid.reshape(B, 1, kL, 1), k2s := (B, Hk, kL, Dk)), mx.take_along_axis(keys, idx_k, axis=2), 0)
    v2 = mx.where(mx.broadcast_to(valid.reshape(B, 1, kL, 1), (B, values.shape[1], kL, values.shape[3])),
                  mx.take_along_axis(values, mx.broadcast_to(mx.minimum(src, kL - 1).reshape(B, 1, kL, 1),
                                                             (B, values.shape[1], kL, values.shape[3])), axis=2), 0)
    mb = mx.broadcast_to(m, (B, m.shape[1], m.shape[2], kL))
    idx_m = mx.broadcast_to(mx.minimum(src, kL - 1), mb.shape)
    m2 = mx.logical_and(mx.take_along_axis(mb, idx_m, axis=-1), mx.broadcast_to(valid, mb.shape))
    return k2.astype(keys.dtype), v2.astype(values.dtype), m2


def exact_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """queries [B, Hq, L, D]. For L <= 8 pad to 9 rows (copies of the first row placed BEFORE the real rows) and use
    a causal mask, so MLX runs the full-attention kernel; return the last L rows."""
    if hasattr(cache, "bits"):
        # quantized KV cache: mlx-lm's own composition (two quantized matmuls + precise softmax); its arithmetic is
        # defined here by fixed-order kernels, the same for every query count
        if sinks is not None:
            raise NotImplementedError("exact mode: attention sinks with a quantized KV cache")
        STATS["quantized_kv"] = STATS.get("quantized_kv", 0) + 1
        return qkv_attention_exact(queries, keys, values, scale, mask, cache.group_size, cache.bits)
    L = queries.shape[2]
    D = queries.shape[3]
    if mask is not None and not isinstance(mask, str):
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
        for s0 in range(0, L, 1008):
            e0 = min(L, s0 + 1008)
            kv_end = kL - (L - e0)
            outs.append(exact_sdpa(queries[:, :, s0:e0], keys[:, :, :kv_end], values[:, :, :kv_end], cache, scale,
                                   "causal", sinks))
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
        out = mx.fast.scaled_dot_product_attention(qp, keys, values, scale=scale, mask="causal", sinks=sinks)
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
    out = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=scale, mask="causal", sinks=sinks)
    return out[:, :, before:before + L]
