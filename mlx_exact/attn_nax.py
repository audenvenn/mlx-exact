"""Exact attention without query padding or key copies: MLX's own fused accelerator attention kernel (`attention_nax`
from steel/attn/kernels/steel_attention_nax.h, vendored verbatim from v0.32.4, see _vendor/), called directly with a
per-sequence key offset, for any number of queries.

MLX runs this kernel (64-query tiles, 32-key blocks, online softmax in fp32) for long prefills; its arithmetic is the
canonical one. For short query blocks MLX dispatches other kernels, and batched left padding shifts a sequence's keys
against the 32-key blocks, so mlx-exact padded the queries to 9 rows and copied K and V to re-align every sequence
(FINDINGS F14, F33). Calling MLX's kernel directly removes both: for sequence b the key pointer starts at its first valid
key and the key count is kL - pads[b], which is exactly what the kernel sees when the sequence runs alone.

Edits to MLX's code, made when the header is assembled (the vendored files are verbatim):
- the six function constants (align_Q, align_K, has_mask, do_causal, has_sinks, save_lse) become template parameters;
- the [[kernel]] entry point becomes an inline function whose parameter structs live in thread memory;
- includes are dropped (the headers are concatenated in dependency order).
align_Q / align_K are always false here: with an aligned length the bounds-checked branch is never entered, so the
arithmetic equals the aligned variant. Rows of a query tile are independent (each SIMD group owns 16 rows; row maxima
and sums are reduced within a row), so decode may use 16-row tiles (one SIMD group) and pack the query heads that
share a KV head into the rows of one tile, which reads that head's keys and values once.
"""
import re
from pathlib import Path

import mlx.core as mx

TP = Path(__file__).resolve().parent / "_vendor" / "mlx_v0_32_4"
HEAD_DIMS = (64, 96, 128)      # MLX's non-split accelerator attention (256 / 512 use a head-dim-split variant)


def _strip(txt):
    return "\n".join(l for l in txt.splitlines() if not l.startswith('#include "mlx/') and not l.startswith("#pragma once"))


def _attn_header():
    parts = [_strip((TP / f).read_text()) for f in ("defines.h", "type_traits.h", "integral_constant.h", "nax.h",
                                                    "attn_params.h")]
    lines = (TP / "steel_attention_nax.h").read_text().splitlines()
    assert lines[7] == "using namespace mlx::steel;" and lines[21] == "template <typename T>", "unexpected MLX header"
    assert lines[84].endswith("void attention_nax(") and lines[507] == "}", "unexpected MLX header"
    body = "\n".join(lines[21:508])           # helper ops (TransformScale ... DivOp) and the attention_nax kernel
    old_sig = ("    typename MaskType = float,\n    typename AccumType = float>\n"
               "[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(")
    new_sig = ("    typename MaskType,\n    typename AccumType,\n    bool align_Q,\n    bool align_K,\n    bool has_mask,\n"
               "    bool do_causal,\n    bool has_sinks,\n    bool save_lse>\nMETAL_FUNC void attention_nax_impl(")
    assert body.count(old_sig) == 1
    body = body.replace(old_sig, new_sig)
    i = body.index("METAL_FUNC void attention_nax_impl(")
    j = body.index(") { // clang-format on", i)
    sig = body[i:j]
    sig, n_attr = re.subn(r" \[\[(buffer\(\d+\)(, function_constant\(\w+\))?|thread_index_in_simdgroup|"
                          r"simdgroup_index_in_threadgroup|threadgroup_position_in_grid|thread_position_in_threadgroup)\]\]",
                          "", sig)
    assert n_attr == 13, n_attr
    assert sig.count("const constant Attn") == 2
    sig = sig.replace("const constant AttnParams*", "const thread AttnParams*")
    sig = sig.replace("const constant AttnMaskParams*", "const thread AttnMaskParams*")
    body = body[:i] + sig + body[j:]
    parts.append("using namespace mlx::steel;\n\n" + body)
    # MLX appends the generated `template <...>` line right after the header: end with a newline
    return "\n\n".join(parts) + "\n"


# One threadgroup per (query tile, head, sequence); each sequence gets its own key offset and key count.
_SRC = r"""
    constexpr int BQ = 16 * WM;
    constexpr int BK = 32;
    const int b = int(threadgroup_position_in_grid.z);
    // pads outside [0, kL) are a caller error: clamp so that reads stay in bounds and the key count stays >= 1
    const int off = clamp(int(pads[b]), 0, keys_shape[2] - 1);
    const int qL = queries_shape[2];
    const int kLb = keys_shape[2] - off;
    AttnParams p;
    p.B = queries_shape[0];
    p.H = queries_shape[1];
    p.D = queries_shape[3];
    p.qL = qL;
    p.kL = kLb;
    p.gqa_factor = queries_shape[1] / keys_shape[1];
    p.scale = scale[0];
    p.NQ = (qL + BQ - 1) / BQ;
    p.NK = (kLb + BK - 1) / BK;
    p.NQ_aligned = qL / BQ;
    p.NK_aligned = kLb / BK;
    p.qL_rem = qL - p.NQ_aligned * BQ;
    p.kL_rem = kLb - p.NK_aligned * BK;
    p.qL_off = kLb - qL;
    for (int i = 0; i < 3; i++) {
      p.Q_strides[i] = queries_strides[i];
      p.K_strides[i] = keys_strides[i];
      p.V_strides[i] = values_strides[i];
    }
    p.O_strides[0] = int64_t(queries_shape[1]) * qL * BD;
    p.O_strides[1] = int64_t(qL) * BD;
    p.O_strides[2] = BD;
    AttnMaskParams mp;
    mp.M_strides[0] = (mask_shape[0] == 1) ? 0 : mask_strides[0];
    mp.M_strides[1] = (mask_shape[1] == 1) ? 0 : mask_strides[1];
    mp.M_strides[2] = (mask_shape[2] == 1) ? 0 : mask_strides[2];
    attention_nax_impl<T, BQ, BK, BD, BD, WM, 1, bool, float, false, false, HAS_MASK, DO_CAUSAL, false, false>(
        queries, keys + int64_t(off) * keys_strides[2], values + int64_t(off) * values_strides[2], out, &p, &mp,
        mask + (HAS_MASK ? off : 0), queries, reinterpret_cast<device float*>(out), thread_index_in_simdgroup,
        simdgroup_index_in_threadgroup, threadgroup_position_in_grid, thread_position_in_threadgroup);
"""

_KERNELS = {}


def _dummy_mask():
    """Placeholder for calls without a mask (never read); 8 elements, so that MLX passes it as a device buffer."""
    if "mask" not in _KERNELS:
        _KERNELS["mask"] = mx.zeros((1, 1, 1, 8), mx.bool_)
    return _KERNELS["mask"]


def _kernel():
    if "attn" not in _KERNELS:
        _KERNELS["attn"] = mx.fast.metal_kernel(
            name="mlx_exact_attention_nax_offsets", input_names=["queries", "keys", "values", "mask", "pads", "scale"],
            output_names=["out"], source=_SRC, header=_attn_header(), ensure_row_contiguous=False)
    return _KERNELS["attn"]


def attention_offsets(queries, keys, values, scale, mask=None, pads=None, causal=False, kv_inplace=False, wm=None,
                      pack=None):
    """MLX's accelerator attention with per-sequence key offsets.
    - queries [B, Hq, L, D]; keys, values [B, Hkv, kL, D]; D in HEAD_DIMS; one dtype, bf16 or fp16.
    - mask: None or boolean, broadcastable to [B, Hq, L, kL]. causal (no array mask): bottom-right aligned over each
      sequence's kL - pads[b] keys.
    - pads: int [B] or None. Sequence b's key j is keys[b, :, pads[b] + j], with mask column pads[b] + j; keys before
      pads[b] are not read. Requires 0 <= pads[b] < kL (other values are clamped), and with causal also
      kL - pads[b] >= L (otherwise the first rows have no key).
    - kv_inplace: keys / values are KV-cache slices whose last dimension is contiguous (mlx-lm caches); they are read
      in place, whatever their outer strides. Otherwise they are made contiguous first (a copy only if they are not).
      Queries and the mask are always made contiguous (no copy if they already are).
    - Rows without any valid key get unspecified, finite values.
    - wm / pack override the tile choice (tests only)."""
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("attention_offsets: queries, keys and values must be 4-D")
    B, Hq, L, D = queries.shape
    Hkv, kL = keys.shape[1], keys.shape[2]
    if (tuple(keys.shape) != (B, Hkv, kL, D) or tuple(values.shape) != (B, Hkv, kL, D) or D not in HEAD_DIMS
            or Hkv < 1 or Hq < 1 or Hq % Hkv or kL < 1 or keys.dtype != queries.dtype or values.dtype != queries.dtype
            or queries.dtype not in (mx.bfloat16, mx.float16)):
        raise ValueError(f"attention_offsets: unsupported inputs q {tuple(queries.shape)} {queries.dtype}, "
                         f"k {tuple(keys.shape)} {keys.dtype}, v {tuple(values.shape)} {values.dtype}")
    if causal and mask is not None:
        raise ValueError("attention_offsets: causal and an array mask are exclusive")
    if B == 0 or L == 0:
        return mx.zeros((B, Hq, L, D), queries.dtype)
    pads = mx.zeros((B,), mx.int32) if pads is None else mx.contiguous(pads.astype(mx.int32).reshape(B))
    has_mask = mask is not None
    if has_mask:
        if mask.dtype != mx.bool_ or mask.ndim > 4:
            raise ValueError("attention_offsets: boolean masks of at most 4 dimensions only")
        m = mask if mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
        if any(a not in (1, t) for a, t in zip(m.shape, (B, Hq, L, kL))):
            raise ValueError(f"attention_offsets: mask {tuple(mask.shape)} does not broadcast to {(B, Hq, L, kL)}")
        if m.shape[-1] != kL:
            m = mx.broadcast_to(m, tuple(m.shape[:-1]) + (kL,))
        # MLX's kernel reads mask rows with a unit key stride: materialize broadcast or stepped views (no copy if the
        # mask is already contiguous); outer singleton dims stay broadcast (stride 0 in the kernel)
        m = mx.contiguous(m)
        if m.size < 8:
            # MLX passes inputs with fewer than 8 elements in the constant address space, MLX's kernel reads the mask
            # through a device pointer: append False columns (columns at or past a sequence's key count are never read)
            m = mx.concatenate([m, mx.zeros(tuple(m.shape[:-1]) + (8,), mx.bool_)], axis=-1)
    else:
        m = _dummy_mask()
    q = mx.contiguous(queries)
    if not kv_inplace:
        keys, values = mx.contiguous(keys), mx.contiguous(values)
    if pack is None:
        pack = L == 1 and Hq > Hkv and not causal and m.shape[1] == 1
    if pack:
        q = q.reshape(B, Hkv, Hq // Hkv, D)
    heads, rows = q.shape[1], q.shape[2]
    if wm is None:
        wm = 1 if rows <= 16 else 4
    bq = 16 * wm
    out = _kernel()(inputs=[q, keys, values, m, pads, mx.array([scale], mx.float32)],
                    template=[("T", queries.dtype), ("BD", D), ("WM", wm), ("HAS_MASK", has_mask),
                              ("DO_CAUSAL", bool(causal))],
                    grid=(((rows + bq - 1) // bq) * 32, heads * wm, B), threadgroup=(32, wm, 1),
                    output_shapes=[(B, heads, rows, D)], output_dtypes=[queries.dtype])[0]
    return out.reshape(B, Hq, L, D) if pack else out


def first_valid_key(mask, B, kL):
    """Per sequence, the smallest index of a valid key over all its query rows and heads (rows with no valid key are
    excluded; 0 if the sequence has none). Every valid key of the sequence lies at or after it."""
    m = mask if mask.ndim == 4 else mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
    mb = m if m.shape[0] == B else mx.broadcast_to(m, (B,) + tuple(m.shape[1:]))
    has = mx.any(mb, axis=-1)
    fv = mx.where(has, mx.argmax(mb.astype(mx.int32), axis=-1), kL)
    pad = mx.min(fv.reshape(B, -1), axis=-1)
    return mx.where(pad >= kL, 0, pad).astype(mx.int32)
