"""Bit-pattern comparison for checking exactness.

`mx.array_equal` compares values, so +0.0 equals -0.0. A claim of bit-identical results needs the stored bits
compared. That is what `bits_equal` does: for floating-point arrays of the same dtype it compares the bit patterns
through an unsigned-integer view. Other arrays (integers, booleans, or two different dtypes) fall back to
`mx.array_equal`.
"""
import mlx.core as mx

_VIEW = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}


def bits_equal(a, b, equal_nan=False):
    """True (as a scalar mx.array, like mx.array_equal) if a and b have the same shape and the same bits. NaNs compare
    by their bits; `equal_nan` is accepted for drop-in use and has no further effect."""
    a, b = mx.array(a), mx.array(b)
    if a.shape != b.shape:
        return mx.array(False)
    if a.dtype == b.dtype and a.dtype in _VIEW:
        u = _VIEW[a.dtype]
        return mx.array_equal(mx.view(mx.contiguous(a), u), mx.view(mx.contiguous(b), u))
    return mx.array_equal(a, b, equal_nan=equal_nan)
