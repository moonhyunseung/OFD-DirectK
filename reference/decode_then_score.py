from __future__ import annotations

import torch

from reference.quant import PackedInt4, dequantize_int4_groupwise


def decode_then_score(
    q: torch.Tensor,
    k_packed: PackedInt4,
    alpha: float,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dense decoded-quantized reference score.

    Expected shapes:
      q: [r, d]
      decoded K: [L, d]
    """
    dtype = q.dtype if out_dtype is None else out_dtype
    khat = dequantize_int4_groupwise(k_packed, out_dtype=dtype)
    if q.ndim != 2 or khat.ndim != 2:
        raise ValueError("reference expects q=[r,d], K=[L,d]")
    if q.shape[-1] != khat.shape[-1]:
        raise ValueError("feature dimensions do not match")
    return (q.to(dtype) @ khat.T) * alpha
