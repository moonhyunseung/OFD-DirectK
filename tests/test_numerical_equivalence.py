import torch

from reference.decode_then_score import decode_then_score
from reference.direct_packed_score import direct_packed_score
from reference.quant import quantize_int4_groupwise


def _errors(a: torch.Tensor, b: torch.Tensor):
    diff = a - b
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / b.norm().clamp_min(1e-30)).item()
    return max_abs, rel_l2


def test_float32_paths_are_numerically_close():
    torch.manual_seed(2)

    L, d, r = 129, 256, 6
    x = torch.randn(L, d, dtype=torch.float32) * 3.0
    q = torch.randn(r, d, dtype=torch.float32)
    alpha = d ** -0.5

    packed = quantize_int4_groupwise(x, group_size=32)

    z_ref = decode_then_score(q, packed, alpha, out_dtype=torch.float32)
    z_direct = direct_packed_score(q, packed, alpha, compute_dtype=torch.float32)

    max_abs, rel_l2 = _errors(z_direct, z_ref)

    # This is a numerical contract, not the proof of algebraic exactness.
    assert max_abs < 5e-5
    assert rel_l2 < 5e-6


def test_float64_is_tighter_than_float32_on_same_case():
    torch.manual_seed(3)

    L, d, r = 257, 256, 6
    x = torch.randn(L, d, dtype=torch.float32)
    q32 = torch.randn(r, d, dtype=torch.float32)
    alpha = d ** -0.5
    packed = quantize_int4_groupwise(x, group_size=32)

    ref32 = decode_then_score(q32, packed, alpha, out_dtype=torch.float32)
    direct32 = direct_packed_score(q32, packed, alpha, compute_dtype=torch.float32)
    err32 = (direct32 - ref32).abs().max().item()

    q64 = q32.double()
    ref64 = decode_then_score(q64, packed, alpha, out_dtype=torch.float64)
    direct64 = direct_packed_score(q64, packed, alpha, compute_dtype=torch.float64)
    err64 = (direct64 - ref64).abs().max().item()

    assert err64 <= err32 + 1e-12
