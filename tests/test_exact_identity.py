import torch

from reference.decode_then_score import decode_then_score
from reference.direct_packed_score import direct_packed_score, direct_signed_score_from_qint
from reference.quant import quantize_int4_groupwise, unpack_signed_int4


def test_direct_matches_decode_then_score_in_float64():
    torch.manual_seed(0)

    L, d, r = 17, 256, 6
    x = torch.randn(L, d, dtype=torch.float32)
    q = torch.randn(r, d, dtype=torch.float64)
    alpha = d ** -0.5

    packed = quantize_int4_groupwise(x, group_size=32)

    z_ref = decode_then_score(q, packed, alpha, out_dtype=torch.float64)
    z_direct = direct_packed_score(q, packed, alpha, compute_dtype=torch.float64)

    assert torch.allclose(z_direct, z_ref, rtol=1e-12, atol=1e-12)


def test_signed_normal_form_matches_offset_binary_form():
    torch.manual_seed(1)

    L, d, r = 11, 256, 6
    x = torch.randn(L, d, dtype=torch.float32)
    q = torch.randn(r, d, dtype=torch.float64)
    alpha = d ** -0.5

    packed = quantize_int4_groupwise(x, group_size=32)
    signed_qint = unpack_signed_int4(packed)

    z_offset = direct_packed_score(q, packed, alpha, compute_dtype=torch.float64)
    z_signed = direct_signed_score_from_qint(
        q,
        signed_qint,
        packed.scale,
        alpha,
        compute_dtype=torch.float64,
    )

    assert torch.allclose(z_signed, z_offset, rtol=1e-12, atol=1e-12)
