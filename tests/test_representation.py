import torch

from reference.quant import (
    dequantize_int4_groupwise,
    offset_bytes_to_twos_complement,
    quantize_int4_groupwise,
    unpack_codes,
    unpack_signed_int4,
)


def test_offset_binary_contract_roundtrip():
    x = torch.tensor(
        [[-7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0, 2.0] * 4],
        dtype=torch.float32,
    )
    obj = quantize_int4_groupwise(x, group_size=32)
    code = unpack_codes(obj).to(torch.int16)
    signed = unpack_signed_int4(obj)

    assert torch.equal(signed, code - 8)
    assert code.min().item() >= 0
    assert code.max().item() <= 15

    xhat = dequantize_int4_groupwise(obj)
    assert xhat.shape == x.shape


def test_bytewise_xor_flips_each_nibble_sign_bit():
    # Low nibble code 7 (q=-1), high nibble code 11 (q=+3).
    packed_offset = torch.tensor([0xB7], dtype=torch.uint8)
    packed_twos = offset_bytes_to_twos_complement(packed_offset)

    # 0xB7 XOR 0x88 = 0x3F:
    # low  0xF is two's-complement -1
    # high 0x3 is two's-complement +3
    assert packed_twos.item() == 0x3F
