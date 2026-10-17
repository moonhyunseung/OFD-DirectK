import torch
from reference.quant import PackedInt4, quantize_int4_groupwise
from reference.signed_bytes_score import direct_signed_bytes_score
from reference.decode_then_score import decode_then_score

def test_all_byte_states_and_endpoint():
    raw=torch.arange(256,dtype=torch.uint8).reshape(256,1,1)
    p=PackedInt4(raw,torch.ones(256,1,dtype=torch.float16),(256,2),2)
    q=torch.tensor([[1.,2.],[-3.,4.]],dtype=torch.float64)
    assert torch.equal(direct_signed_bytes_score(q,p,1.),decode_then_score(q,p,1.))

def test_qwen_geometry():
    gen=torch.Generator().manual_seed(1729)
    p=quantize_int4_groupwise(torch.randn(33,256,generator=gen))
    q=torch.randn(6,256,generator=gen,dtype=torch.float64)
    torch.testing.assert_close(direct_signed_bytes_score(q,p,1/16),decode_then_score(q,p,1/16),rtol=1e-12,atol=1e-12)

def test_zero_scale_quantizer_image_is_preserved():
    p=quantize_int4_groupwise(torch.zeros(2,32))
    assert (p.scale==0).all()  # frozen FP16 scale storage underflow, not changed
    assert torch.equal(direct_signed_bytes_score(torch.ones(1,32),p,1.),torch.zeros(1,2))
