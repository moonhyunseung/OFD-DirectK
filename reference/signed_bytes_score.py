"""Reference recoding path; materializes logical integers, not a native kernel."""
import torch
from reference.quant import offset_bytes_to_twos_complement
from reference.direct_packed_score import direct_signed_score_from_qint

def direct_signed_bytes_score(q, packed, alpha, *, compute_dtype=None):
    raw = offset_bytes_to_twos_complement(packed.packed)
    n = torch.stack((raw & 15, (raw >> 4) & 15), -1).reshape(*raw.shape[:-1], packed.group_size).to(torch.int16)
    signed = torch.where(n >= 8, n - 16, n)
    return direct_signed_score_from_qint(q, signed, packed.scale, alpha, compute_dtype=compute_dtype)
