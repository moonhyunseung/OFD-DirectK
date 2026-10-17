# Claims

This repository separates theorem-level claims, implementation claims, and empirical claims.

## Claim ladder

| ID | Claim | Type | Current status | Evidence required |
|---|---|---|---|---|
| K-C0 | The packed offset-binary INT4 representation is completely specified. | Representation | SUPPORTED | Codec contract + exhaustive nibble/byte tests |
| K-C1 | Dense dequantized K is not mathematically required to compute decoded-quantized logits. | Algebraic | PROVED | Exact identity |
| K-C2 | Group scale can move after the width-32 coordinate reduction exactly. | Algebraic | PROVED | Distributivity under groupwise shared scale |
| K-C3 | Offset correction can be reduced to a query-side group sum instead of per-coordinate subtraction. | Algebraic | PROVED | Exact identity |
| K-C4 | Signed nibble recoding can remove explicit `-8` correction without changing the represented quantized integer. | Representation | PROVED + FINITE CODEC VERIFIED | Exact code bijection + exhaustive byte test |
| K-C5 | A concrete packed lowering removes enough reconstruction work to beat decode-first execution on at least one workload regime. | Hardware/empirical | OPEN | Verified lowering + crossover experiment |
| K-C6 | The advantage transfers to real model decode without changing the declared numerical semantics. | System/empirical | OPEN | Model integration + paired measurement |

## Rules

- `PROVED` is reserved for algebraic or discrete finite claims that do not depend on sampled workloads.
- `SUPPORTED` means the representation or implementation contract is directly verified.
- Hardware speed claims remain `OPEN` until the relevant machine path is inspected and measured.
- A failed empirical claim does not invalidate K-C1–K-C3; it only closes a hardware realization branch.

## Non-claims

This repository does not claim lower quantization error, model-quality improvement, lower mathematical rank, or universal speedup.
