# Status

## Current phase

**T1 — Reference semantics / algebra closure**

The repository has a frozen packed-INT4 representation contract, an algebraic DirectK identity, and reference implementations. Hardware performance is not yet claimed.

## Stage gates

| Stage | Requirement | Status |
|---|---|---|
| T0 Theory | Exact target operator and representation contract written | PASS |
| T1 Reference | Decode-first and direct reference paths defined | PASS |
| T2 Numerical | Floating-point comparison contract fixed and tested | IN PROGRESS |
| T3 Lowering | Exact packed form mapped to concrete machine primitives | NOT STARTED |
| T4 Kernel | Native microkernel implemented | NOT STARTED |
| T5 Boundary | Crossover/failure region measured | NOT STARTED |
| T6 Model | Real model decode integration validated | NOT STARTED |
| T7 Freeze | Commit, manifests, results, and claims frozen | NOT STARTED |

## Current theorem-level result

For frozen decoded quantized semantics

```text
Khat[t,g,j] = S_K[t,g] * (C_K[t,g,j] - 8)
```

the score can be evaluated without materializing `Khat`:

```text
Z[h,t]
= alpha * sum_g S_K[t,g]
  * (sum_j Q[h,g,j] C_K[t,g,j]
     - 8 * sum_j Q[h,g,j]).
```

This is algebraically exact over the reals under the frozen representation contract.

## Open blockers

1. Decide the preferred exact normal form: offset-binary + correction vs inline signed recoding.
2. Fix the numerical acceptance contract for FP32/FP16 accumulation orders.
3. Characterize native packed lowering on target GPUs before any speed claim.
4. Derive the crossover experiment from the hardware cost model rather than from blind sweeps.
