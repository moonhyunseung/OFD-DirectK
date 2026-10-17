# OFD-DirectK

**Exact representation-native attention-score evaluation from packed quantized K, without materializing dense dequantized keys.**

OFD-DirectK isolates one claim from the broader OFD research line and studies it to completion before any wider optimization is considered.

## Scope

This repository studies only the K-side map

```text
packed INT4 K + per-group scales + Q
    -> exact attention logits
```

under the frozen groupwise quantization contract

```text
Khat[t,g,j] = S_K[t,g] * (C_K[t,g,j] - 8)
```

with current geometry

```text
r = 6 query heads per KV head
G = 8 quantization groups
m = 32 coordinates per group
d = G*m = 256
```

The exact target operator is

```text
Z = alpha * Q * Khat^T
```

and the central algebraic rewrite is

```text
Z[h,t]
= alpha * sum_g S_K[t,g]
  * ( sum_j Q[h,g,j] C_K[t,g,j]
      - 8 * sum_j Q[h,g,j] ).
```

This identity removes dense K dequantization as a mathematical intermediate while preserving the same decoded-quantized score operator over the reals.

## Non-goals

This repository does **not** currently study:

- V-side execution;
- SVD or low-rank approximation;
- persistent query nullspaces;
- Hadamard transforms;
- scale-matrix low-rank approximation;
- token dropping or block dropping;
- quality/perplexity changes;
- whole-model speed claims.

Those questions stay outside this repository until the DirectK claim is closed.

## Research order

The order is mandatory:

1. **Representation contract** — define exactly what the packed bytes and scales mean.
2. **Algebraic identity** — prove equality with decode-then-score semantics.
3. **Reference implementation** — implement both paths without performance claims.
4. **Numerical contract** — distinguish algebraic equality from floating-point evaluation-order differences.
5. **Symbolic cost model** — identify which intermediates and scalar operations are removed or moved.
6. **Hardware questions** — only after the above is fixed, determine whether an exact packed realization maps efficiently to a target GPU.
7. **Boundary experiments** — measure only numerical or hardware boundaries of an already-proved exact form.

## Exactness terminology

`ALGEBRAIC-EXACT` means the two forms define the same real-valued decoded-quantized operator for every admissible input under the frozen representation contract.

It does **not** automatically mean bitwise-identical IEEE floating-point results, because reassociation changes rounding order.

## Repository map

```text
README.md
THEORY.md
REPRESENTATION.md
EXACT_IDENTITY.md
COST_MODEL.md
NUMERICAL_CONTRACT.md
HARDWARE_QUESTIONS.md
reference/
  quant.py
  decode_then_score.py
  direct_packed_score.py
tests/
  test_representation.py
  test_exact_identity.py
  test_numerical_equivalence.py
experiments/
  README.md
```

## Experimental rule

No experiment in this repository may be used to test whether the algebraic identity is true. That is a theorem-level question.

Experiments may test only:

- whether a concrete floating-point implementation satisfies a declared numerical contract;
- whether an exact packed lowering reaches the intended machine primitive;
- where the exact packed realization ceases to outperform a decode-first implementation.

## Reproducible reference checkpoint

See [reference findings](analysis/REFERENCE_FINDINGS.md) for the pinned source SHA, raw run, numerical observations and remaining gates. Install the recorded environment and run `python experiments/verify_reference.py` from a clean worktree. No GPU performance claim is made.
