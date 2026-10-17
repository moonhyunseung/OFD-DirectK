# Numerical contract

## 1. Algebraic identity vs floating-point execution

The DirectK derivation is exact over real arithmetic:

```text
B_direct(Q,C,S) = B(Q,D4(C,S)).
```

A concrete floating-point implementation may reassociate operations. IEEE floating-point arithmetic is not distributive/associative in the same way as real arithmetic, so two implementations of the same polynomial need not be bitwise identical.

Therefore this repository distinguishes:

- `ALGEBRAIC-EXACT`: same real-valued decoded-quantized operator;
- `BITWISE-EXACT`: identical floating-point bit patterns under a specified dtype/order;
- `NUMERICALLY-EQUIVALENT`: results differ only within a declared floating-point error contract.

Only the first status is currently proved by the algebra.

## 2. Reference numerical path

The reference implementation is decode-then-score:

```text
q = C - 8
Khat = q * S
Z_ref = alpha * Q @ Khat^T
```

The direct implementation is

```text
A = reduce_sum(Q, width=m)
R = contract(Q,C, width=m)
Z_direct = alpha * reduce_group(S * (R - 8*A))
```

The numerical comparison must state explicitly:

- storage dtype of `S`;
- compute dtype for `Q`;
- conversion dtype for code values;
- accumulation dtype;
- reduction order;
- whether FMA is used;
- output dtype.

## 3. Initial test contract

Before hardware-specific kernels, the reference tests should include:

1. high-precision (`float64`) evaluation of both forms to verify the symbolic implementation;
2. the frozen practical dtype path used by the original OFD representation;
3. reporting of max absolute error, max relative error, and optionally ULP distance;
4. adversarial values near rounding boundaries and scale extremes;
5. no claim of bitwise identity unless reduction order and instruction semantics make that statement provable or directly verified.

## 4. Failure interpretation

If `float64` implementations disagree beyond ordinary rounding from the chosen expression order, that is an implementation bug or representation-contract mismatch.

If lower-precision implementations differ slightly while both implement the same real-valued identity, that is a numerical-order issue, not evidence that DirectK is algebraically approximate.

## 5. Promotion rule

A hardware kernel may be called an exact DirectK implementation only when:

- its intended real-valued operator is the proved DirectK identity;
- its numerical behavior is documented against the decode-first reference;
- no additional approximation such as clipping, truncated accumulation, token dropping, or lossy recoding is introduced silently.
