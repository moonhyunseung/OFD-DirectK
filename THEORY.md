# Theory

## 1. Object of study

For one GQA KV head, let

```text
Q in R^(r x d)
Khat in R^(L x d)
```

with `r=6`, `d=256`, and feature coordinates factored as

```text
d = G*m = 8*32.
```

The decoded-quantized attention score operator is

```text
B(Q,Khat) = alpha * Q * Khat^T.
```

The repository asks whether the same operator can be evaluated directly from the packed quantized representation of `Khat`, without first constructing the dense floating tensor `Khat`.

## 2. Semantic reference

Quantization is already lossy relative to the pre-quantized FP tensor. Therefore this repository does not claim equivalence to the original FP key.

The semantic reference is the frozen decoder `D4`:

```text
D4(C,S)[t,g,j] = S[t,g] * (C[t,g,j] - 8).
```

An exact DirectK operator `B_direct` must satisfy

```text
B_direct(Q,C,S) = B(Q,D4(C,S))
```

for every admissible `Q`, packed code tensor `C`, and scale tensor `S` under the representation contract.

## 3. Exactness level

The primary theorem is over real arithmetic:

```text
B_direct == B o D4.
```

This is called `ALGEBRAIC-EXACT`.

Floating-point implementations may evaluate the same polynomial in different orders and therefore need not be bitwise identical. Bitwise identity, ULP bounds, or tolerance-based equality are separate numerical properties and are not prerequisites for the algebraic theorem.

## 4. Local-chain property

The rewrite terminates at the same logit tensor `Z` as the decode-first path. Therefore no downstream representation change is introduced:

```text
packed(C,S) --decode--> Khat --score--> Z --mask/softmax--> ...
      |                                ^
      +-------- direct score ----------+
```

The downstream operator receives the same mathematical object `Z`; no inverse transform, reconstruction map, or conjugated softmax is required.

## 5. What the theorem does and does not prove

The theorem proves:

- dense K dequantization is not mathematically necessary;
- per-coordinate scale application may be moved after the width-`m` reduction because the scale is constant within each group;
- the offset `-8` may be represented as a reduced scalar correction;
- both decode-first and direct forms compute the same real-valued decoded-quantized score operator.

The theorem does not prove:

- lower GPU latency;
- lower instruction count on any specific ISA;
- better Tensor Core utilization;
- lower memory traffic in a concrete implementation;
- bitwise equality under arbitrary floating-point evaluation orders;
- a reduction in the generic bilinear coordinate interaction count.

## 6. Research boundary

Only after the algebraic identity and numerical contract are fixed may hardware work ask which exact normal form is cheapest on a target device.
