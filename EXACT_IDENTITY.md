# Exact DirectK identity

## 1. Decode-first reference

For query head `h`, token `t`, quantization group `g`, and coordinate `j`, define

```text
Khat[t,g,j] = S[t,g] * (C[t,g,j] - 8).
```

The reference logit is

```text
Z_ref[h,t]
= alpha * sum_g sum_j Q[h,g,j] * Khat[t,g,j].
```

Substitution gives

```text
Z_ref[h,t]
= alpha * sum_g sum_j
    Q[h,g,j] * S[t,g] * (C[t,g,j] - 8).
```

Because `S[t,g]` is constant over `j` within group `g`, distributivity and associativity yield

```text
Z_ref[h,t]
= alpha * sum_g S[t,g]
  * ( sum_j Q[h,g,j] * C[t,g,j]
      - 8 * sum_j Q[h,g,j] ).
```

Define

```text
R[h,t,g] = sum_j Q[h,g,j] * C[t,g,j]
A[h,g]   = sum_j Q[h,g,j].
```

Then

```text
Z_direct[h,t]
= alpha * sum_g S[t,g] * (R[h,t,g] - 8*A[h,g]).
```

Therefore

```text
Z_direct[h,t] = Z_ref[h,t]
```

for every admissible `Q,C,S` over real arithmetic.

Equivalently,

```text
B_direct(Q,C,S) = B(Q,D4(C,S)).
```

This is an `ALGEBRAIC-EXACT` identity.

## 2. Why no downstream repair is required

Both paths terminate at the same mathematical logit tensor `Z`.

```text
decode-first:
(C,S) -> Khat -> Z

direct:
(C,S) --------> Z
```

No changed representation is passed to masking, softmax, or later attention stages. Therefore there is no inverse basis transform, reconstruction step, or conjugated downstream operator to account for.

## 3. Exact consequences

The identity proves that the following dense intermediate is unnecessary as a mathematical object:

```text
Khat in R^(L x G x m).
```

It also proves that:

```text
per-coordinate scale application
    -> post-width-reduction scale application
```

and

```text
per-coordinate subtraction of 8
    -> reduced scalar correction using A[h,g].
```

## 4. Offset-binary normal form

Using stored unsigned nibbles directly gives

```text
Z[h,t]
= alpha * sum_g S[t,g]
  * (R[h,t,g] - 8*A[h,g]).
```

This is useful when the machine consumes unsigned code values naturally.

## 5. Signed-recoded normal form

Let

```text
q[t,g,j] = C[t,g,j] - 8.
```

If the same 4-bit integer is represented in two's-complement form, then

```text
T = C XOR 0x8
```

at the nibble level, and the decoded signed integer represented by `T` is exactly `q`.

The score may therefore be written

```text
Z[h,t]
= alpha * sum_g S[t,g]
  * sum_j Q[h,g,j] * q[t,g,j].
```

No zero-point correction remains in the arithmetic expression. This is not a different model or approximation; it is an exact code-space recoding.

## 6. Packed-pair partition

If a byte contains code coordinates `(c_even,c_odd)`, then splitting query coordinates in the same way gives the coordinate identity

```text
sum_j Q_j C_j
= sum_k Q_(2k) C_(2k)
+ sum_k Q_(2k+1) C_(2k+1).
```

This is exact. It only exposes a packed-byte-aware evaluation form; it does not by itself prove a lower hardware cost.

## 7. Bit-plane identity

For unsigned code

```text
C = B0 + 2*B1 + 4*B2 + 8*B3,
Bk in {0,1},
```

so

```text
Q*C^T
= Q*B0^T + 2*Q*B1^T + 4*Q*B2^T + 8*Q*B3^T.
```

For signed two's-complement integer `q`,

```text
q = B0 + 2*B1 + 4*B2 - 8*B3,
```

so

```text
Q*q^T
= Q*B0^T + 2*Q*B1^T + 4*Q*B2^T - 8*Q*B3^T.
```

Again, these are exact algebraic normal forms, not performance claims.

## 8. What is not implied

Nothing in this proof establishes:

- bitwise floating-point identity;
- a lower bilinear rank for the core `Q`-code contraction;
- a specific GPU instruction sequence;
- reduced latency on H200 or any other accelerator.

Those are separate numerical and hardware questions.
