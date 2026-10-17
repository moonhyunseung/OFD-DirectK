# Symbolic cost model

This file counts algebraically visible logical work and intermediates. It does **not** claim GPU instruction counts or latency.

Let

```text
r = query heads per KV head
L = active tokens
G = quantization groups
m = coordinates per group
d = G*m
```

Current geometry:

```text
r=6, G=8, m=32, d=256.
```

## 1. Decode-first logical path

The reference path logically constructs

```text
Khat[t,g,j] = S[t,g] * (C[t,g,j]-8)
```

for `L*G*m` coordinates, then evaluates

```text
Z[h,t] = alpha * sum_g sum_j Q[h,g,j] * Khat[t,g,j].
```

Logical dense-decoded state size:

```text
L*G*m = L*d
```

floating values per KV head.

## 2. Direct offset-binary path

DirectK evaluates

```text
A[h,g]   = sum_j Q[h,g,j]
R[h,t,g] = sum_j Q[h,g,j] * C[t,g,j]
Z[h,t]   = alpha * sum_g S[t,g] * (R[h,t,g]-8*A[h,g]).
```

The dense `Khat` intermediate disappears.

`A[h,g]` depends on the current query but not on token `t`, so it is computed once per decode query and reused across all active tokens.

## 3. Scale-placement change

Decode-first semantics attach scale to each decoded K coordinate:

```text
L*G*m
```

logical scale applications.

The direct normal form applies scale after the width-`m` reduction, in `(h,t,g)` space:

```text
r*L*G
```

logical scale applications.

For the current geometry,

```text
(L*G*m)/(r*L*G) = m/r = 32/6 = 16/3 ~= 5.33.
```

This is a symbolic reduction in the space at which scale is applied. It is **not** a claim that a compiled kernel executes exactly 5.33x fewer machine instructions.

## 4. Zero-point work

Decode-first logical semantics expose subtraction of 8 at each code coordinate.

Direct offset-binary form instead uses

```text
A[h,g] = sum_j Q[h,g,j]
```

and the correction

```text
-8*A[h,g]
```

inside the reduced group expression.

The query-dependent width reduction for `A` is only

```text
r*G*m
```

per decode query and is independent of `L`.

Signed recoding removes this explicit arithmetic correction entirely at the algebraic-expression level, replacing offset-binary code interpretation with an exact 4-bit signed interpretation.

## 5. Core bilinear work remains

The generic direct contraction

```text
R[h,t,g] = sum_j Q[h,g,j] * C[t,g,j]
```

still contains the same width-`m` coordinate interactions across `(h,t,g)`:

```text
r*L*G*m.
```

Therefore DirectK's proven algebraic gain is currently:

- elimination of dense dequantized K materialization;
- movement of scale work into a smaller reduced space;
- reduction/reorganization of zero-point work;
- exposure of packed-code normal forms.

It does **not** yet establish a reduction in the generic bilinear coordinate interaction count.

## 6. Memory-state distinction

The fact that `Khat` is mathematically unnecessary means an exact implementation may avoid storing or materializing an `L*d` dense decoded tensor.

It does not by itself determine:

- whether an implementation would otherwise materialize K in global memory, shared memory, registers, or only transient instructions;
- the number of actual HBM bytes transferred;
- cache reuse;
- unpack bandwidth;
- register pressure.

Those quantities belong to hardware mapping.

## 7. Hardware question derived from the symbolic model

The correct performance question is not whether the identity holds. It is:

```text
Does a machine realization of the packed normal form save more decode/materialization/scalar work than it adds in unpack, conversion, mixed-type contraction, and scheduling overhead?
```

The crossover of that inequality is an empirical hardware boundary to be studied later.
