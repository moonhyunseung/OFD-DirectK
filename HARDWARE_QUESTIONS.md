# Hardware questions

No hardware performance claim is established by the current algebra.

This file records only questions that become meaningful **after** the exact DirectK identity is fixed.

## 1. Primitive compatibility

For the core contraction

```text
R[h,t,g] = sum_j Q[h,g,j] * C[t,g,j]
```

or its signed-recoded equivalent, determine:

- which native low-bit dot/MMA primitives can consume the code operand directly;
- whether the Q operand must be converted/decomposed;
- whether signed or unsigned nibble interpretation is cheaper;
- whether two packed nibbles can be reused from one byte load without materializing a widened dense vector;
- whether scale can be fused into the reduction epilogue.

No source-level operation is assumed to lower to a desired instruction until generated PTX/SASS is inspected.

## 2. Offset-binary vs signed recoding

Two algebraically exact forms exist:

```text
A. unsigned code contraction + zero-point correction
B. exact signed recoding + signed code contraction
```

The hardware question is which form minimizes the actual combination of:

- nibble manipulation;
- sign handling;
- correction work;
- operand conversion;
- instruction count;
- register pressure.

## 3. Storage and traffic

The algebra removes dense decoded K as a required intermediate. Hardware analysis must determine whether the baseline actually incurs significant cost from:

- HBM traffic;
- shared-memory traffic;
- register materialization;
- unpack/convert instructions;
- temporary tensor writes.

A reduction in logical intermediate state is not automatically a reduction in HBM bytes.

## 4. Utilization boundary

Current model geometry has `r=6`, which may not align naturally with common matrix-tile dimensions. Exact packed lowering must therefore account for:

- M-dimension underutilization;
- token-axis tile fill;
- group scheduling;
- batch/GQA aggregation;
- occupancy and register limits.

## 5. Crossover experiment

Only after a native exact lowering is verified should performance experiments search for the boundary where

```text
saved decode/materialization/scalar work
>
added packed-unpack/conversion/scheduling overhead.
```

Boundary variables may include context length, batch, GQA ratio, tile shape, and cache state.

## 6. Required evidence before any speed claim

A speed claim requires all of:

1. generated instruction verification;
2. equality/numerical-contract validation against the decode-first reference;
3. matched workload and storage semantics;
4. explicit accounting of setup and conversion work;
5. crossover measurements rather than a single favorable point.
