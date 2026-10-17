# Experiments

Experiments in OFD-DirectK are boundary tests for an already-proved exact algebraic form.

## Forbidden

Do not run experiments to ask whether

```text
sum_j Q_j * S * (C_j - 8)
```

equals

```text
S * (sum_j Q_j*C_j - 8*sum_j Q_j)
```

over the reals. That is fixed by algebra.

Do not introduce V-side changes, truncated SVD, token dropping, block dropping, or unrelated basis transforms into this repository.

## Allowed experiment classes

### A. Numerical boundary

Question:

```text
How does floating-point evaluation-order error behave under a declared dtype/accumulation contract?
```

Required outputs:

- max absolute error;
- max relative error;
- dtype and accumulation order;
- adversarial rounding-boundary cases.

### B. Lowering verification

Question:

```text
Does a source-level exact packed contraction lower to the intended machine primitive?
```

Required outputs:

- compiler/toolchain versions;
- generated PTX/SASS or equivalent;
- operand types and tile shape;
- all conversion/unpack instructions.

### C. Hardware crossover

Question:

```text
For a verified exact implementation, where does packed-direct execution stop outperforming decode-first execution?
```

Required variables may include:

- context length;
- batch;
- GQA ratio;
- tile geometry;
- cache state;
- implementation form: offset-binary correction vs signed recoding.

## Mandatory protocol

Every experiment must state:

1. theorem/identity already proved;
2. unresolved numerical or hardware assumption;
3. exact failure/crossover condition;
4. boundary variable;
5. minimal measurement needed;
6. stopping rule;
7. why algebra alone cannot answer the question.
