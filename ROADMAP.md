# Roadmap

The repository advances only when the previous stage is closed.

## Phase 1 — Algebra closure

- Freeze source/codomain, shapes, dtypes, and group semantics.
- Maintain both exact normal forms:
  1. offset-binary code contraction + reduced zero-point correction;
  2. exact signed-nibble recoding + signed contraction.
- Derive symbolic operation/intermediate counts without GPU claims.
- Record any lower-bound statement with its exact scope; matrix-rank bounds must not be overstated as general bilinear-complexity bounds.

Exit criterion: every DirectK rewrite has a pointwise algebraic proof or an explicit representation bijection.

## Phase 2 — Numerical contract

- Compare evaluation orders at FP64, FP32, and intended production accumulation precision.
- Separate `ALGEBRAIC-EXACT` from `BITWISE-EXACT`.
- Define tolerances only after measuring rounding behavior of the frozen reference implementations.

Exit criterion: accepted numerical semantics are written before kernel optimization.

## Phase 3 — Hardware lowering

- Enumerate candidate packed primitives and required unpack/recode operations.
- Inspect generated code/instructions.
- Count register/shared-memory/HBM implications.
- Reject lowerings that reconstruct dense K internally.

Exit criterion: at least one representation-native lowering is verified.

## Phase 4 — Boundary experiments

Experiments answer only: **where does an already-proved exact form stop being cheaper?**

Primary variables:

- context length `L`;
- batch size;
- GQA ratio / query heads per KV head;
- tile geometry;
- accumulation dtype;
- inline signed recoding vs offset correction.

Exit criterion: crossover/failure region is mapped, including negative results.

## Phase 5 — Model integration

- Integrate the frozen best exact kernel into a real decode path.
- Compare against the frozen decode-first baseline under matched semantics.
- Record end-to-end and layer-level results without extrapolating beyond measured regimes.

## Phase 6 — Freeze

Freeze claims, commit SHA, source model revision, compiler/toolchain versions, hardware, manifests, and raw results.
