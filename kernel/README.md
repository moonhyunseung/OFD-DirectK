# Kernel gate

Optimized kernels enter this directory only after the representation, algebraic, and numerical contracts are frozen.

Every candidate must declare:

- exact source/target representation;
- offset-binary or signed-recoded code path;
- accumulation dtype/order;
- whether dense K is reconstructed internally;
- intended machine primitive and tile geometry;
- symbolic work/intermediate reduction;
- failure/crossover hypothesis.

A kernel that reconstructs dense K internally is a baseline or diagnostic, not a DirectK implementation.