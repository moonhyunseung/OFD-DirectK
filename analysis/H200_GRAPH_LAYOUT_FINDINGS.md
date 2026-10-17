# H200: exact cache layout and graph-amortized score evaluation

## Scope and reproducible source

This result isolates physical packed-cache layout while keeping the quantizer, groups, stored scales and decoded-quantized score operator fixed. It is a resident, repeated-operand microbenchmark with B=1, one KV head, six query heads, d=256 and group size 32. It is not a full attention, HBM-throughput or serving result.

- Graph control: source commit e2c9f47d0006702e951452764fd4f6e235554a60, experiments/h200_graph_boundary.py; SHA256 69573f2946d235656e78c6fdf4a0dfeadf2579eb110dd083b853b3d199c54047.
- Layout experiment: source commit 1d25e3350a96d9d6d6e2057b5a2e836fb64c134d, experiments/h200_layout_boundary.py; SHA256 f32e4dd1413aa0d896d22a3ea2dae5d58b9c8fb7cef13f59710d2cc98b2c79ad. The worktree was clean when this run started.
- Zero-floor numerical test: source commit e9e391e93d001555ea013267817ae4f06f3e0e2e, experiments/h200_fp_boundary.py; SHA256 4b263e3e100e77bb182b80805e3a820742df228e8bc31563a368388ddc5b040f.
- Target: NVIDIA H200; torch 2.14.1+cu130; CUDA 13.0; Triton 3.8.0.
- Commands: run each script with --out pointing to a new output directory. Outputs include manifests, per-case numerical records, instruction audit, and raw graph timing samples.

## Representation identity

For original packed bytes P[t,g,b] and stored scales S[t,g], define Pnew[g,b,t]=P[t,g,b] and Snew[g,t]=S[t,g]. The inverse transposes recover every original byte and scale. Both layouts decode

Khat[t,g,j] = S[t,g] * (((P[t,g,floor(j/2)] >> (4*(j mod 2))) & 15) - 8).

Substituting Pnew and Snew gives the same value pointwise. This is a bijective stride permutation, with no changed clipping, groups, scale values, rounding or bit budget. Optional byte XOR 0x88 is the existing exact signed recoding, not requantization.

The candidate forms FP16-exact signed integers, contracts FP16 Q with each group in FP32 Tensor Core accumulation, then multiplies by that group's stored scale before the group sum. It writes only score outputs globally in source; no full decoded scaled K array is created. The compiler's global-store count alone is not a proof about every possible memory alias.

## Numerical and instruction observations

- Layout variants: 50/50 predeclared diagnostic cases passed, including zero-code8, alternating endpoints, maximum and subnormal FP16 scales and non-tile-aligned lengths.
- Signed SIMD and original-layout Tensor Core paths: 256/256 zero-floor tests passed. Cases include subnormal Q, subnormal scales, maximum Q/scales, mixed exponents and cancellation, across random/endpoint/zero codes and lengths 1,17,129,1024.
- That tighter test uses error <= (gamma64_FP32 + gamma256_FP64) * sum_abs_decoded_terms, no absolute floor. Budget coefficient: 3.814711846017449e-06. Maximum observed error/budget ratio: 0.06508297403509988.
- This conditional floating-point diagnostic is not a universal hardware proof or downstream model-quality guarantee.
- Original-layout tensor candidate: 134 registers, zero spills, 5120 shared bytes, 32 static mma.sync occurrences.
- Group-major inline variant: 128 registers, zero spills, 5120 shared bytes, 32 static mma.sync occurrences.
- Group-major pre-recoded signed variant: 142 registers, zero spills, 5120 shared bytes, 32 static mma.sync occurrences.
- The generated PTX has 168 static ld.global occurrences for the original tensor path and 32 for either group-major path. These are static instruction occurrences, not measured dynamic transactions or DRAM-byte counts.
- Nsight Compute 2025.1 was attempted for dram__bytes_read.sum, dram__bytes_write.sum and lts__t_bytes.sum. It failed with ERR_NVGPUCTRPERM; no traffic values are established and no permission settings were changed.

## Independent held-out measurement

The tile was fixed at 64 tokens and four warps before these results. Lengths 2048,16384,65536,262144 and seeds 1741,1742,1743 were held out from the first graph experiment. Each method has eight warmups and 31 CUDA-event samples, with 32 captured operations per replay and randomized method order. TF32 and FP fusion are disabled. Values below are medians of the three per-seed medians, in microseconds, rounded.

| Length | Original tensor layout | Group-major inline | Group-major signed storage | FP32 predecoded oracle | Full transpose cost (inline) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 12.841 | 4.675 | 4.776 | 4.009 | 5.939 |
| 16384 | 23.038 | 5.425 | 5.594 | 14.032 | 16.211 |
| 65536 | 85.719 | 14.428 | 16.555 | 35.427 | 55.668 |
| 262144 | 333.086 | 67.774 | 67.857 | 127.619 | 267.339 |

Group-major inline is about 2.75x/4.25x/5.94x/4.91x faster than the corresponding original-layout tensor path on these four shapes. Against an already decoded FP32 matrix it is slower at 2048 and about 2.59x/2.46x/1.88x faster on the three longer shapes.

The oracle excludes reconstruction and holds a dense FP32 matrix. It is not an FP16 FlashAttention baseline or an optimized quantized-serving engine. Repeated reuse can favor cache-resident packed data; model weights and concurrent KV heads can change that regime. These results do not attribute gains solely to DRAM traffic.

The full initial transpose/recode is measured separately. Incremental append, capacity strides, paged layout and full-model costs are not established by this table. A modeled amortization count is not measured model speedup.

## Decision

Physical layout is a concrete optimization axis worth integrating. Native signed storage is not uniformly beneficial and is not selected solely from representation algebra. Carry the group-major inline candidate into a separately gated, batched, capacity-aware staged integration; retain the negative short-context region and the unavailable-counter limitation. A full-model numerical gate must pass before broader end-to-end performance claims.
