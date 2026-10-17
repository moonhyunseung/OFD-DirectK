# Packed K representation contract

## 1. Geometry

For one KV head:

```text
L = active token count
G = 8 groups
group size m = 32
d = G*m = 256
```

The logical quantized key before byte packing is

```text
q[t,g,j] in {-8,...,7}.
```

The current symmetric absmax quantizer normally emits `{-7,...,7}`; the storage contract still represents `-8`.

## 2. Stored code

The signed integer is stored as an unsigned nibble

```text
C[t,g,j] = q[t,g,j] + 8
```

so

```text
C[t,g,j] in {0,...,15}.
```

Two nibbles are packed per byte, low nibble first.

## 3. Stored scale

Each group shares one scale across its 32 coordinates:

```text
S[t,g] > 0.
```

For the frozen OFD prototype contract,

```text
S[t,g] = max_j |x[t,g,j]| / 7
```

up to the implementation's minimum-scale clamp and stored floating-point precision.

The key algebraic property used by DirectK is only

```text
S[t,g,j] = S[t,g]
```

within a group. The exact formula used to produce the scale is not needed for the direct-score identity once `(C,S)` is fixed.

## 4. Decoder

The semantic decoder is

```text
Khat[t,g,j] = S[t,g] * (C[t,g,j] - 8).
```

All exact executor claims in this repository are relative to this decoded quantized tensor `Khat`, not to the pre-quantized tensor.

## 5. Admissible state set

The packed representation is not treated as a real vector space. It is a representation set

```text
R_K = Codes x Scales
```

with

```text
Codes subset {0,...,15}^(L x G x m)
Scales subset R_+^(L x G).
```

The decoder is a map

```text
D4 : R_K -> R^(L x d).
```

DirectK is an alternative exact evaluator of the downstream score map from `R_K`; it is not a new quantizer.

## 6. Equivalent signed recoding

Nibble values may also be recoded exactly into 4-bit two's-complement form. If `C=q+8`, then the two's-complement nibble representing the same signed integer is

```text
T = C XOR 0x8.
```

For a packed byte containing two nibbles, the equivalent byte recoding is

```text
packed_twos = packed_offset XOR 0x88.
```

This is a code-space bijection and introduces no approximation. Whether offset-binary plus scalar correction or signed recoding is preferable is a later hardware question.
