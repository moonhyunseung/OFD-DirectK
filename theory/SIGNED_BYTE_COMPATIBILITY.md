# Signed byte path

XOR0x88 independently maps offset codes to signed two’s-complement nibbles. Decoding those nibbles gives exactly C-8, including -8. Therefore substituting the recoded signed integer into each DirectK summand preserves the decoded-quantized operator termwise over the reals. The pinned SignedInt4 codec proves the map; all256 bytes are tested against the tensor implementation. Reassociation is not bitwise equality. This reference still materializes logical codes and has no native packed consumption or performance claim.

The unchanged quantizer can store zero FP16 scale after underflow. This is an explicitly characterized degenerate storage state, separate from strictly positive-scale assumptions. It introduces no new quantization policy.
