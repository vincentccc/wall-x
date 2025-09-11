# Fusion Operators (CSRC)

High-performance CUDA kernels for accelerating model training.

## Operators

### Asymmetric Dual Expert GEMM
- `asym_dual_gmm`: Simultaneous matrix multiplication for two experts
- Supports all transpose combinations (NN, TN, NT, TT)

### Token Permutation
- `permute`: Token permutation for MoE routing
- `unpermute`: Token recovery after expert computation
- `unpermute_bwd`: Backward pass for token recovery

### Multimodal RoPE
- `rope`: Rotary Position Embedding forward pass
- `rope_bwd`: RoPE backward pass
- Support for multimodal inputs with configurable sections

## Acknowledgments

The `permute` and `unpermute` operators are adapted from [fanshiqing/grouped_gemm](https://github.com/fanshiqing/grouped_gemm). Thanks for their open-source contributions.
