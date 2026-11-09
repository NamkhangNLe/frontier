# Session 1 derivations

Date: 2026-11-09
Session goal: roofline fluency plus transformer FLOPs and memory accounting, from foundations.

## 1. Roofline, Scaling Book Chapter 1

Worked example (my own numbers): a single dense matmul, C = A @ B, with A and B both 4096x4096, fp32, on an A100 80GB (312 TFLOPS fp16 tensor, 2 TB/s HBM).

- Matmul shape: 4096x4096 @ 4096x4096, fp32
- FLOPs: 2 * M * N * K = 2 * 4096^3 = 137,438,953,472, about 137.4 GFLOP
- Bytes moved: read A (4096x4096x4 = 67.1 MB), read B (67.1 MB), write C (67.1 MB), total 201.3 MB. This assumes no cache reuse, the worst case for a single matmul with no tiling.
- Arithmetic intensity: 137.4e9 / 201.3e6 = 682.7 FLOP/byte
- A100 ridge point: 312e12 / 2e12 = 156 FLOP/byte
- Verdict: compute bound, because 682.7 > 156. Attainable performance is the peak 312 TFLOPS, and the matmul takes about 137.4e9 / 312e12 = 0.44 ms.

Contrast, a memory-bound op: an elementwise add of two 4096x4096 fp32 arrays.

- FLOPs: 16,777,216 (one add per element)
- Bytes moved: read A, read B, write C, again 201.3 MB
- Arithmetic intensity: 16.8e6 / 201.3e6 = 0.083 FLOP/byte
- Verdict: memory bound, because 0.083 is far below the ridge point of 156. Attainable is 0.083 * 2 TB/s = 166.7 GFLOPS, and the op takes 201.3e6 / 2e12 = 0.10 ms. It moves the same bytes as the matmul but does 8000x less arithmetic per byte, so bandwidth, not compute, is the limit.

## 2. Transformer FLOPs

The model for this session is a decoder-only transformer: d_model 384, 6 layers, 6 heads, d_ff 1536, vocab 16, max_len 32. Parameter count from the code (see notebooks/session-01.ipynb): 10,671,360, about 10.7M.

Where the 6N rule comes from, in my own derivation:

- Forward pass, per token: every weight is used once per token in a matmul, and a matmul costs 2 FLOPs per weight (one multiply, one add). So the forward pass costs about 2N FLOPs per token, where N is the parameter count. For this model, 2 * 10.7M = 21.3 MFLOP per token. The attention scores themselves add a term that grows with sequence length, but at short sequences (here 16) it is small next to the weight terms, so 2N is the right leading order.
- Backward pass: backprop costs about twice the forward pass, because it computes gradients with respect to both the inputs and the weights, so about 4N FLOPs per token. For this model, 42.7 MFLOP per token.
- Total training cost per token: 2N + 4N = 6N. For this model, 6 * 10.7M = 64.0 MFLOP per token. That is the 6N rule: training a transformer costs about 6 FLOPs per parameter per token.

## 3. Memory accounting

For the same 10.7M parameter model:

- Weights: fp32, 10.7M * 4 bytes = 42.7 MB. In bf16, 21.3 MB.
- Activations: the values saved for the backward pass. Order of magnitude, per token per layer, roughly 20 bytes per hidden unit after counting the attention and MLP intermediates, so for a batch of 32 and sequence length 16, about 6 layers * 32 * 16 * 384 * 20 bytes = 23.6 MB. This scales with batch and sequence length, unlike the weights, which is why activation checkpointing matters at scale.
- Optimizer state: AdamW keeps two fp32 moments per parameter (8 bytes) plus an fp32 master copy of the weights (4 bytes), so 10.7M * 12 bytes = 128.1 MB.
- Total training memory, bf16 weights plus fp32 master plus AdamW: 21.3 + 42.7 + 85.4 = 149.4 MB. The optimizer state, not the weights, dominates at this size.

## What surprised me

The elementwise add and the matmul move the exact same 201.3 MB, but the matmul is compute bound and the add is memory bound: arithmetic intensity, not byte count, decides which roof you hit.
