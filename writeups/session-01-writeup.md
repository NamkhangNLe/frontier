# Session 1 write-up

Date: 2026-11-09

## What was built

PyTorch environment verified (torch 2.15.0+cpu, CPU device), dataset and tokenizer for up to 3 digit addition, transformer model skeleton. No training yet, training is Session 2.

- Environment: torch imports, version 2.15.0+cpu, CPU device. A tiny training loop fits y = 2x + 1 in 20 steps: loss 3.876 down to 0.000198, learned weight 1.9933 and bias 0.9923 against the true 2.0 and 1.0.
- Dataset: addition up to 3 digits, generated on the fly, e.g. "864+394=1258". 1,000,000 possible pairs (0 to 999 plus 0 to 999).
- Tokenizer: 15 tokens, digits 0-9 plus '+', '=', and pad/bos/eos. Encode and decode round-trip verified on "123+456=579".
- Model skeleton: decoder-only transformer, d_model 384, 6 layers, 6 heads, d_ff 1536, vocab 15, max_len 32. Parameter count from the code: 10,671,360, about 10.7M. One forward pass on a real batch of 4 runs: logits shape (4, 16, 15), untrained loss 3.0513.

## One worked roofline example

Summary, with the full derivation in derivations/session-01-roofline-and-flops.md:

A 4096x4096 fp32 matmul does 137.4 GFLOP while moving 201.3 MB, an arithmetic intensity of 682.7 FLOP/byte. On an A100 (ridge point 156 FLOP/byte) that is compute bound, attainable 312 TFLOPS, about 0.44 ms. An elementwise add of the same two arrays moves the same 201.3 MB but does only 16.8 MFLOP, intensity 0.083 FLOP/byte, so it is memory bound at 166.7 GFLOPS and 0.10 ms. Same bytes, different roof: intensity decides.

The 6N rule, derived for this model: forward about 2N per token (21.3 MFLOP), backward about 4N (42.7 MFLOP), training about 6N per token (64.0 MFLOP). Training memory for the 10.7M model is about 149.4 MB (bf16 weights 21.3 MB, fp32 master 42.7 MB, AdamW moments 85.4 MB); the optimizer state, not the weights, dominates.

## What surprised me

The untrained model's loss (3.0513) is worse than random guessing over 15 tokens (ln 15 = 2.7081). An untrained model is not a uniform random model: its output distribution is already peaked, just peaked in the wrong places, so it scores worse than chance until training starts moving it.

## Open questions for Session 2

- Can a small transformer learn 3-digit addition at all on CPU in a reasonable number of steps, and what exact-match accuracy does it reach?
- The ~10.7M skeleton needs about 6 seconds per training step at batch 64 on this CPU, so a real run on it would take hours. Session 2 trains smaller siblings and compares them, and the 10.7M run waits for a GPU.
- Does loss on the answer tokens only (masking the problem part) train faster than loss on every token?
