# frontier-reps

Namkhang Le's frontier research training log.

Baseline: Vlad Feinberg, "How to Land a Frontier Lab Job."
Target: researcher, TBD Labs (Meta's frontier research org).
Stack: PyTorch. Meta built PyTorch and TBD Labs works in it, so the
builds run in PyTorch even though Feinberg's original exercises are
written for JAX. The math and the curriculum transfer unchanged.

## Structure

- `notebooks/` session work, run in Colab, saved back here
- `derivations/` the math, typed up as markdown, checked line by line
- `writeups/` per experiment and per paper pieces, each ending in an artifact

## Rules

- Private by default. Individual pieces go public only with explicit
  approval, one piece at a time.
- Nothing about internal Meta work goes in this repo.
- Reading without an artifact does not count.
- Code for the builds is typed by hand, not pasted.

## Phase A, Session 1

See `notebooks/session-01.ipynb`: PyTorch environment, Scaling Book
Chapter 1 derivations, and step zero of the roughly 10M parameter
transformer that learns arithmetic (dataset, tokenizer, skeleton).
