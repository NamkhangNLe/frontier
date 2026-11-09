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
Executed 2026-11-09, torch 2.15.0+cpu.

## Phase A, Session 2

See `notebooks/session-02.ipynb`: trained the addition transformer on
CPU. Two smaller siblings of the ~10.7M skeleton, 400 steps each: small
(603k params) final loss 1.3444, medium (1.79M) final loss 1.1496, both
1% exact-match on 200 held-out problems. Bigger is better on loss, but
neither learns 3-digit addition in 400 steps. The ~10.7M run waits for
a GPU. Write-up in `writeups/session-02-writeup.md`.
