# Session 2 write-up

Date: 2026-11-09

## What was built

Trained the addition transformer for real. Session 1 built the ~10.7M skeleton; Session 2 trains it, on CPU, so it trains two smaller siblings of the same architecture and compares them.

- Hardware: CPU only (torch 2.15.0+cpu, 2 threads). The ~10.7M model costs about 6 seconds per step at batch 64 on this machine, so a real run on it would take hours. It waits for a GPU.
- Models: small, d_model 128, 3 layers, 4 heads, d_ff 512, 603,008 params; medium, d_model 192, 4 layers, 4 heads, d_ff 768, 1,791,744 params. Same decoder-only architecture as Session 1.
- Data: 200,000 pre-generated random addition examples (0 to 999 plus 0 to 999), drawn in batches of 64. Loss is next-token cross-entropy on the answer tokens only.
- Recipe: AdamW, lr 3e-4, weight decay 0.01, cosine decay to 10%, 400 steps each.
- Eval: greedy decoding on 200 held-out problems, exact-match on the full answer string.

## Results

| model | params | steps | final loss | exact-match (n=200) |
|---|---|---|---|---|
| small | 603,008 | 400 | 1.3444 | 0.010 (2/200) |
| medium | 1,791,744 | 400 | 1.1496 | 0.010 (2/200) |

Loss curves (answer-token cross-entropy):
- small: 2.9554, 1.5624, 1.4312, 1.3295, 1.3444 at steps 1, 100, 200, 300, 400.
- medium: 2.8746, 1.4611, 1.2631, 1.1725, 1.1496 at steps 1, 100, 200, 300, 400.

The bigger model reaches lower loss (1.15 vs 1.34), so the scaling direction is right. But neither learns 3-digit addition in 400 steps: exact-match is 1% for both. The loss is still falling at step 400, so the runs are undertrained, not incapable.

## A bug caught by the eval

The first version of this session scored the loss wrong: it compared logits[t] to ids[t], which scores copying the current token. The loss looked tiny (0.0065) while exact-match was 0/200. The eval caught it, because a model that only copies cannot generate an answer. The targets are now shifted so logits[t] is scored against ids[t+1], and the numbers above are from the corrected run. This is why the eval exists: a loss number alone can lie.

## What surprised me

How fast the token loss falls and how little that means. After 400 steps the models assign high probability to most answer tokens, yet get only 2 of 200 full answers exactly right. Addition is unforgiving: one wrong digit fails the whole answer, and the carry logic is what the model has not learned yet. Token loss and exact-match measure different things, and for this task exact-match is the honest one.

## Open questions for Session 3

- How many steps does it take for exact-match to lift off? The loss is still falling at step 400, so run longer (a few thousand steps) and watch for the point where carries click.
- Does the ~10.7M model learn it faster per step, and does it reach high exact-match where the small ones stall? That is the GPU run.
- Would a curriculum (1-digit, then 2-digit, then 3-digit) or reversed-digit output help, as the arithmetic literature suggests?
