<!--
  The Results tab for v2. Same format as v1.md. Update this once the run finishes.
-->

# Results
<!-- id: results -->

> **Pending.** v2 is training: 30 epochs of 56 steps, batch 512, on one Mac. The numbers go here when it finishes.

## What will be measured
<!-- id: plan -->

- **Retrieval**, exactly as for v1: each of the 1,014 validation photos against all 5,070 validation captions, R@1, R@5 and R@10 in both directions. The photo is shown whole at test time, with no hiding. v1's numbers are the bar: 14.6% R@1 from photo to caption, 10.7% from caption to photo.
- **The two baselines in the repo**: always predicting the average target, and a ridge regression from the averaged V-JEPA 2 tokens. v2's predictor has to beat both to be worth its 11.3M weights.
- **The spread of its guesses** (`pred_std`) next to the spread of the targets, every step. If the guesses all shrink toward one point, it is predicting the average caption and nothing else.
