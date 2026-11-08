<!--
  The Results tab for v2. Same format as v1.md. Update this once v2 is trained.
-->

# Results
<!-- id: results -->

> **Nothing to show yet.** v2 has not been coded or trained. When it is, this tab gets the same live demo as v1, running v2's weights in your browser.

## What will be measured
<!-- id: plan -->

- **Retrieval**, exactly as for v1: each of the 1,014 validation photos against all 5,070 validation captions, R@1, R@5 and R@10 in both directions. v1's numbers are the bar to beat: 14.6% R@1 from photo to caption.
- **The spread of the slow copy's vectors**, every epoch. If it heads toward zero, the run has collapsed.
- **Each of the three losses** separately, to see which game is actually being learned.

## What I expect
<!-- id: expect -->

Filling in blanks gives each encoder something to learn from inside its own photo or caption, rather than only from comparisons across the batch. With 29,000 photos and a from-scratch model, that extra signal is where the gain should come from. If it doesn't beat v1, that is a result too.
