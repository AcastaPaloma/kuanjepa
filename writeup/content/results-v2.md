<!--
  The Results tab for v2. Same format as v1.md. A paragraph that is only {{name}} is filled in by results.js:
  {{v2-retrieval}} the final table, {{v2-loss}} the loss curves, {{v2-recall}} recall by epoch, {{v2-spread}} the spread.
-->

# Results
<!-- id: results -->

Thirty epochs, under two hours of masked training on one Mac, and only the 11.3M-weight predictor learning. Every validation photo is shown whole and scored against all 5,070 validation captions, exactly like v1.

{{v2-retrieval}}

v2 finds the right caption first **26.9%** of the time, against v1's 14.6%, and the right photo **34.3%** of the time, against 10.7%. Most of that is the frozen encoders: a plain ridge regression from the averaged V-JEPA 2 tokens to the target gets within 2 points of the predictor at R@1. The predictor's edge is small, 1 to 4 points, but it holds at every $K$. Always guessing the average target lands exactly on chance, as it should.

## The training run
<!-- id: v2-training-run -->

{{v2-loss}}

Training loss is on photos with about two-thirds hidden; validation loss is on whole photos, so it sits lower. Unlike v1, validation flattens out instead of climbing back up: with the encoders frozen there is little to memorise, and the last epoch is the one kept.

{{v2-recall}}

v2 passes v1's final caption → photo score in its third epoch, and its photo → caption score in its seventh.

## Is it collapsing?
<!-- id: v2-spread -->

{{v2-spread}}

How different v2's guesses are from photo to photo, next to how different the targets are. A collapsing model would drift to zero here. v2's guesses settle at about 60% of the targets' spread: it hedges toward the average caption, which is what a squared-distance loss rewards when it is unsure.

## The two demos
<!-- id: demos -->

**Reveal.** Each tile is a cell of the 8 × 8 grid v2 trained on. In the honest version the encoder only ever sees the uncovered tiles, as in training. In the leaky one, the photo was encoded whole and the tiles were hidden afterwards, so every token has already looked at the hidden parts. Across the ten photos, honest v2 puts one of the photo's own captions first after 3 to 20 tiles, 5 for the dog. The leaky version gets there sooner on 8 of the 10.

**Arithmetic.** v2's guesses live in EmbeddingGemma's text space, so a word's embedding can be added to a photo's. Beach → snow turns the dog into dogs in snow, and dog → horse finds horse riders. Dog → cat finds more dogs: Flickr30k has almost no cats to find.
