<!--
  The Results tab for v1. Same format as v1.md. A paragraph that is only {{name}} is filled in by results.js:
  {{loss}} the loss curves, {{recall}} recall by epoch, {{retrieval}} the final table, {{temperature}} the learned scale.
-->

# Results
<!-- id: results -->

The left panel runs the epoch-59 checkpoint in your browser: 14 million weights, stored as 8-bit integers (14 MB), with no server involved. Pick a photo or drop in your own, then type a caption. Or switch to search, and the caption picks the photo.

The score is the cosine between the two 256-number vectors. On its own it means little, so it is ranked against the 5,070 captions of the 1,014 validation photos: rank 1 means no validation caption fits the photo better.

## The training run
<!-- id: training-run -->

{{loss}}

Training loss is plotted every step, validation loss once per epoch on the 1,014 held-out photos. Guessing at random gives $\ln 1014 \approx 6.92$. Validation bottoms out at **4.67** at epoch 59 and climbs from there, while training loss keeps falling to 0.87: it is memorising the 29,000 training photos.

## Finding the right caption
<!-- id: retrieval -->

{{retrieval}}

Every validation photo is scored against all 5,070 validation captions; it counts as found if one of its own five is in the top $K$. The reverse direction ranks the 1,014 photos for each caption. At random, R@1 would be 0.1%.

{{recall}}

## The learned scale
<!-- id: temperature -->

{{temperature}}

The scale $e^t$ multiplies every score before the softmax. It starts at $1/0.07 \approx 14.3$ and is learned with everything else. It barely moves for ten epochs, then climbs steadily: about 20 at the best checkpoint, 31 by the end. A sharper softmax keeps lowering the training loss, one more way to memorise.
