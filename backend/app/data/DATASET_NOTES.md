# Local training dataset

`app/data/dataset/` is the active training set used by `python -m app.ml.train`
(one subfolder per class, scanned via `torchvision.datasets.ImageFolder`).
`app/data/dataset_extra/` holds a few extra `paint_peeling` photos held back
to avoid re-introducing class imbalance (see below) — move them into
`dataset/paint_peeling/` only alongside a matching increase in the other
classes.

Current counts: 11 spalling / 26 stagnant_water / 25 paint_peeling /
22 cracked_tiles (84 total). `app/ml/train.py` holds out a 15% validation
split and keeps the best-val-accuracy checkpoint.

## Provenance

All images are real photos (not synthetic), pulled from Wikimedia Commons and
Openverse-indexed Flickr CC images, chosen for cross-source *diversity*
(different cameras, lighting, locations, photographers) rather than volume.

## The accuracy story (read this before retraining)

The original model was trained on large single-source Kaggle datasets
(thousands of images per class, one dataset per class) and scored 99.9%
validation accuracy on Kaggle but only **60% (6/10)** on real held-out photos
— it had learned each dataset's visual "fingerprint" (resolution,
compression, color grading) rather than transferable defect features.

Fixing this took several iterations, each measured against a real-photo
held-out eval set (`data/` at the repo root, never used in training):

1. **First rebalance** (balanced small dataset, ~60 images): 90% on the
   original 10-photo eval set — but this was a small, noisy sample; a
   same-data rerun with a different random seed scored only 70%.
2. **Added more data without checking balance** (36 paint_peeling vs.
   10 spalling): accuracy on a newly-expanded 23-photo eval set was only
   **56.5%**. The classifier had started defaulting to whichever class had
   the most images. **Lesson: balance matters more than volume.**
3. **Rebalanced again + fixed random seed** in `app/ml/train.py` for
   reproducibility: 69.6% on the 23-photo eval set.
4. **Removed a batch of generic "crumbling concrete texture" images** added
   to `spalling` — they were visually closer to generic damaged-surface
   texture than to spalling specifically, and were pulling predictions
   toward `spalling` across other classes: 69.6% → still investigating.
5. **Audited training images by eye, not just by search-result title** —
   found 3 `paint_peeling` training images that were actually mislabeled
   (titles said "cracked ___" and did show cracks, not peeling paint) and
   removed them: accuracy improved further.
6. **Audited the eval set the same way** — found 2 `spalling` eval photos
   that were themselves bad examples (one was a road-crack photo, not
   spalling at all; one was a wide bridge shot where no spalling damage was
   actually visible at that zoom) and removed them.

7. **Tried unfreezing and fine-tuning the last block of the backbone**
   (instead of keeping it fully frozen). Held-out accuracy did not improve
   — train accuracy hit 100% while internal validation stayed ~65%, i.e.
   it overfit. Reverted to the frozen backbone; the option is still in
   `app/ml/train.py` (`finetune_last_n_blocks`, default 0) for future
   experimentation with a larger dataset.
8. Repeated retrains on cleaned/rebalanced data all landed at the same
   **81% (17/21)** on the held-out eval set — errors shuffle between
   classes run to run but the overall number is a stable plateau for this
   dataset size and a frozen-backbone linear head.

**Final result: 81% (17/21) on a clean, 21-photo real-world held-out eval
set** — up from the original 60% (6/10). Rerun the eval command in
`backend/README.md` any time to reproduce this.

## Remaining known weak spots (be honest about these in the report)

- `paint_peeling` and `spalling` are still occasionally confused with each
  other and with `stagnant_water` — both are texture-heavy, and stagnant
  water photos with dark/wet-looking surfaces can visually resemble
  discolored/flaking concrete.
- Sample sizes are still small (11-26 images/class); `spalling` in
  particular is thin (11 training / 3 eval) and is where most remaining
  errors are. Real evaluation photos may land outside what this dataset has
  seen.
- The biggest remaining lever is simply more, carefully-curated data per
  class — especially `spalling`. Backbone fine-tuning only helps once
  there's enough data that it stops overfitting.
- Two general lessons worth stating explicitly in the documentation report:
  1. Naively adding more data can *hurt* accuracy if it breaks class
     balance.
  2. Search-result titles are not reliable ground truth — every image was
     eventually eye-checked, and several were mislabeled by their own
     Commons/Flickr titles.

## Backup

`app/models_weights/head_kaggle_original.pt.bak` is the original
Kaggle-trained head (60% real-photo accuracy, thousands of images/class from
a single dataset per class) kept for comparison/rollback. `head.pt` is the
current model (81%).
