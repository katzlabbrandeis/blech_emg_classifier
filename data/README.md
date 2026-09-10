# Training Data

This directory holds the raw data used to derive the classifier's training
dataset, plus the training dataset itself. It's a scoped subset of a much
larger raw-data pool that lives on lab storage — only what actually feeds
`src/training_pipeline/generate_training_dataset.py` is included here.

## Layout

```
data/
├── raw/
│   ├── NB27/<basename>/
│   │   ├── emg_env.npy      # EMG envelope, shape (n_tastes, n_trials, n_timepoints)
│   │   ├── <basename>.info  # taste/palatability mapping (JSON)
│   │   └── taste_order.csv  # trial -> taste index (see "Taste order" below)
│   ├── NB32/<basename>/...
│   └── NB34/<basename>/...
├── scores/
│   └── <basename>_scores.csv  # manually-scored movement events (BORIS-style export)
└── training/
    └── fin_training_dataset.pkl  # final labeled feature table (X=features, y=event_codes)
```

## Scope: which sessions, and why

9 sessions across 3 animals — `nb27`, `nb32`, `nb34` (3 test sessions each) —
are included because these are exactly the sessions whose data went into
`fin_training_dataset.pkl`. A 4th animal (`NB35`, 2 sessions) exists in the
original raw-data pool (`/media/fastdata/Natasha_classifier_data`) but was
never used to train the classifier, so it's intentionally excluded.

Two other kinds of raw data from the original pool are also intentionally
excluded, since neither is used to train the classifier:
- **BSA (Bayesian Spectrum Analysis) output** — a frequency-decomposition of
  the EMG signal computed alongside the envelopes. It isn't consumed by
  `generate_training_dataset.py`'s feature-extraction path.
- **Video files** — there aren't any. Movement scoring for this classifier
  was done directly against EMG envelope traces (see the `scores/*.csv`
  files), not via video annotation, despite the "scoring_type: video" label
  baked into some historical column names in the pipeline code.

## Taste order

Tastes are delivered in a pseudorandom order during each session. The
original pipeline (in `NBT_EMB_Classifier_Analyses`) recovered this order by
opening the session's raw `.h5` file and reading its `digital_in` (TTL taste
trigger) channels — but those `.h5` files are 6–17GB each and aren't
otherwise needed for anything in this repo.

`taste_order.csv` (columns: `trial`, `taste`) is a precomputed, one-time
extraction of exactly that derived order, done directly against the live
`.h5` files while they were still accessible on lab storage. It lets
`generate_training_dataset.py` run standalone from this repo without needing
the multi-GB raw `.h5` files. The extraction logic mirrors
`return_taste_orders()` in `src/training_pipeline/utils/extract_scored_data.py`
(that function itself is kept for reference/regeneration, but isn't called
by the training pipeline in this repo).

## Regenerating the training dataset

```bash
pip install -r requirements-training.txt
python src/training_pipeline/generate_training_dataset.py
```

This reads everything under `data/raw/` and `data/scores/`, plus
`src/training_pipeline/artifacts/{nothing_labels.csv,nothing_label_inds.npy}`
(hand-picked "no movement" pseudo-labels, positionally indexed against these
same 9 sessions), and writes a fresh `data/training/fin_training_dataset.pkl`.
`src/training_pipeline/create_classifier.py` then trains and saves an XGBoost
model from that dataset.
