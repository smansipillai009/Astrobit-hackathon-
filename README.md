# Exoplanet Detection Pipeline — Computational Astronomy Hackathon (IIT Tirupati)

Detects and characterizes transiting exoplanets in raw Kepler SAP-flux
photometry. Built against the schema in `Problem_Statement.pdf` and
`Submission_Format.pdf` before the real data packs were available —
point it at `train_pack/`, `dev_pack/`, `private_pack/` once you have them.

## Pipeline

```
raw parquet (time, flux, flux_err, quality, quarter)
  -> quality masking + per-quarter iterative detrending   (pipeline/detrend.py)
  -> coarse-to-fine Box Least Squares search               (pipeline/bls_search.py)
  -> vetting features: odd/even, secondary eclipse,
     per-quarter recurrence                                (pipeline/vetting.py)
  -> feature row per star                                  (pipeline/features.py)
  -> trained classifier -> calibrated confidence           (train_classifier.py / predict.py)
  -> threshold tuned on dev_pack                            (evaluate_on_dev.py)
  -> submission CSV, validated against the spec             (make_submission.py)
```

## How to run

**0. Install deps** (pinned versions — see `requirements.txt`):
```
pip install -r requirements.txt
```

### Expected data layout

For the config-driven workflow, keep the pipeline and packs in this relative
layout:

```
workspace/
├── private_pack/                         # 87 parquet files (including nested folders)
└── Initial docs/
    ├── train_pack/
    │   ├── train_labels.csv
    │   ├── train_truth.csv
    │   └── train/*.parquet
    ├── dev_pack/
    │   ├── dev_labels.csv
    │   ├── dev_truth.csv
    │   └── dev/*.parquet
    └── exo_pipeline (1)/exo_pipeline/
        ├── config.yaml
        └── main.py
```

The paths in `config.yaml` are relative to that file. Run the reproducible
workflow from the pipeline directory:

```powershell
python main.py --config config.yaml
```

The checked-in config uses a fixed threshold selected with a per-bin weighted
dev objective: shallow and earth-analog injected positives receive 2x weight,
while mid and deep positives receive 1x. This prioritizes the competition's
harder weighted bands and avoids selecting an unstable aggregate F1 threshold.

Candidate features also include a soft, independent Lomb-Scargle cross-check
at the BLS period. It is used by the classifier rather than as a hard cut
because transit signals are not sinusoidal and can have weak Lomb-Scargle
power.

**1. Sanity check the pipeline itself** (no real data needed — this is the
injection-recovery check the problem statement explicitly asks you to do
before trusting the pipeline on real stars):
```
python test_synthetic.py
```
Should print `PASS: period matches truth`. If it doesn't, something in
your environment differs from what this was built against — don't move
on until it passes.

**2. Train the classifier on `train_pack`:**
```
python train_classifier.py \
    --pack_dir train_pack/ \
    --labels train_pack/train_labels.csv \
    --truth train_pack/train_truth.csv \
    --out model.pkl \
    --features_out train_features.csv
```
This runs the full detrend+BLS+vetting pipeline over all 269 training
stars, derives the target from injected membership in `train_truth.csv`,
and trains a gradient-boosted
classifier with 5-fold cross-validated ROC-AUC / PR-AUC printed to
stdout so you know if it's actually learning anything before moving on.

**3. Predict on `dev_pack` and tune your threshold:**
```
python predict.py --pack_dir dev_pack/ --model model.pkl --out predictions_dev.csv
python evaluate_on_dev.py --pred predictions_dev.csv --labels dev_pack/dev_labels.csv --truth dev_pack/dev_truth.csv
```
`evaluate_on_dev.py` reports the dev ROC-AUC, PR-AUC, per-bin recall, and
characterization metrics. For the reproducible submission, use the checked-in
`submission.fixed_threshold: 0.25` from `config.yaml`; it was selected with
the weighted objective described above rather than plain aggregate F1.

**4. Predict on `private_pack` and build the submission:**
```
python predict.py --pack_dir private_pack/ --model model.pkl \
    --out predictions_private.csv --threshold 0.25
python make_submission.py --pred predictions_private.csv --team yourteamname
```

The recommended workflow is the one-command reproducible run, which trains,
evaluates, applies the fixed weighted threshold, and validates the final
submission:

```bash
python main.py --config config.yaml
```

The final checked-in run processes 87 private stars and writes
`outputs/submission_lalaland.csv`. If you run the individual commands instead,
keep the model, prediction, and submission paths consistent with the config.

See [writeup.md](./writeup.md) for the methodology and known limitations.
`make_submission.py` runs the organizers' own validation assertions
(exact row count, star_id format, confidence range, no flat confidence
column, etc.) before writing the file — fix anything it complains about.

## Data contract

The parquet files use `KIC_########` filename stems and the CSV files use
numeric `kepid` values. Training and dev targets come from membership in
the corresponding `*_truth.csv` file where `injected == 1`; catalog
`label` and `koi_disposition` values are not used as targets or features.
Pack discovery is recursive so both flat packs and the provided nested
`train/` and `dev/` layouts work.

## Known limitations / where to spend extra time if you have it

1. **Depth/duration precision near the noise floor.** Injection testing
   (`test_synthetic.py`) showed period recovery is solid (within 0.1% of
   truth, well inside the competition's 2% tolerance), but at some noise
   draws the BLS duration grid can lock onto a wider-than-true box,
   diluting the depth estimate. If you have time: after fixing
   period+t0, do a proper least-squares trapezoid-transit refit (e.g.
   `scipy.optimize.curve_fit`) instead of relying on the BLS duration
   grid's own depth estimate — this is the single highest-value
   improvement for your characterization score.
2. **No GP-based detrending.** The current detrender is a fast iterative
   median-filter approach. A Gaussian Process (e.g. `celerite2`) fit to
   the out-of-transit flux would likely recover more of the shallow
   Earth-analog signals the difficulty breakdown rewards, at the cost of
   more compute per star — worth trying if step 2's classifier
   cross-val AUC looks weak on the shallow-depth subset.
3. **No centroid-shift or catalogue cross-match.** Both are required for
   the optional "Ultimate Challenge" bonus (independent candidate). Not
   attempted here since it needs the public KOI/TOI/EB catalogues and is
   explicitly a separate, optional, human-judged track — worth doing
   only after the core submission is solid and validated.
4. **Confidence calibration.** The classifier's raw `predict_proba` is
   used as-is. If cross-val shows it's poorly calibrated (check a
   reliability diagram), wrap it in
   `sklearn.calibration.CalibratedClassifierCV` — a few extra lines, and
   PR-AUC is a big chunk of your score.

## Files

```
pipeline/
  data_io.py      - parquet loading + quality masking
  detrend.py       - iterative per-quarter detrending
  bls_search.py    - coarse-to-fine BLS search
  vetting.py       - odd/even, secondary eclipse, quarter recurrence
  features.py      - feature-row assembly
  run.py           - ties one star's full pipeline together
test_synthetic.py   - injection-recovery sanity check (run this first)
train_classifier.py - train on train_pack
predict.py           - run trained model on dev_pack / private_pack
evaluate_on_dev.py   - threshold tuning + characterization scoring on dev_pack
make_submission.py   - format + validate the final submission CSV
requirements.txt     - pinned dependencies
```
