# Exoplanet Detection Pipeline — Computational Astronomy Hackathon (IIT Tirupati)

Detects and characterizes transiting exoplanets in raw Kepler SAP-flux
photometry. Built against the schema in `Problem_Statement.pdf` and
`Submission_Format.pdf` 

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
Astrobit-hackathon-LaLaLand-2026/
├── __pycache__/
├── starter_notebook  
├── private_pack/                         # 87 parquet files (including nested folders)
├── train_pack/
│   ├── train_labels.csv
│   ├── train_truth.csv
│   └── train/*.parquet
├── dev_pack/
|   ├── dev_labels.csv
|   ├── dev_truth.csv
|   └── dev/*.parquet
|── pipeline/
|   ├──__pycache__/
|   └── __init__
|   └── bls_search.py
|   └── data_io.py
|   └── detrend.py
|   └── features.py
|   └── run.py
|   └── vetting.py
├── outputs/
|   ├── model.pkl
|   ├── predictions_dev.csv
|   └── predictions_private.csv
|   └── predictions_private_t005.csv
|   └── predictions_private_t035.csv
|   └── submission_lalaland.csv
|   └── train_features.csv
├── .gitignore
├── writeup.md
├── README.md
├── requirements.txt
├── config.yaml
├── evaluate_on_dev.py
├── main.py
├── make_submission.py
├── predict.py 
├── test_synthetic.py  
├── train_classifier.py



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

## Data contract

The parquet files use `KIC_########` filename stems and the CSV files use
numeric `kepid` values. Training and dev targets come from membership in
the corresponding `*_truth.csv` file where `injected == 1`; catalog
`label` and `koi_disposition` values are not used as targets or features.
Pack discovery is recursive so both flat packs and the provided nested
`train/` and `dev/` layouts work.

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

## Author
```
Mansi Pillai  
3rd Year 
Dept. of Mechanical Engineering
IET DAVV Indore
```
