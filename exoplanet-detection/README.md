# Exoplanet Detection Pipeline — Computational Astronomy Hackathon (IIT Tirupati)

AI/computational pipeline to detect transiting Earth-size to super-Earth
exoplanets (including long-period Earth analogs) in raw Kepler SAP flux
photometry.

## Pipeline

```
raw light curve
  -> Stage A: cleaning + per-quarter normalization      (src/clean.py)
  -> Stage B: detrending / stellar variability removal   (src/detrend.py)
  -> Stage C: period search (coarse BLS/TLS -> fine)      (src/period_search.py)
  -> Stage D: candidate vetting                            (src/vetting.py)
  -> Stage E: feature extraction                            (src/features.py)
  -> Stage F: classifier + confidence calibration            (src/train_model.py, src/predict.py)
  -> Stage G: submission assembly + validation                 (src/make_submission.py)
```

## Important note on ground truth

`train_labels.csv` / `dev_labels.csv` contain **real KOI catalog dispositions**
(`label`, `koi_disposition`) — these are NOT the detection target and are
never used as a model feature or label. The actual target is whether a
star's `kepid` appears in `train_truth.csv` / `dev_truth.csv` (the injected
signal ground truth used for scoring). See `writeup.md` for the full
explanation. Only `kepmag`, `teff`, `logg`, `radius` from the labels files
are used, as auxiliary stellar parameters.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download and unzip the organizer-provided packs, and place them so the
directory looks like this:

```
data/
├── train_pack/
│   ├── train_labels.csv
│   ├── train_truth.csv
│   └── train/*.parquet          (269 files)
├── dev_pack/
│   ├── dev_labels.csv
│   ├── dev_truth.csv
│   └── dev/*.parquet             (89 files)
└── private_pack/
    └── *.parquet                  (87 files, STAR_0000.parquet ... STAR_0086.parquet)
```

`data/` is git-ignored — the raw packs are not committed to this repo.
Paths are configurable in `config.yaml` if your layout differs.

## Reproducing the full pipeline

Single entry point, runs Stage A through G end to end and writes the final
submission file:

```bash
python main.py --config config.yaml
```

This will:
1. Clean + detrend every star in `train/`, `dev/`, and `private/`
2. Run the coarse-to-fine period search and vet candidates
3. Build feature tables for train and dev
4. Train and calibrate the classifier on train, validate on dev
   (prints PR-AUC, F1, and per-difficulty-bin recall)
5. Run inference on `private_pack` and write
   `outputs/submission_<teamname>.csv`
6. Run the organizer's submission-format validation checks automatically

**Expected runtime:** ~X hours on a standard machine (dominated by the
period search stage). Update this once timed on your machine — a judge
reproducing this should not think it has hung.

To run an individual stage instead of the full pipeline (useful while
developing):

```bash
python -m src.clean --config config.yaml
python -m src.detrend --config config.yaml
python -m src.period_search --config config.yaml
python -m src.vetting --config config.yaml
python -m src.features --config config.yaml
python -m src.train_model --config config.yaml
python -m src.predict --config config.yaml
python -m src.make_submission --config config.yaml
```

## Validating a submission manually

```bash
python -m src.make_submission --validate-only outputs/submission_<teamname>.csv
```

Runs the exact pandas assertion checks from the organizer's
Submission_Format spec (column names, 87 rows, unique/valid star_id format,
prediction in {0,1}, confidence in [0,1], required fields present when
prediction=1, empty when prediction=0).

## Repo structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── main.py                    # single entry point, runs Stage A-G
├── src/
│   ├── clean.py
│   ├── detrend.py
│   ├── period_search.py
│   ├── vetting.py
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│   └── make_submission.py
├── notebooks/                  # exploratory analysis, validation-against-known-signal checks
├── models/                      # trained classifier artifact
├── outputs/                      # generated submission csv(s)
└── writeup.md                     # methodology, per-bin validation results, limitations
```

## Team

<!-- team name, members -->

## Methodology write-up

See [writeup.md](./writeup.md) for detrending approach, period-search
strategy and computational shortcuts, vetting logic, feature list, dev-set
validation results by difficulty bin (shallow / mid / deep / earth_analog),
and known limitations.
