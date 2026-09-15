"""
Run the trained pipeline on a pack of stars (dev_pack or private_pack) and
produce per-star predictions: confidence, prediction (0/1), and -- for
prediction=1 -- period/depth/duration.

Usage:
    python predict.py --pack_dir private_pack/ --model model.pkl \
        --out predictions.csv --threshold 0.5

The threshold should be TUNED ON dev_pack (where you have truth) before
you trust it on private_pack -- see the "tune threshold on dev" step in
the README. Don't just ship the default.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.run import process_star
from pipeline.features import FEATURE_NAMES, STELLAR_FEATURE_NAMES
from pipeline.data_io import kepid_from_star_id, star_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="confidence threshold for prediction=1")
    ap.add_argument("--labels", default=None,
                    help="optional labels CSV for stellar-parameter features")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    clf, feature_names = bundle["model"], bundle["feature_names"]
    labels_by_id = {}
    if args.labels:
        labels = pd.read_csv(args.labels)
        missing = set(["kepid"] + STELLAR_FEATURE_NAMES) - set(labels.columns)
        if missing:
            raise ValueError(f"{args.labels} missing columns: {sorted(missing)}")
        labels_by_id = labels.set_index("kepid")[STELLAR_FEATURE_NAMES].to_dict("index")

    files = star_files(args.pack_dir)
    print(f"Found {len(files)} stars in {args.pack_dir}")

    out_rows = []
    for i, path in enumerate(files):
        star_id = os.path.splitext(os.path.basename(path))[0]
        try:
            best, row, meta = process_star(path)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {star_id}: ERROR {e}")
            best, row = None, None

        if row is None:
            feat_row = {k: np.nan for k in FEATURE_NAMES}
            feat_row["found_candidate"] = 0
        else:
            feat_row = dict(row)
            feat_row["found_candidate"] = 1
        for name in STELLAR_FEATURE_NAMES:
            feat_row.setdefault(name, np.nan)
        feat_row.update(labels_by_id.get(kepid_from_star_id(star_id), {}))

        X = pd.DataFrame([feat_row])[feature_names].fillna(-1.0)
        confidence = float(clf.predict_proba(X)[0, 1])
        prediction = int(confidence >= args.threshold)

        out = {
            "star_id": star_id,
            "prediction": prediction,
            "confidence": confidence,
            "period": best["period"] if (prediction and best) else np.nan,
            "depth_ppm": best["depth"] * 1e6 if (prediction and best) else np.nan,
            "duration_hours": best["duration"] * 24 if (prediction and best) else np.nan,
        }
        out_rows.append(out)
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(files)}")

    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Predicted positive: {df.prediction.sum()} / {len(df)}")
    print(f"Confidence range: {df.confidence.min():.3f} - {df.confidence.max():.3f}, "
          f"unique values: {df.confidence.nunique()}")


if __name__ == "__main__":
    main()
