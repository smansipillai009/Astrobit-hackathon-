"""
Build the labeled feature table from train_pack and train a classifier
that turns (BLS + vetting features) into a calibrated confidence score.

Usage:
    python train_classifier.py \
        --pack_dir train_pack/ \
        --labels train_pack/train_labels.csv \
        --truth train_pack/train_truth.csv \
        --out model.pkl \
        --features_out train_features.csv

The training target is derived from train_truth.csv: a kepid is positive
when it appears as an injected signal. The catalog label is intentionally
not used because it describes real KOI dispositions, not this benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.run import process_star
from pipeline.features import FEATURE_NAMES, STELLAR_FEATURE_NAMES
from pipeline.data_io import kepid_from_star_id, star_files

STAR_COL = "kepid"


def _process_file(path):
    star_id = os.path.splitext(os.path.basename(path))[0]
    try:
        _, row, _ = process_star(path)
    except Exception as exc:
        return star_id, None, str(exc)
    return star_id, row, None


def build_feature_table(pack_dir, workers=1):
    rows = []
    files = star_files(pack_dir)
    print(f"Found {len(files)} stars in {pack_dir}")
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_file, path) for path in files]
            results = [future.result() for future in as_completed(futures)]
    else:
        results = [_process_file(path) for path in files]

    for i, (star_id, row, error) in enumerate(results, start=1):
        if error:
            print(f"  [{i}/{len(files)}] {star_id}: ERROR {error}")
        if row is None:
            row = {k: np.nan for k in FEATURE_NAMES}
            row["found_candidate"] = 0
        else:
            row["found_candidate"] = 1
        row[STAR_COL] = kepid_from_star_id(star_id)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(files)}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, help="folder of *.parquet star files (e.g. train_pack/)")
    ap.add_argument("--labels", required=True, help="train_labels.csv path")
    ap.add_argument("--truth", required=True, help="train_truth.csv path")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel stars to process; 4 is a reasonable local setting")
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--features_out", default="train_features.csv")
    args = ap.parse_args()

    feat_df = build_feature_table(args.pack_dir, workers=max(1, args.workers))
    feat_df.to_csv(args.features_out, index=False)
    print(f"Saved raw features to {args.features_out}")

    labels = pd.read_csv(args.labels)
    if STAR_COL not in labels.columns:
        print(f"ERROR: expected '{STAR_COL}' in {args.labels}, found {list(labels.columns)}")
        sys.exit(1)

    truth = pd.read_csv(args.truth)
    if STAR_COL not in truth.columns:
        print(f"ERROR: expected '{STAR_COL}' in {args.truth}, found {list(truth.columns)}")
        sys.exit(1)

    df = feat_df.merge(
        labels[[STAR_COL] + STELLAR_FEATURE_NAMES],
        on=STAR_COL, how="left", validate="one_to_one",
    )
    positive_ids = set(truth.loc[truth["injected"].astype(bool), STAR_COL])
    df["target"] = df[STAR_COL].isin(positive_ids).astype(int)

    model_features = FEATURE_NAMES + STELLAR_FEATURE_NAMES + ["found_candidate"]
    X = df[model_features].fillna(-1.0)
    y = df["target"]

    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"OOF ROC-AUC: {roc_auc_score(y, oof_proba):.3f}")
    print(f"OOF PR-AUC (average precision): {average_precision_score(y, oof_proba):.3f}")

    clf.fit(X, y)
    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "feature_names": model_features}, f)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()
