"""
Score predictions.csv (from predict.py, run on dev_pack) against
dev_labels.csv / dev_truth.csv, and sweep the confidence threshold to
find the one that maximizes F1 -- use THAT threshold for the private_pack
run, not the default 0.5.

Usage:
    python evaluate_on_dev.py --pred predictions_dev.csv --labels dev_labels.csv \
        --truth dev_truth.csv

The positive target is membership in dev_truth.csv, not the catalog label.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                              average_precision_score, precision_recall_curve)

TRUTH_ID_COL = "kepid"
TRUTH_PERIOD_COL = "period_days"
TRUTH_DEPTH_COL = "depth_ppm"


def prediction_kepid(star_id):
    stem = str(star_id)
    return int(stem[4:]) if stem.startswith("KIC_") else int(stem)


def find_best_threshold(confidence, target):
    """Return the F1-maximising threshold using the documented sweep."""
    best_f1, best_t = -1.0, 0.5
    for threshold in np.arange(0.05, 0.96, 0.05):
        predicted = (confidence >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            target, predicted, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(threshold)
    return best_t, best_f1


def find_best_weighted_threshold(confidence, target, bins=None, bin_weights=None,
                                  true_prevalence=None, rate_tolerance=0.15):
    """Select weighted recall/F1 while constraining the predicted rate."""
    bin_weights = bin_weights or {}
    confidence = np.asarray(confidence, dtype=float)
    target = np.asarray(target, dtype=int)
    bins = np.asarray(bins) if bins is not None else None
    candidates = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        predicted = confidence >= threshold
        tp = int((predicted & (target == 1)).sum())
        fp = int((predicted & (target == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        positive_idx = np.where(target == 1)[0]
        if bins is None:
            weights = np.ones(len(positive_idx))
        else:
            weights = np.array(
                [bin_weights.get(bins[i], 1.0) for i in positive_idx],
                dtype=float,
            )
        weighted_recall = (
            float((weights * predicted[positive_idx]).sum() / weights.sum())
            if weights.sum() else 0.0
        )
        weighted_f1 = (
            2 * precision * weighted_recall / (precision + weighted_recall)
            if precision + weighted_recall else 0.0
        )
        candidates.append((
            float(threshold), weighted_f1, precision, weighted_recall,
            float(predicted.mean()),
        ))
    pool = candidates
    if true_prevalence is not None:
        constrained = [
            item for item in candidates
            if abs(item[4] - true_prevalence) <= rate_tolerance
        ]
        if constrained:
            pool = constrained
    return max(pool, key=lambda item: item[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--truth", required=False, default=None)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    labels = pd.read_csv(args.labels)
    pred[TRUTH_ID_COL] = pred["star_id"].map(prediction_kepid)
    truth = pd.read_csv(args.truth) if args.truth else None
    positive_ids = set(truth.loc[truth["injected"].astype(bool), TRUTH_ID_COL]) if truth is not None else set()
    df = pred.merge(labels[[TRUTH_ID_COL]], on=TRUTH_ID_COL, how="inner")
    y_true = df[TRUTH_ID_COL].isin(positive_ids).astype(int).to_numpy()
    conf = df["confidence"].to_numpy()

    print(f"ROC-AUC:  {roc_auc_score(y_true, conf):.3f}")
    print(f"PR-AUC:   {average_precision_score(y_true, conf):.3f}")
    print()
    print("Threshold sweep (pick the F1-maximizing one for your private_pack run):")
    best_f1, best_t = -1, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred = (conf >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <-- best so far"
        print(f"  t={t:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}{marker}")
    print(f"\nBest threshold: {best_t:.2f} (F1={best_f1:.3f})")

    if truth is not None:
        truth_bins = truth[[TRUTH_ID_COL, "bin"]].drop_duplicates(TRUTH_ID_COL)
        binned = df.merge(truth_bins, on=TRUTH_ID_COL, how="inner")
        binned["target"] = binned[TRUTH_ID_COL].isin(positive_ids).astype(int)
        binned["predicted"] = (binned["confidence"] >= best_t).astype(int)
        print("\nRecall by injected difficulty bin:")
        for bin_name, group in binned[binned["target"] == 1].groupby("bin"):
            recall = float(group["predicted"].mean()) if len(group) else 0.0
            print(f"  {bin_name}: {recall:.1%} ({int(group['predicted'].sum())}/{len(group)})")

        truth_characterization = truth[
            [TRUTH_ID_COL, TRUTH_PERIOD_COL, TRUTH_DEPTH_COL]
        ].rename(columns={
            TRUTH_PERIOD_COL: "truth_period_days",
            TRUTH_DEPTH_COL: "truth_depth_ppm",
        })
        merged = df[df.prediction == 1].merge(
            truth_characterization, on=TRUTH_ID_COL, how="inner"
        )
        if len(merged):
            aliases = [1.0, 2.0, 0.5, 3.0, 1 / 3]
            def period_ok(row):
                for m in aliases:
                    if abs(row["period"] - row["truth_period_days"] * m) / (row["truth_period_days"] * m) < 0.02:
                        return True
                return False
            merged["period_correct"] = merged.apply(period_ok, axis=1)
            merged["depth_rel_err"] = (
                (merged["depth_ppm"] - merged["truth_depth_ppm"]).abs() / merged["truth_depth_ppm"]
            )
            print(f"\nCharacterization (on {len(merged)} true-positive-with-truth stars):")
            print(f"  period correct (within 2%, incl. aliases): "
                  f"{merged['period_correct'].mean():.1%}")
            print(f"  median depth relative error: {merged['depth_rel_err'].median():.2f}")


if __name__ == "__main__":
    main()
