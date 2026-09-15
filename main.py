"""Run the complete train -> dev-tune -> private-submission workflow.

The raw packs are intentionally not bundled with the repository. Configure
their locations in config.yaml, then run:

    python main.py --config config.yaml
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from evaluate_on_dev import find_best_weighted_threshold, prediction_kepid
from pipeline.features import FEATURE_NAMES, STELLAR_FEATURE_NAMES


def _path(config_path, value):
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _run(command, cwd):
    print("$ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def _threshold(predictions, truth, labels, config):
    fixed = config.get("submission", {}).get("fixed_threshold")
    if fixed is not None:
        fixed = float(fixed)
        if not 0.0 <= fixed <= 1.0:
            raise ValueError("submission.fixed_threshold must be between 0 and 1")
        print(f"Using weighted fixed threshold from config.yaml: {fixed:.2f}")
        return fixed

    pred = pd.read_csv(predictions)
    truth_df = pd.read_csv(truth)
    labels_df = pd.read_csv(labels)
    pred["_kepid"] = pred["star_id"].map(prediction_kepid)
    positives = set(truth_df.loc[truth_df["injected"].astype(bool), "kepid"])
    merged = pred.merge(labels_df[["kepid"]], left_on="_kepid", right_on="kepid")
    target = merged["kepid"].isin(positives).astype(int).to_numpy()
    truth_bins = truth_df[["kepid", "bin"]].drop_duplicates("kepid")
    merged = merged.merge(truth_bins, on="kepid", how="left")
    weights = config.get("submission", {}).get("positive_bin_weights", {})
    threshold, weighted_f1, precision, weighted_recall, predicted_rate = (
        find_best_weighted_threshold(
            merged["confidence"].to_numpy(),
            target,
            bins=merged["bin"].to_numpy(),
            bin_weights=weights,
            true_prevalence=float(target.mean()),
            rate_tolerance=0.15,
        )
    )
    print(
        f"Selected weighted dev threshold: {threshold:.2f} "
        f"(weighted F1={weighted_f1:.3f}, P={precision:.3f}, "
        f"weighted R={weighted_recall:.3f}, predicted rate={predicted_rate:.1%})"
    )
    if abs(predicted_rate - target.mean()) > 0.15:
        print(
            f"WARNING: F1-argmax threshold {threshold:.2f} predicts "
            f"{predicted_rate:.1%} positive vs {target.mean():.1%} prevalence; "
            "set submission.fixed_threshold to override it."
        )
    return threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = config_path.parent
    data = config["data"]
    outputs = config.get("outputs", {})
    output_dir = _path(config_path, outputs.get("directory", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _path(config_path, outputs.get("model", "outputs/model.pkl"))
    train_features = _path(config_path, outputs.get("train_features", "outputs/train_features.csv"))
    dev_predictions = _path(config_path, outputs.get("dev_predictions", "outputs/predictions_dev.csv"))
    private_predictions = _path(config_path, outputs.get("private_predictions", "outputs/predictions_private.csv"))
    workers = max(1, int(config.get("runtime", {}).get("workers", 1)))

    train_dir = _path(config_path, data["train_dir"])
    dev_dir = _path(config_path, data["dev_dir"])
    private_dir = _path(config_path, data["private_dir"])
    required = [
        train_dir, _path(config_path, data["train_labels"]),
        _path(config_path, data["train_truth"]),
        dev_dir, _path(config_path, data["dev_labels"]),
        _path(config_path, data["dev_truth"]),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required data paths:\n" + "\n".join(missing))

    python = sys.executable
    train_script = root / "train_classifier.py"
    predict_script = root / "predict.py"
    evaluate_script = root / "evaluate_on_dev.py"
    submit_script = root / "make_submission.py"

    retrain = not model.exists() or not train_features.exists()
    if not retrain:
        import pickle
        with model.open("rb") as handle:
            bundle = pickle.load(handle)
        expected_features = FEATURE_NAMES + STELLAR_FEATURE_NAMES + ["found_candidate"]
        retrain = bundle.get("feature_names") != expected_features
        if retrain:
            print("Model feature schema is stale; retraining with current features.")
    if retrain:
        _run([
            python, str(train_script), "--pack_dir", str(train_dir),
            "--labels", str(_path(config_path, data["train_labels"])),
            "--truth", str(_path(config_path, data["train_truth"])),
            "--workers", str(workers),
            "--out", str(model), "--features_out", str(train_features),
        ], root)

    if retrain or not dev_predictions.exists():
        _run([
            python, str(predict_script), "--pack_dir", str(dev_dir),
            "--model", str(model), "--out", str(dev_predictions),
            "--labels", str(_path(config_path, data["dev_labels"])),
        ], root)
    _run([
        python, str(evaluate_script), "--pred", str(dev_predictions),
        "--labels", str(_path(config_path, data["dev_labels"])),
        "--truth", str(_path(config_path, data["dev_truth"])),
    ], root)
    threshold = _threshold(
        dev_predictions,
        _path(config_path, data["dev_truth"]),
        _path(config_path, data["dev_labels"]),
        config,
    )

    if not private_dir.exists():
        print("private_dir is not present; train/dev workflow completed.")
        return
    _run([
        python, str(predict_script), "--pack_dir", str(private_dir),
        "--model", str(model), "--out", str(private_predictions),
        "--threshold", str(threshold),
    ], root)
    team = config.get("team_name", "astrobit")
    if not re.fullmatch(r"[a-z0-9_]+", team):
        raise ValueError("team_name must contain only lowercase letters, digits, and underscores")
    submission = output_dir / f"submission_{team}.csv"
    _run([
        python, str(submit_script), "--pred", str(private_predictions),
        "--team", team, "--out", str(submission),
    ], root)
    print(f"Submission ready: {submission}")


if __name__ == "__main__":
    main()
