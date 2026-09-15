"""
Turn predictions.csv (from predict.py, run on private_pack) into the
exact submission format from Submission_Format.pdf, and run the
organizers' own validation assertions before you submit.

Usage:
    python make_submission.py --pred predictions_private.csv \
        --team yourteamname --out submission_yourteamname.csv
"""
import argparse
import numpy as np
import pandas as pd

NEED = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]


def validate(sub):
    assert list(sub.columns) == NEED, f"columns must be exactly {NEED}"
    assert len(sub) == 87, f"expected 87 rows, got {len(sub)}"
    assert sub.star_id.nunique() == 87, "duplicate star_id"
    assert sub.star_id.astype(str).str.fullmatch(r"STAR_\d{4}").all(), "bad star_id format"
    assert sub.prediction.isin([0, 1]).all(), "prediction must be 0 or 1"
    assert sub.confidence.between(0, 1).all(), "confidence must be in [0, 1]"
    pos = sub[sub.prediction == 1]
    for c in ("period", "depth_ppm", "duration_hours"):
        assert pos[c].notna().all(), f"{c} missing for some prediction=1 rows"
    assert (pos.period > 0).all(), "period must be positive"
    print(f"OK — {len(pos)} detections, {len(sub) - len(pos)} non-detections")
    print(f"confidence: min {sub.confidence.min():.3f}, "
          f"max {sub.confidence.max():.3f}, unique {sub.confidence.nunique()}")
    if sub.confidence.nunique() <= 2:
        print("WARNING: confidence column is nearly flat -- this will hurt "
              "PR-AUC / average precision badly. Check your classifier is "
              "actually outputting a probability, not a rounded 0/1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction CSV to validate/assemble")
    ap.add_argument("--team", required=False, help="lowercase, no spaces")
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate-only", action="store_true",
                    help="validate an existing submission without writing a new file")
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    df = df[NEED].copy()
    # spec: leave the last three fields EMPTY (not NA/null/-1/0) when prediction=0
    for c in ("period", "depth_ppm", "duration_hours"):
        df.loc[df.prediction == 0, c] = np.nan

    df = df.sort_values("star_id").reset_index(drop=True)
    validate(df)

    if args.validate_only:
        return
    if not args.team:
        ap.error("--team is required unless --validate-only is used")
    out_path = args.out or f"submission_{args.team}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
