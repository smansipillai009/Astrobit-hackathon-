"""
Injection-recovery sanity check.

The problem statement says explicitly: "Whenever your pipeline reports
nothing, verify it against a known injected signal from the training set
before concluding the data is empty." This script is that check, using a
synthetic light curve instead of a real star (since we don't have
train_pack yet) so the pipeline's correctness doesn't depend on data
that hasn't been released.

Simulates ~4 years at 29.4-min cadence with:
  - quarter-to-quarter offsets + slow instrumental drift
  - stellar rotational variability (sinusoid + slow spline wiggle)
  - a box-shaped transit with a true period/depth/duration
  - Gaussian photon noise + a few quality-flagged bad cadences

Then runs the full pipeline and checks the recovered period is within
2% of truth (or a 2x/0.5x/3x/(1/3)x alias, matching the scoring rule).
"""
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.run import process_star

rng = np.random.default_rng(42)

CADENCE = 29.4 / (24 * 60)  # days
BASELINE = 4 * 365.25
N = int(BASELINE / CADENCE)

TRUE_PERIOD = 47.3       # days
TRUE_T0 = 12.0
TRUE_DURATION_HR = 5.5
TRUE_DEPTH_PPM = 900.0   # moderately shallow; validates pipeline mechanics.
# NOTE: at much shallower depths (a few hundred ppm, "near the noise
# floor" per the problem statement) a single BLS run WILL produce false
# positives from pure look-elsewhere effect across thousands of trial
# periods/durations. That's expected and is exactly why the classifier
# (train_classifier.py) exists: it learns, from hundreds of labeled train
# stars, which combination of BLS SNR + vetting features actually
# separates real transits from noise, rather than trusting one star's
# raw periodogram peak.


def make_synthetic_star():
    time = np.arange(N) * CADENCE
    quarter = (time // 93.0).astype(int)  # ~quarterly cadence like Kepler

    flux = np.ones(N)
    # quarter offsets
    for q in np.unique(quarter):
        flux[quarter == q] *= 1.0 + rng.normal(0, 0.003)
    # slow instrumental drift per quarter + stellar rotation signal
    flux += 0.0015 * np.sin(2 * np.pi * time / 13.0 + rng.uniform(0, 6))
    flux += 0.0008 * np.sin(2 * np.pi * time / 240.0)
    # photon noise
    flux += rng.normal(0, 0.0004, size=N)

    # inject box transit
    duration_days = TRUE_DURATION_HR / 24.0
    phase = ((time - TRUE_T0 + 0.5 * TRUE_PERIOD) % TRUE_PERIOD) - 0.5 * TRUE_PERIOD
    in_transit = np.abs(phase) < duration_days / 2.0
    flux[in_transit] -= TRUE_DEPTH_PPM * 1e-6

    flux_err = np.full(N, 0.0004)
    quality = np.zeros(N, dtype=int)
    # sprinkle some bad cadences
    bad = rng.choice(N, size=int(0.01 * N), replace=False)
    quality[bad] = 1
    flux[bad] += rng.normal(0, 0.02, size=len(bad))

    df = pd.DataFrame({
        "time": time, "flux": flux, "flux_err": flux_err,
        "quality": quality, "quarter": quarter,
    })
    return df


def period_matches(found, truth, tol=0.02):
    for mult in (1.0, 2.0, 0.5, 3.0, 1 / 3):
        if abs(found - truth * mult) / (truth * mult) < tol:
            return True, mult
    return False, None


def main():
    df = make_synthetic_star()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "STAR_TEST.parquet")
        df.to_parquet(path)
        best, row, meta = process_star(path)

    print("=== Injection-recovery test ===")
    print(f"true period={TRUE_PERIOD} d, true depth={TRUE_DEPTH_PPM} ppm, "
          f"true duration={TRUE_DURATION_HR} hr")
    if best is None:
        print(f"FAIL: pipeline found nothing. meta={meta}")
        sys.exit(1)

    print(f"recovered period={best['period']:.4f} d, "
          f"depth={best['depth']*1e6:.1f} ppm, "
          f"duration={best['duration']*24:.2f} hr, snr={best['snr']:.2f}")
    ok, mult = period_matches(best["period"], TRUE_PERIOD)
    if ok:
        print(f"PASS: period matches truth (alias x{mult})")
    else:
        print("FAIL: recovered period does not match truth within tolerance")
        sys.exit(1)

    print("Feature row for classifier:")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
