# Exoplanet Detection Methodology

## Target and data handling

The classifier target is membership in `*_truth.csv` with `injected == 1`.
The catalog disposition columns in `*_labels.csv` are not used as labels or
features. Each parquet light curve is sorted by time, rows with missing
required measurements are removed, and nonzero Kepler quality flags are
masked. The pipeline discovers nested or flat pack layouts from parquet
files.

## Detrending

Each observing quarter is normalized independently. A robust median-filter
trend is fit per quarter, with points below the trend masked and the trend
refit for several iterations. After that, a conservative phase-binned
template with a 372-day period is estimated across the four-quarter Kepler
season cycle and blended into the correction. This targets repeatable
same-detector systematics without fitting a free annual sinusoid. The
normalized flux is then sigma-clipped to remove surviving extreme outliers.
The 1.5-day filter window is longer than the expected transit durations,
reducing transit self-subtraction while removing slow stellar and
instrumental variability.

## Period search and characterization

The search uses Astropy Box Least Squares. A logarithmic coarse period grid
localizes the strongest peaks between 0.5 and 400 days. The highest local
peaks are refined on a window sized from the coarse spacing and the required
resolution relation `dP ~ P^2 / (baseline * duration)`. Duration, epoch, depth,
power, depth SNR, and the number of supporting transits are retained. Candidate
ranking uses depth significance with a soft penalty for too few transits.
The coarse peak pool is also down-ranked near the approximately annual
spacecraft systematic (and quarter-scale harmonics) before fine refinement;
this prevents a recurring 340--370 day artifact from consuming the peak pool.

The synthetic injection-recovery test recovers the 47.3-day injected period
within the competition tolerance (the recovered period was 47.3175 days).
Depth can be biased low when the discrete duration grid selects a wider box;
this is a known limitation and is reported rather than hidden.

## Vetting and classifier

Features include BLS power/SNR, depth, period, duration, transit count,
duration-to-period ratio, odd/even depth difference, secondary-eclipse depth,
per-quarter recurrence, depth SNR, two independent Lomb-Scargle
cross-check features at the BLS period, and safe stellar parameters
(`kepmag`, `teff`, `logg`, `radius`) when available. Vetting statistics are
features rather than hard rejection rules so shallow and Earth-analog
injections are not discarded prematurely. A gradient-boosted classifier is
trained on injected membership and emits a continuous probability. The
threshold is selected with a per-bin weighted objective on the held-out
dev pack. Shallow and earth-analog positives receive 2x weight, while mid
and deep positives receive 1x. The selection is constrained to a predicted
positive rate within 15 percentage points of the dev prevalence, preventing
unstable low thresholds from overwhelming the submission. The final fixed
threshold is 0.25. PR-AUC and per-bin metrics are printed by
`evaluate_on_dev.py`.

Candidates whose duration reaches the 15-hour BLS duration-grid ceiling are
marked with `duration_at_grid_max`. This is a suspicion feature rather than a
hard rejection, allowing the classifier to down-rank duration fits that are
likely too broad while retaining difficult real transits.

## Final dev results

| Metric | Final value |
|---|---:|
| ROC-AUC | 0.649 |
| PR-AUC | 0.490 |
| Weighted threshold | 0.25 |

## Reproducibility and limitations

Run `python main.py --config config.yaml` after placing the packs at the paths
in the config. The entry point reuses generated model/features/predictions,
validates the final six-column submission, and stops cleanly before private
inference if the private pack has not yet been released. A TLS/GP detrending
variant and a trapezoid refit are natural future improvements, but were not
added because the current dependency-light path is already injection-tested.
