import numpy as np
from .data_io import load_star
from .detrend import detrend_lightcurve
from .bls_search import coarse_to_fine_bls
from .features import build_feature_row


def process_star(parquet_path, min_period=0.5, max_period=400.0):
    """Full pipeline for one star. Returns (best_candidate_or_None,
    feature_row_or_None, meta) where meta has diagnostics useful for
    debugging (n_points, baseline, etc.)."""
    d = load_star(parquet_path)
    time, flux, flux_err, quarter = d["time"], d["flux"], d["flux_err"], d["quarter"]

    if len(time) < 100:
        return None, None, {"reason": "too few good-quality points", "n_points": len(time)}

    baseline = time.max() - time.min()
    norm_flux, keep = detrend_lightcurve(time, flux, quarter)
    time, norm_flux, flux_err, quarter = time[keep], norm_flux[keep], flux_err[keep], quarter[keep]

    # flux_err isn't detrended; use it as a relative weight only
    rel_err = flux_err / np.nanmedian(np.abs(flux)) if np.nanmedian(np.abs(flux)) else flux_err

    candidates = coarse_to_fine_bls(
        time, norm_flux, rel_err, baseline,
        min_period=min_period, max_period=min(max_period, baseline / 2.5),
    )
    if not candidates:
        return None, None, {"reason": "no BLS candidates", "n_points": len(time)}

    best = candidates[0]
    row = build_feature_row(time, norm_flux, quarter, best, flux_err=rel_err)
    meta = {"n_points": len(time), "baseline": baseline, "all_candidates": candidates}
    return best, row, meta
