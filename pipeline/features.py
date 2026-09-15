"""
Turn (light curve, best BLS candidate) into one feature row for the
classifier that produces the final confidence score.
"""
import numpy as np
from .vetting import (
    odd_even_depth_diff,
    secondary_eclipse_depth,
    quarter_recurrence_fraction,
    lomb_scargle_crosscheck,
)

FEATURE_NAMES = [
    "bls_power", "bls_snr", "depth", "duration_hr", "period",
    "n_transits", "duration_over_period", "odd_even_diff",
    "secondary_depth_ppm", "quarter_recurrence_frac", "depth_snr",
    "duration_at_grid_max", "ls_power_at_bls", "ls_local_power_ratio",
]
STELLAR_FEATURE_NAMES = ["kepmag", "teff", "logg", "radius"]


def build_feature_row(time, flux, quarter, candidate, flux_err=None):
    period = candidate["period"]
    duration = candidate["duration"]
    t0 = candidate["t0"]
    depth = candidate["depth"]

    oe = odd_even_depth_diff(time, flux, period, t0, duration)
    sec = secondary_eclipse_depth(time, flux, period, t0, duration)
    qrec = quarter_recurrence_fraction(time, flux, quarter, period, t0, duration)
    ls_power, ls_ratio = lomb_scargle_crosscheck(time, flux, flux_err, period)

    row = {
        "bls_power": candidate["power"],
        "bls_snr": candidate["snr"],
        "depth": depth,
        "duration_hr": duration * 24.0,
        "period": period,
        "n_transits": candidate["n_transits"],
        "duration_over_period": duration / period if period > 0 else np.nan,
        "odd_even_diff": oe,
        "secondary_depth_ppm": sec * 1e6 if np.isfinite(sec) else np.nan,
        "quarter_recurrence_frac": qrec,
        "depth_snr": candidate.get("depth_snr", np.nan),
        "duration_at_grid_max": float(duration >= (15.0 / 24.0 - 1e-9)),
        "ls_power_at_bls": ls_power,
        "ls_local_power_ratio": ls_ratio,
    }
    return row
