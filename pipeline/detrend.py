"""
Detrending: remove stellar variability + instrumental drift WITHOUT
destroying a genuine shallow transit.

Strategy (per-quarter, iterative fit-mask-refit):
  1. Split by quarter (fixes quarter-to-quarter offsets/discontinuities).
  2. Within each quarter, fit a smooth trend with a median filter of a
     window much longer than any plausible transit duration.
  3. Flag points that sit well below the trend (candidate in-transit
     points) and MASK them out of the fit, then refit. Repeat a few
     times. This is the standard trick to stop the detrender from
     flattening out the very signal you're trying to find.
  4. Divide flux by the final trend -> normalized, ~1.0 out of transit.
"""
import numpy as np
from scipy.ndimage import median_filter


def _seasonal_template_correction(time, norm_flux, season_days=372.0,
                                   n_bins=180, strength=0.5):
    """Remove a conservative repeating four-quarter instrumental template.

    Kepler revisits the same detector geometry about every four quarters.
    A phase-binned median across years estimates that repeatable component
    without fitting a free annual sinusoid that could absorb long transits.
    The correction is blended to preserve sensitivity to near-annual planets.
    """
    if len(time) < 200 or not np.all(np.isfinite(norm_flux)):
        return norm_flux
    phase = np.mod(time - time.min(), season_days)
    edges = np.linspace(0.0, season_days, n_bins + 1)
    indices = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    template = np.full(n_bins, np.nan)
    for i in range(n_bins):
        values = norm_flux[indices == i]
        if len(values) >= 5:
            template[i] = np.nanmedian(values)
    valid = np.isfinite(template)
    if valid.sum() < n_bins // 3:
        return norm_flux
    # Interpolate through gaps and smooth the template on the phase circle.
    filled = np.interp(centers, centers[valid], template[valid])
    extended = np.concatenate([filled[-n_bins // 2:], filled, filled[:n_bins // 2]])
    smooth = median_filter(extended, size=9, mode="nearest")
    template = smooth[n_bins // 2:n_bins // 2 + n_bins]
    template /= np.nanmedian(template)
    correction = np.power(np.maximum(template[indices], 1e-6), strength)
    return norm_flux / correction


def _smooth_trend(time, flux, window_days, mask):
    """Median-filter trend estimate, evaluated only where mask==True,
    then linearly interpolated back onto the full time array."""
    if mask.sum() < 10:
        return np.full_like(flux, np.nanmedian(flux))
    t_valid = time[mask]
    f_valid = flux[mask]
    # approximate window in points, from median cadence spacing
    dt = np.median(np.diff(t_valid)) if len(t_valid) > 1 else 0.02
    win_pts = max(5, int(round(window_days / max(dt, 1e-6))))
    if win_pts % 2 == 0:
        win_pts += 1
    trend_valid = median_filter(f_valid, size=min(win_pts, len(f_valid) - (1 - len(f_valid) % 2)))
    return np.interp(time, t_valid, trend_valid)


def detrend_quarter(time, flux, window_days=1.5, n_iter=4, clip_sigma=4.0):
    """Iterative fit-mask-refit detrending for a single quarter's data."""
    mask = np.ones_like(flux, dtype=bool)
    trend = _smooth_trend(time, flux, window_days, mask)
    for _ in range(n_iter):
        resid = flux - trend
        sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))  # robust MAD sigma
        if sigma <= 0 or not np.isfinite(sigma):
            break
        # mask points that dip well below the trend (candidate transits / outliers)
        # note: only mask *negative* excursions so we don't also chase upward flares
        new_mask = resid > -clip_sigma * sigma
        if new_mask.sum() < 10:
            break
        mask = new_mask
        trend = _smooth_trend(time, flux, window_days, mask)
    return trend


def detrend_lightcurve(time, flux, quarter, window_days=1.5,
                       season_days=372.0, seasonal_strength=0.5):
    """Detrend quarter-by-quarter, return normalized flux (median ~1.0)."""
    norm_flux = np.full_like(flux, np.nan)
    for q in np.unique(quarter):
        qmask = quarter == q
        if qmask.sum() < 20:
            # too little data in this quarter to detrend sensibly; just
            # normalize by its own median
            norm_flux[qmask] = flux[qmask] / np.nanmedian(flux[qmask])
            continue
        trend = detrend_quarter(time[qmask], flux[qmask], window_days=window_days)
        trend = np.where(trend == 0, np.nanmedian(flux[qmask]), trend)
        norm_flux[qmask] = flux[qmask] / trend
    norm_flux = _seasonal_template_correction(
        time, norm_flux, season_days=season_days, strength=seasonal_strength
    )
    # final global sigma-clip of residual outliers (cosmic rays that survived quality mask)
    resid = norm_flux - 1.0
    sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    keep = np.abs(resid) < 8 * sigma if sigma > 0 else np.ones_like(resid, dtype=bool)
    return norm_flux, keep
