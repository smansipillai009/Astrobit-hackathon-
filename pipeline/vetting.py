"""
Candidate vetting metrics, per the "Candidate vetting" directions in the
problem statement: odd/even depth consistency, secondary eclipse search,
per-quarter recurrence. These become classifier features (features.py)
rather than hard cuts, since a hard cut would throw away real-but-noisy
Earth-analog signals that the difficulty breakdown rewards.
"""
import numpy as np
from astropy.timeseries import LombScargle


def _phase(time, period, t0):
    ph = ((time - t0 + 0.5 * period) % period) / period - 0.5
    return ph  # in [-0.5, 0.5), transit centered at 0


def _in_transit_mask(phase, duration, period):
    half_width = 0.5 * duration / period
    return np.abs(phase) < half_width


def odd_even_depth_diff(time, flux, period, t0, duration):
    """Relative difference between odd- and even-numbered transit depths.
    A large difference is a classic sign of a background eclipsing binary
    at half the reported period, not a genuine planet."""
    epoch = np.round((time - t0) / period)
    phase = _phase(time, period, t0)
    in_tr = _in_transit_mask(phase, duration, period)
    if in_tr.sum() < 4:
        return np.nan
    odd = in_tr & (np.mod(epoch, 2) == 1)
    even = in_tr & (np.mod(epoch, 2) == 0)
    if odd.sum() < 2 or even.sum() < 2:
        return np.nan
    depth_odd = 1.0 - np.median(flux[odd])
    depth_even = 1.0 - np.median(flux[even])
    denom = max(abs(depth_odd), abs(depth_even), 1e-8)
    return float(abs(depth_odd - depth_even) / denom)


def secondary_eclipse_depth(time, flux, period, t0, duration):
    """Depth at phase 0.5 (opposite the primary transit). A significant
    secondary eclipse suggests a stellar binary, not a planet (planets can
    have secondaries but they're far too shallow to see at this SNR)."""
    phase = _phase(time, period, t0 + 0.5 * period)
    in_sec = _in_transit_mask(phase, duration, period)
    out = ~_in_transit_mask(_phase(time, period, t0), duration, period) & ~in_sec
    if in_sec.sum() < 3 or out.sum() < 10:
        return np.nan
    baseline = np.median(flux[out])
    sec_depth = baseline - np.median(flux[in_sec])
    return float(sec_depth)  # relative flux units; convert to ppm by *1e6 upstream


def quarter_recurrence_fraction(time, flux, quarter, period, t0, duration):
    """Fraction of quarters that had transit-window coverage AND show a
    dip there, out of quarters that had coverage at all. Guards against a
    single-quarter instrumental glitch masquerading as a periodic signal."""
    phase = _phase(time, period, t0)
    in_tr = _in_transit_mask(phase, duration, period)
    quarters_with_coverage = np.unique(quarter[in_tr])
    if len(quarters_with_coverage) == 0:
        return np.nan
    hits = 0
    for q in quarters_with_coverage:
        qmask = quarter == q
        out_q = qmask & ~in_tr
        in_q = qmask & in_tr
        if out_q.sum() < 5 or in_q.sum() < 1:
            continue
        local_depth = np.median(flux[out_q]) - np.median(flux[in_q])
        if local_depth > 0:
            hits += 1
    return float(hits / len(quarters_with_coverage))


def lomb_scargle_crosscheck(time, flux, flux_err, period):
    """Measure an independent sinusoidal periodicity at the BLS period.

    Lomb-Scargle is structurally different from the box-shaped BLS model.
    A transit is not sinusoidal, so this is deliberately a soft feature:
    consistent periodic power supports a candidate, while a BLS-only peak
    receives no cross-check boost.
    """
    if not np.isfinite(period) or period <= 0 or len(time) < 20:
        return np.nan, np.nan
    frequency = 1.0 / period
    frequencies = np.array([frequency, 2.0 * frequency])
    finite = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        finite &= np.isfinite(flux_err)
    if finite.sum() < 20:
        return np.nan, np.nan
    y = np.asarray(flux[finite], dtype=float)
    t = np.asarray(time[finite], dtype=float)
    dy = None
    if flux_err is not None:
        dy = np.asarray(flux_err[finite], dtype=float)
        dy = np.maximum(dy, np.nanmedian(dy) * 1e-3)
    try:
        ls = LombScargle(t, y, dy=dy, center_data=True)
        power = np.asarray(ls.power(frequencies, normalization="standard"), dtype=float)
        local = np.linspace(0.8 * frequency, 1.2 * frequency, 15)
        local_power = np.asarray(ls.power(local, normalization="standard"), dtype=float)
    except (ValueError, FloatingPointError):
        return np.nan, np.nan
    if not np.all(np.isfinite(power)) or not np.any(np.isfinite(local_power)):
        return np.nan, np.nan
    local_median = max(float(np.nanmedian(local_power)), 1e-8)
    return float(np.max(power)), float(np.max(power) / local_median)
