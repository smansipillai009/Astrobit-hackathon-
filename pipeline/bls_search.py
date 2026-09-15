"""
Coarse-to-fine Box Least Squares search.

The problem statement is explicit about why a fixed-resolution grid over
the full period range is infeasible (~14M trials, ~5h/star) and why a
coarse fixed grid silently fails (transit smears across phase and you
detect nothing). The required period step scales as:

    dP ~ P^2 / (baseline * duration)

So we:
  1. Run a CHEAP coarse grid (log-spaced, sparse) across the full range
     to localize candidate peaks in BLS power.
  2. For each peak, run a FINE grid sized by the resolution formula,
     around a narrow window (+/- a few coarse steps) of that peak.
  3. Return period/duration/depth/t0 for each peak, ranked by depth
     significance (depth / depth_uncertainty from BLS's transit-by-
     transit statistics), softly penalized for having very few observed
     transits.

Two pitfalls found and fixed via injection testing (see test_synthetic.py)
before trusting this on real data -- keep these in mind if you extend it:
  - The duration grid must never be allowed to exceed ~0.4x the shortest
    period searched (astropy enforces period > duration everywhere), so
    the period floor is raised to fit the widest duration tested, rather
    than shrinking durations and silently losing long-duration transits.
  - objective="likelihood" is systematically biased toward duration grid
    points wider than the true transit (it maximizes the wrong thing --
    verified: recovered period was correct to 4 decimal places but depth
    came out ~8x too shallow with a box pegged at the max duration).
    objective="snr" does not have this problem and is used throughout.
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares


def _systematic_period_penalty(period):
    """Down-rank common Kepler spacecraft/instrumental timescales.

    This is intentionally a soft penalty: a real planet near one of these
    periods must remain recoverable, but an annual artifact should not
    routinely outrank a well-supported shorter-period transit.
    """
    penalty = 1.0
    if abs(period - 372.0) / 372.0 < 0.12:
        penalty *= 0.03
    if abs(period - 90.0) / 90.0 < 0.05:
        penalty *= 0.65
    if abs(period - 180.0) / 180.0 < 0.05:
        penalty *= 0.65
    # After the seasonal template correction, the dominant negative-only
    # contamination moved into a broad 246--310 day region, with a sharp
    # concentration near 308 days.  Use smooth factors so real planets near
    # the band remain recoverable and the search does not simply create a new
    # hard boundary.
    penalty *= 1.0 - 0.45 * np.exp(-0.5 * ((np.log(period) - np.log(275.0)) / 0.12) ** 2)
    penalty *= 1.0 - 0.35 * np.exp(-0.5 * ((period - 308.0) / 7.0) ** 2)
    return penalty


def _duration_grid(min_hr=1.0, max_hr=15.0, n=8):
    """Duration grid in days, spanning the full physically-plausible range.
    NOT tied to min_period -- the caller raises the period floor instead,
    see coarse_to_fine_bls."""
    return np.linspace(min_hr / 24.0, max_hr / 24.0, n)


def coarse_to_fine_bls(time, flux, flux_err, baseline,
                        min_period=0.5, max_period=400.0,
                        n_coarse=3000, n_peaks=3, n_fine=500,
                        n_pool=30, fine_window_factor=5.0,
                        min_transits=2):
    """Run coarse-to-fine BLS. Returns a list of candidate dicts, best first."""
    durations = _duration_grid()
    # astropy BLS requires every period > every duration tested; raise the
    # effective floor of the period search rather than shrinking durations
    # (shrinking durations starves real long-duration transits of a match).
    eff_min_period = max(min_period, 2.5 * durations.max())
    bls = BoxLeastSquares(time, flux, dy=flux_err if np.all(np.isfinite(flux_err)) else None)

    # --- coarse sweep (log-spaced: equal sensitivity in period ratio) ---
    coarse_periods = np.exp(np.linspace(np.log(eff_min_period), np.log(max_period), n_coarse))
    coarse_result = bls.power(coarse_periods, durations, objective="snr")
    power = coarse_result.power

    # find local peaks in coarse power, excluding array edges
    peak_idx = []
    for i in range(2, len(power) - 2):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            peak_idx.append(i)
    if not peak_idx:
        peak_idx = [int(np.argmax(power))]
    # Apply the systematic-period penalty before selecting the pool. Otherwise
    # a strong annual artifact can consume the entire pool and prevent the
    # true shorter-period peak from ever reaching fine refinement.
    peak_idx = sorted(
        peak_idx,
        key=lambda i: power[i] * _systematic_period_penalty(coarse_periods[i]),
        reverse=True,
    )[:n_pool]

    candidates = []
    seen_periods = []
    for i in peak_idx:
        p0 = coarse_periods[i]
        # de-duplicate peaks that are really the same signal / harmonic cluster
        if any(abs(p0 - sp) / sp < 0.02 for sp in seen_periods):
            continue
        seen_periods.append(p0)

        # local coarse step size, used to size the fine search window
        lo_i, hi_i = max(0, i - 1), min(len(coarse_periods) - 1, i + 1)
        local_step = coarse_periods[hi_i] - coarse_periods[lo_i]
        window = fine_window_factor * max(local_step, 0.01 * p0)

        d_guess = durations[int(np.argmax(bls.power([p0] * len(durations), durations,
                                                      objective="snr").power))]
        # required fine resolution near this peak, from the problem's formula
        dP = max((p0 ** 2) / (baseline * max(d_guess, 1e-3)), 1e-5)
        n_this = int(np.clip(2 * window / dP, 50, n_fine))

        fine_periods = np.linspace(max(eff_min_period, p0 - window), min(max_period, p0 + window), n_this)
        fine_result = bls.power(fine_periods, durations, objective="snr")
        j = int(np.argmax(fine_result.power))

        best_period = float(fine_result.period[j])
        best_duration = float(fine_result.duration[j])
        best_power = float(fine_result.power[j])
        t0 = float(fine_result.transit_time[j])
        stats = bls.compute_stats(best_period, best_duration, t0)

        n_transits = len(stats["transit_times"])
        depth_val, depth_err = stats["depth"]
        depth_snr = float(depth_val / depth_err) if depth_err > 0 else 0.0

        candidates.append({
            "period": best_period,
            "duration": best_duration,
            "t0": t0,
            "depth": float(depth_val),
            "power": best_power,
            "n_transits": int(n_transits),
            "depth_snr": depth_snr,
            # final ranking score: depth significance, softly penalized
            # when very few transits back it up (guards against a single
            # deep noise excursion masquerading as a long-period planet)
            "snr": depth_snr * min(1.0, n_transits / max(min_transits, 1)),
        })
        candidates[-1]["rank_score"] = (
            candidates[-1]["snr"] * _systematic_period_penalty(best_period)
        )

    candidates.sort(key=lambda c: c["rank_score"], reverse=True)
    return candidates[:n_peaks]
