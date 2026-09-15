"""
Load raw Kepler SAP-flux light curves.

Expected parquet schema (per Problem_Statement.pdf):
    time      - days
    flux      - raw SAP flux
    flux_err  - uncertainty per measurement
    quality   - mission quality bitmask, 0 = good
    quarter   - observing quarter identifier
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_star(parquet_path, quality_mask=True):
    """Load one star's light curve.

    Returns a dict of numpy arrays: time, flux, flux_err, quality, quarter.
    If quality_mask=True, cadences with quality != 0 are dropped (this is
    the mission's own bad-data flag: cosmic rays, thruster firings, safe
    modes, etc. -- keeping them in will corrupt detrending and BLS).
    """
    df = pd.read_parquet(parquet_path)
    required = {"time", "flux", "flux_err", "quality", "quarter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} missing columns: {missing}")

    df = df.dropna(subset=["time", "flux", "flux_err", "quality", "quarter"])
    if quality_mask:
        df = df[df["quality"] == 0]

    df = df.sort_values("time")
    return {
        "time": df["time"].to_numpy(dtype=float),
        "flux": df["flux"].to_numpy(dtype=float),
        "flux_err": df["flux_err"].to_numpy(dtype=float),
        "quality": df["quality"].to_numpy(dtype=int),
        "quarter": df["quarter"].to_numpy(),
    }


def star_id_from_path(path):
    """STAR_0043.parquet -> STAR_0043"""
    return Path(path).stem


def star_files(pack_dir):
    """Find star parquet files whether the pack is flat or nested."""
    return sorted(Path(pack_dir).rglob("*.parquet"))


def kepid_from_star_id(star_id):
    """Convert a KIC filename stem to the numeric catalog identifier."""
    stem = str(star_id)
    if stem.startswith("KIC_"):
        return int(stem[4:])
    if stem.startswith("STAR_"):
        return int(stem[5:])
    return int(stem)
