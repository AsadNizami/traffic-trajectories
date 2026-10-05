"""Smoothing and velocity estimation for ground-plane trajectories."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _savgol(values, window, deriv=0, delta=1.0):
    """Local quadratic fit; deriv=1 gives a noise-robust velocity estimate."""
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return np.gradient(values, delta) if deriv else values
    return savgol_filter(values, window, polyorder=2, deriv=deriv, delta=delta)


def add_kinematics(df, min_track_len=10, window=15):
    """Smooth per-track x_m/y_m and add vx, vy (m/s), speed_kmh and heading_deg.

    Expects columns: track_id, t, x_m, y_m. Tracks shorter than `min_track_len`
    samples are dropped (mostly false positives / fragments).
    """
    out = []
    for _, g in df.sort_values("t").groupby("track_id"):
        if len(g) < min_track_len:
            continue
        g = g.copy()
        dt = float(np.median(np.diff(g["t"])))
        x, y = g["x_m"].to_numpy(), g["y_m"].to_numpy()
        g["x_m"], g["y_m"] = _savgol(x, window), _savgol(y, window)
        g["vx"], g["vy"] = _savgol(x, window, 1, dt), _savgol(y, window, 1, dt)
        g["speed_kmh"] = np.hypot(g["vx"], g["vy"]) * 3.6
        g["heading_deg"] = np.degrees(np.arctan2(g["vy"], g["vx"]))
        out.append(g)
    cols = list(df.columns) + ["vx", "vy", "speed_kmh", "heading_deg"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def track_summary(df):
    """One row per track: class, duration, path length, median/95th-percentile speed."""
    if df.empty:
        return pd.DataFrame(columns=["track_id", "class", "duration_s", "distance_m",
                                     "median_kmh", "p95_kmh"])
    rows = []
    for tid, g in df.groupby("track_id"):
        dist = np.hypot(np.diff(g["x_m"]), np.diff(g["y_m"])).sum()
        rows.append({
            "track_id": int(tid),
            "class": g["cls"].mode().iat[0],
            "duration_s": round(g["t"].iat[-1] - g["t"].iat[0], 2),
            "distance_m": round(dist, 1),
            "median_kmh": round(g["speed_kmh"].median(), 1),
            "p95_kmh": round(g["speed_kmh"].quantile(0.95), 1),
        })
    return pd.DataFrame(rows)
