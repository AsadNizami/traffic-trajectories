import numpy as np
import pandas as pd

from traj.kinematics import add_kinematics, track_summary


def _track(tid, speed_ms, n=50, fps=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    return pd.DataFrame({
        "frame": np.arange(n), "t": t, "track_id": tid, "cls": "car",
        "x_m": 3.0 + rng.normal(0, noise, n),
        "y_m": speed_ms * t + rng.normal(0, noise, n),
    })


def test_constant_velocity_speed():
    kin = add_kinematics(_track(1, 25.0))  # 25 m/s = 90 km/h
    np.testing.assert_allclose(kin["speed_kmh"], 90.0, atol=1e-6)
    np.testing.assert_allclose(kin["heading_deg"], 90.0, atol=1e-6)


def test_noise_is_smoothed():
    kin = add_kinematics(_track(1, 25.0, noise=0.3), window=25)
    assert abs(kin["speed_kmh"].median() - 90.0) < 5.0


def test_short_tracks_dropped():
    df = pd.concat([_track(1, 10.0), _track(2, 10.0, n=5)])
    kin = add_kinematics(df, min_track_len=10)
    assert set(kin["track_id"]) == {1}


def test_summary():
    s = track_summary(add_kinematics(_track(7, 20.0, n=51)))
    row = s.iloc[0]
    assert row["track_id"] == 7 and row["class"] == "car"
    assert abs(row["distance_m"] - 40.0) < 0.1  # 2 s at 20 m/s
    assert abs(row["median_kmh"] - 72.0) < 0.1


def test_empty_input():
    empty = pd.DataFrame(columns=["frame", "t", "track_id", "cls", "x_m", "y_m"])
    assert add_kinematics(empty).empty
    assert track_summary(add_kinematics(empty)).empty
