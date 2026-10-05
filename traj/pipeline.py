"""End-to-end pipeline: video + calibration -> trajectories, plot, annotated video."""

from pathlib import Path

from .export import bev_figure, render_video, write_csv
from .geometry import GroundPlane
from .kinematics import add_kinematics, track_summary
from .tracking import track_video


def run(video_path, plane: GroundPlane, out_dir, max_seconds=15.0, stride=1, conf=0.3, progress=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, info = track_video(video_path, max_seconds=max_seconds, stride=stride, conf=conf,
                            progress=(lambda p: progress(0.85 * p, "detecting + tracking")) if progress else None)
    raw[["x_m", "y_m"]] = plane.to_world(raw[["u", "v"]].to_numpy()) if len(raw) else []
    raw = raw[plane.in_range(raw[["x_m", "y_m"]].to_numpy())]
    # Window ~1 s of samples, independent of frame rate and stride.
    samples_per_s = info.fps / stride
    kin = add_kinematics(raw, min_track_len=max(5, round(samples_per_s)),
                         window=max(5, round(samples_per_s)) | 1)
    if progress:
        progress(0.9, "rendering video")
    paths = {
        "csv": write_csv(kin, out_dir / "trajectories.csv"),
        "video": render_video(video_path, raw, kin, out_dir / "annotated.mp4", stride=stride),
        "bev": out_dir / "bev.png",
    }
    bev_figure(kin, plane).savefig(paths["bev"], dpi=120)
    return kin, track_summary(kin), paths
