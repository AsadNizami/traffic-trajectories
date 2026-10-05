"""Outputs: trajectory CSV, bird's-eye-view plot, annotated video."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402

CSV_COLUMNS = ["frame", "t", "track_id", "cls", "x_m", "y_m", "vx", "vy", "speed_kmh", "heading_deg"]
CLASS_COLORS = {"car": "#1f77b4", "truck": "#d62728", "bus": "#ff7f0e",
                "motorcycle": "#9467bd", "bicycle": "#2ca02c", "person": "#8c564b"}


def write_csv(df, path):
    df[CSV_COLUMNS].round(3).to_csv(path, index=False)
    return path


def bev_figure(df, plane=None):
    """Top-down plot of all trajectories in metric road coordinates.

    The longer world axis is drawn horizontally so long, narrow roads stay readable.
    """
    swap = not df.empty and np.ptp(df["y_m"]) > np.ptp(df["x_m"])
    a, b = ("y_m", "x_m") if swap else ("x_m", "y_m")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if plane is not None:
        poly = np.vstack([plane.world_points, plane.world_points[:1]])[:, ::-1 if swap else 1]
        ax.plot(poly[:, 0], poly[:, 1], "k--", lw=0.8, label="calibration area")
    seen = set()
    for tid, g in df.groupby("track_id"):
        cls = g["cls"].mode().iat[0]
        color = CLASS_COLORS.get(cls, "gray")
        ax.plot(g[a], g[b], color=color, lw=1.8, label=None if cls in seen else cls)
        ax.annotate("", (g[a].iat[-1], g[b].iat[-1]), (g[a].iat[-2], g[b].iat[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color))
        ax.annotate(f"#{tid}", (g[a].iat[0], g[b].iat[0]), fontsize=7, color=color)
        seen.add(cls)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{a[0]} [m]")
    ax.set_ylabel(f"{b[0]} [m]")
    ax.set_title(f"Bird's-eye view: {df['track_id'].nunique()} tracks")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    return fig


def render_video(video_path, raw, kin, out_path, stride=1):
    """Draw boxes, IDs, speeds and pixel-space trails onto the processed frames."""
    info = sv.VideoInfo.from_video_path(video_path)
    out_info = sv.VideoInfo(width=info.width, height=info.height, fps=max(1, round(info.fps / stride)))
    speeds = kin.set_index(["frame", "track_id"])["speed_kmh"].to_dict()
    kept = set(kin["track_id"])
    by_frame = {f: g for f, g in raw[raw["track_id"].isin(kept)].groupby("frame")}
    end = int(raw["frame"].max()) + 1 if len(raw) else 0

    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(text_scale=0.5, text_padding=3)
    trace = sv.TraceAnnotator(trace_length=60, thickness=2, position=sv.Position.BOTTOM_CENTER)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = str(Path(tmp) / "raw.mp4")
        with sv.VideoSink(tmp_path, out_info) as sink:
            for i, frame in enumerate(sv.get_video_frames_generator(video_path, stride=stride, end=end)):
                g = by_frame.get(i * stride)
                if g is None:
                    sink.write_frame(frame)
                    continue
                dets = sv.Detections(xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                                     tracker_id=g["track_id"].to_numpy(),
                                     class_id=g["track_id"].to_numpy() % 20)
                labels = [f"#{t} {c} {speeds.get((i * stride, t), 0):.0f} km/h"
                          for t, c in zip(g["track_id"], g["cls"], strict=True)]
                frame = trace.annotate(frame, dets)
                frame = box.annotate(frame, dets)
                frame = label.annotate(frame, dets, labels=labels)
                sink.write_frame(frame)
        _to_h264(tmp_path, out_path)
    return out_path


def _to_h264(src, dst):
    """OpenCV writes mp4v, which browsers can't play; re-encode to H.264 if ffmpeg exists."""
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast", str(dst)], check=True)
    else:
        shutil.copy(src, dst)


def first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read {video_path}")
    return frame
