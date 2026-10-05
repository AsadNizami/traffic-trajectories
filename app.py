"""Gradio demo: traffic video -> tracked road users -> metric bird's-eye-view trajectories."""

import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from traj.export import bev_figure, first_frame
from traj.geometry import GroundPlane
from traj.pipeline import run

ROOT = Path(__file__).parent
SAMPLE_VIDEO = str(ROOT / "samples" / "highway.mp4")
SAMPLE_CALIB = (ROOT / "calib" / "highway.json").read_text()


def _plane(calib_json):
    try:
        return GroundPlane.from_dict(json.loads(calib_json))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        raise gr.Error(f"Invalid calibration: {e}") from e


def preview(video, calib_json):
    if not video:
        raise gr.Error("Upload a video first.")
    grid = _plane(calib_json).draw_grid(first_frame(video))
    return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)


def process(video, calib_json, max_seconds, stride, conf, progress=gr.Progress()):  # noqa: B008 (Gradio idiom)
    if not video:
        raise gr.Error("Upload a video first.")
    plane = _plane(calib_json)
    kin, summary, paths = run(video, plane, tempfile.mkdtemp(), max_seconds=max_seconds,
                              stride=int(stride), conf=conf, progress=progress)
    if kin.empty:
        gr.Warning("No road users tracked long enough. Try a lower confidence or a longer clip.")
    return str(paths["video"]), bev_figure(kin, plane), summary, str(paths["csv"])


with gr.Blocks(title="Traffic Trajectory Extractor") as demo:
    gr.Markdown(
        "# Traffic Trajectory Extractor\n"
        "Detect and track road users (YOLO11 + ByteTrack), map them onto the road plane with a "
        "homography, and estimate trajectories and speeds in **meters** and **km/h**. "
        "The sample clip and calibration are preloaded, so just press **Extract trajectories**. "
        "[Source code](https://github.com/AsadNizami/traffic-trajectories) · "
        "runs on a free CPU, so expect about a minute."
    )
    with gr.Row():
        with gr.Column():
            video = gr.Video(value=SAMPLE_VIDEO, label="Traffic video", sources=["upload"])
            with gr.Accordion("Ground-plane calibration", open=False):
                gr.Markdown(
                    "Four or more image points (pixels, x right / y down) and their road-plane "
                    "coordinates in meters, e.g. lane-marking corners with known lane width and dash "
                    "spacing. Optional `max_range_m` drops far points where 1 px spans meters. "
                    "Use **Preview calibration** to overlay a 5 m grid and check the fit."
                )
                calib = gr.Code(value=SAMPLE_CALIB, language="json", label="Calibration JSON")
                preview_btn = gr.Button("Preview calibration")
                grid_img = gr.Image(label="Calibration grid (5 m)", interactive=False)
            with gr.Row():
                max_seconds = gr.Slider(2, 20, value=12, step=1, label="Max seconds to process")
                stride = gr.Slider(1, 4, value=2, step=1, label="Frame stride (higher = faster)")
                conf = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Detection confidence")
            run_btn = gr.Button("Extract trajectories", variant="primary")
        with gr.Column():
            out_video = gr.Video(label="Tracked video")
            bev = gr.Plot(label="Bird's-eye view")
    with gr.Row():
        table = gr.Dataframe(label="Per-track summary")
        csv = gr.File(label="Trajectories CSV (frame, t, id, class, x_m, y_m, vx, vy, speed_kmh, heading_deg)")

    preview_btn.click(preview, [video, calib], grid_img)
    run_btn.click(process, [video, calib, max_seconds, stride, conf], [out_video, bev, table, csv])

if __name__ == "__main__":
    demo.launch()
