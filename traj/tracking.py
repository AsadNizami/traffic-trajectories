"""Detection (YOLO) + multi-object tracking (ByteTrack) over a video."""

import pandas as pd
import supervision as sv
from ultralytics import YOLO

# COCO class ids we treat as road users.
ROAD_USERS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_models = {}


def load_model(weights="yolo11n.pt"):
    if weights not in _models:
        _models[weights] = YOLO(weights)
    return _models[weights]


def track_video(video_path, max_seconds=15.0, stride=1, weights="yolo11n.pt",
                imgsz=640, conf=0.3, progress=None):
    """Run detection + tracking and return one row per (frame, track).

    Columns: frame, t, track_id, cls, conf, x1, y1, x2, y2, u, v where (u, v) is
    the bottom-center of the box, i.e. the approximate road contact point.
    Boxes touching the image border are dropped: their bottom edge is clipped,
    so the contact point would be wrong.
    """
    model = load_model(weights)
    info = sv.VideoInfo.from_video_path(video_path)
    end = min(info.total_frames or 10**9, int(max_seconds * info.fps))
    mx, my = 0.01 * info.width, 0.01 * info.height  # border margin
    tracker = sv.ByteTrack(frame_rate=max(1, round(info.fps / stride)))
    rows = []
    n_steps = max(1, end // stride)
    frames = sv.get_video_frames_generator(video_path, stride=stride, end=end)
    for i, frame in enumerate(frames):
        frame_idx = i * stride
        result = model(frame, imgsz=imgsz, conf=conf, classes=list(ROAD_USERS), verbose=False)[0]
        dets = tracker.update_with_detections(sv.Detections.from_ultralytics(result))
        x1s, y1s, x2s, y2s = dets.xyxy.T
        dets = dets[(x1s > mx) & (y1s > my) & (x2s < info.width - mx) & (y2s < info.height - my)]
        cols_iter = zip(dets.xyxy, dets.tracker_id, dets.class_id, dets.confidence, strict=True)
        for (x1, y1, x2, y2), tid, cid, c in cols_iter:
            rows.append((frame_idx, frame_idx / info.fps, int(tid), ROAD_USERS[int(cid)], float(c),
                         x1, y1, x2, y2, (x1 + x2) / 2, y2))
        if progress:
            progress((i + 1) / n_steps)
    cols = ["frame", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "u", "v"]
    return pd.DataFrame(rows, columns=cols), info
