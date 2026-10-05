"""CLI: python pipeline.py samples/highway.mp4 --calib calib/highway.json --out out/"""

import argparse

from traj.geometry import GroundPlane
from traj.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--calib", required=True, help="JSON with image_points and world_points")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()
    _, summary, paths = run(a.video, GroundPlane.from_json(a.calib), a.out, a.max_seconds, a.stride, a.conf)
    print(summary.to_string(index=False))
    print({k: str(v) for k, v in paths.items()})
