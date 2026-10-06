---
title: Traffic Trajectory Extractor
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Video to metric road-user trajectories and speeds
---

# Traffic Trajectory Extractor

**Live demo:** https://traffic-trajectories.politedune-edd82718.germanywestcentral.azurecontainerapps.io/

Turns a traffic video into a **trajectory dataset**: every road user is detected, tracked over time,
projected onto the road plane in meters, and given a smoothed velocity. The output is the kind of data
used to build and replay scenarios for testing automated driving functions.

| Tracked video (ID, class, speed) | Bird's-eye-view trajectories |
|---|---|
| ![annotated frame](docs/annotated.jpg) | ![bev](docs/bev.png) |

## Pipeline

```
video ──► YOLO11n detection ──► ByteTrack MOT ──► road contact point ──► homography ──► smoothing ──► CSV / BEV / video
          (COCO road users)     (track IDs)      (bbox bottom-center)   (px → m)        Savitzky–Golay
                                                                                         v, speed, heading
```

| Stage | File | Notes |
|---|---|---|
| Detection + tracking | [traj/tracking.py](traj/tracking.py) | YOLO11n at 640 px, ByteTrack via `supervision`. Boxes touching the frame border are dropped (their bottom edge is clipped, so the contact point would be wrong). |
| Image → ground plane | [traj/geometry.py](traj/geometry.py) | Homography from ≥ 4 point pairs (lane-marking corners with known spacing). `draw_grid` overlays a 5 m grid to check the calibration visually. `max_range_m` drops far points where 1 px spans several meters. |
| Kinematics | [traj/kinematics.py](traj/kinematics.py) | Per-track Savitzky–Golay fit over a ~1 s window. Velocity comes from the filter's analytic derivative, not finite differences of noisy positions. |
| Outputs | [traj/export.py](traj/export.py) | `trajectories.csv`, BEV plot, H.264 annotated video. |
| Orchestration | [traj/pipeline.py](traj/pipeline.py) | Shared by the CLI and the Gradio app. |

## Results on the sample clip

12 s motorway clip, 1280×720 @ 25 fps. On a laptop CPU it takes **~14 s** at stride 1. The free 2-vCPU Space takes about a minute at stride 2.

| track | class | median speed |
|---|---|---|
| 3 | truck | 92 km/h |
| 1, 2, 10, 12, 16 | car | 128–141 km/h |

These are plausible for a Czech motorway (130 km/h limit, trucks governed to ~90 km/h), and the lanes come
out about 3.5 m apart in the BEV. Calibration: lane width 3.75 m and 6 m + 12 m dash spacing on the lane divider
([calib/highway.json](calib/highway.json)). That dash standard is an assumption, cross-checked against the speeds.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python pipeline.py samples/highway.mp4 --calib calib/highway.json --out out   # CLI
python app.py                                                                  # web UI on :7860
pytest && ruff check .
```

Calibration file format:

```json
{"image_points": [[u, v], ...], "world_points": [[x_m, y_m], ...], "max_range_m": 180}
```

## Limitations and next steps

- **Oblique camera ⇒ occlusion.** When a car is hidden behind the car in front, its bounding-box bottom is not its
  ground contact point. The contact point then jumps as the car emerges (visible as a speed spike on track 8). A
  top-down drone view mostly removes this problem, which is why aerial capture is the better sensor setup for trajectory datasets.
- **Planar road and 2D boxes.** The homography assumes a flat road. Next steps: estimate 3D boxes / vehicle footprints, and
  calibrate from the full camera model or SfM instead of 4 manual points.
- **COCO detector.** YOLO11n is not trained on aerial imagery. Fine-tuning on VisDrone / highD-style data would improve
  small-object recall and give rotated boxes.
- **Scenario export.** Detect maneuvers (lane changes, cut-ins, hard braking) from the trajectories and export to
  OpenSCENARIO / OpenDRIVE.

## Deployment

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `ruff` and `pytest` on every push. Pushes to
`main` that pass are then deployed to both targets:

- **Hugging Face Space** (`deploy` job): the repo is mirrored to the Space, which rebuilds the Gradio app on free CPU hardware.
- **Azure Container Apps** (`deploy-azure` job): the [Dockerfile](Dockerfile) image is built, pushed to Azure Container Registry
  and rolled out to the Container App (2 vCPU / 4 GiB, scales to zero when idle). CI signs in to Azure with GitHub OIDC,
  so no Azure secret is stored in the repo.

### Azure setup (one time)

With the az CLI logged in (`az login`) and docker running:

```bash
./deploy/azure-setup.sh          # override with RG=..., LOCATION=..., ACR=..., APP=..., REPO=owner/name
```

It creates the resource group, registry, Container Apps environment and app, plus an Entra app with a federated credential
for pushes to `main` and Contributor on the resource group. At the end it prints the app URL and six values to add as
**repository variables** (Settings → Secrets and variables → Actions → *Variables*, not Secrets):
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RG`, `AZURE_ACR`, `AZURE_APP`.
The `deploy-azure` job is skipped until they are set.

Notes:
- `LOCATION` defaults to `germanywestcentral`. Azure for Students subscriptions only allow a few regions; list them with
  `az policy assignment list --query "[].parameters.listOfAllowedLocations.value"`.
- The app pulls from the registry with its admin credentials, because express Container Apps environments don't support
  managed-identity pulls.
- GitHub's OIDC subject includes numeric IDs (`repo:owner@<id>/name@<id>:ref:refs/heads/main`); the script looks them up.
  A login error `AADSTS700213` means the federated credential's subject doesn't match.

To run the image locally: `docker build -t traj . && docker run -p 7860:7860 traj`.

## Credits

Sample video: `vehicles.mp4` from the [supervision](https://github.com/roboflow/supervision) assets (trimmed and downscaled).
Detector: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0), which is why this repo is AGPL-3.0 too.
