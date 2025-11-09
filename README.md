Perception Model (Detection + Tracking)
======================================

This project provides a modular perception pipeline that detects and tracks multiple objects frame-to-frame and visualizes their trajectories. It demonstrates core skills for ADAS/robotics perception stacks: object detection, multi-object tracking, motion reasoning, and real-time visualization.

Features
--------
- Ultralytics YOLOv8 detector wrapper with configurable weights, thresholds, and device.
- Interchangeable multi-object trackers: ByteTrack (default) or DeepSORT.
- Optional dense optical flow (Farneback by default, RAFT plug-in) to estimate motion vectors that can feed AEB or collision prediction logic.
- Trajectory visualization with per-track color coding, motion arrows, and path history.
- CSV export of track states for downstream analysis or mapping.

Project Layout
--------------

```
src/perception/
  datatypes.py      # Model-agnostic detection/track structures
  detectors.py      # Detector interface and YOLOv8 implementation
  trackers.py       # DeepSORT and ByteTrack wrappers
  optical_flow.py   # Flow computation helpers (Farneback/RAFT)
  visualization.py  # Drawing utilities for tracked objects
  pipeline.py       # Orchestrates detection → tracking → visualization
scripts/
  run_pipeline.py   # CLI entry point
tests/
  test_tracks.py    # Sanity checks for track state logic
requirements.txt    # Python dependencies
```

Getting Started
---------------
1. Create and activate a Python 3.10+ environment.
2. Install dependencies (PyTorch + CUDA wheel selection may vary by platform):

   ```
   pip install -r requirements.txt
   ```

Optional: install CUDA-enabled PyTorch/Torchvision (requires an NVIDIA GPU with matching drivers):

```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

3. Download YOLOv8 weights (the Ultralytics package auto-downloads them on first run with network access). You can also provide a local `.pt` file via `--weights`.

Running the Pipeline
--------------------

Bash / WSL / Git Bash (multi-line):

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --source 'BIBBIDIBA - Hoshimachi Suisei.mp4' \
  --weights yolov8n.pt \
  --tracker bytetrack \
  --device cpu \
  --output-video outputs/annotated.mp4 \
  --output-tracks outputs/tracks.csv
```

Bash / WSL / Git Bash (single-line):

```bash
PYTHONPATH=src python scripts/run_pipeline.py --source 'BIBBIDIBA - Hoshimachi Suisei.mp4' --weights yolov8n.pt --tracker bytetrack --device cpu --output-video outputs/annotated.mp4 --output-tracks outputs/tracks.csv
```

Bash / WSL / Git Bash (CUDA single-line example):

```bash
PYTHONPATH=src python scripts/run_pipeline.py --source 'BIBBIDIBA - Hoshimachi Suisei.mp4' --weights yolov8n.pt --tracker bytetrack --device cuda --output-video outputs/annotated.mp4 --output-tracks outputs/tracks.csv
```

PowerShell (copy/paste friendly single line):

```powershell
$env:PYTHONPATH="src"; python scripts/run_pipeline.py --source "BIBBIDIBA - Hoshimachi Suisei.mp4" --weights yolov8n.pt --tracker bytetrack --device cpu --output-video outputs/annotated.mp4 --output-tracks outputs/tracks.csv
```

GPU variant (after installing the CUDA wheel above):

```powershell
$env:PYTHONPATH="src"; python scripts/run_pipeline.py --source "BIBBIDIBA - Hoshimachi Suisei.mp4" --weights yolov8n.pt --tracker bytetrack --device cuda --output-video outputs/annotated.mp4 --output-tracks outputs/tracks.csv
```

Key options:
- `--tracker`: choose `bytetrack` (default) or `deepsort` if you have `deep-sort-realtime` installed.
- `--flow-method`: `farneback`, `raft`, or `none`. Motion arrows are rendered only when flow is computed.
- `--skip-frames`: process every (N+1)th frame to meet real-time constraints.
- `--no-display` / `--no-visualization`: disable OpenCV preview or drawing when running headless.

Extending Toward AEB/Robotics
----------------------------
- Integrate depth sensing or LiDAR by fusing depth with the `TrackState.center` coordinates and velocity estimates.
- Replace `compute_dense_optical_flow` with a task-specific network such as RAFT for higher fidelity motion cues.
- Use the CSV log to build top-down occupancy maps or vehicle trajectory priors for planning modules.

Testing
-------
Run the lightweight unit tests (e.g. after editing utility functions):

```
PYTHONPATH=src pytest
```

Next Steps
----------
- Add collision prediction using track velocities and scene geometry.
- Fuse GPS/IMU odometry to stabilize trajectories in world coordinates.
- Profile end-to-end latency to validate real-time constraints on target hardware.

Performance Notes (Why This Demo Is Slower Than Production AD Stacks)
--------------------------------------------------------------------

- This repository is a Python reference implementation meant for clarity and experimentation. Real ADAS/robotics perception stacks run highly optimized C++/CUDA code on embedded GPUs/ASICs (e.g., NVIDIA Drive) with quantized models, sensor fusion, and tight scheduling across accelerators.
- The demo processes full-resolution MP4 files frame-by-frame on the host CPU/GPU, computes dense optical flow (Farneback by default), draws overlays, and re-encodes video—steps that production systems either hardware-accelerate or skip entirely.
- Vehicle platforms stream raw camera data directly into GPU memory, run INT8/FP16-optimized models, estimate motion via IMU/stereo/flow accelerators, and publish lightweight track summaries to downstream planners. Visualization is usually optional and downsampled.
- To speed up this demo: disable flow (`--flow-method none`), lower input resolution, process every Nth frame via `--skip-frames`, or use smaller YOLO weights. Expect real autonomous-driving latency only with purpose-built hardware and optimized low-level implementations.
