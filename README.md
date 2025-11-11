Perception Model (Detection + Tracking)
======================================

[Link to Demo](https://www.youtube.com/watch?v=nIukFJ31e8A)

![Screenshot](https://i.imgur.com/apwDCl3.png)

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
---------------------------------------------------------------------

### Technical detail

- The pipeline is written in Python + OpenCV for readability. Every frame crosses the Python↔NumPy boundary multiple times (decode → detector → tracker → optical flow → drawing), which injects tens of milliseconds of overhead that would disappear in fused C++/CUDA kernels.
- Detection relies on a general-purpose Ultralytics YOLOv8 model running in eager PyTorch mode. Production stacks export detectors into TensorRT or custom accelerators, quantize to FP16/INT8, batch operations, and pin execution to GPU streams so inference finishes in a few milliseconds.
- Tracking/optical-flow steps borrow reference implementations (ByteTrack, DeepSORT, Farneback). In ADAS hardware, multi-object tracking typically runs inside the same GPU/ASIC memory space as the detector, or uses dedicated motion-estimation IP blocks (IMU fusion, stereo disparity, sparse optical flow) to avoid dense pixel-level computations every frame.
- Video I/O is another bottleneck: OpenCV decodes compressed MP4 frames on the CPU and copies them into system RAM before shipping them to the GPU. Automotive cameras stream raw frames (often via GMSL) straight into GPU-accessible buffers with deterministic timestamps.
- Rendering annotations and transcoding the annotated MP4 happen synchronously in the same loop. Real vehicles either skip visualization entirely or downsample/async-render on a secondary processor so perception deadlines are never blocked by drawing.
- The control loop here is single-threaded and unbounded—no watchdogs, no rate governors, and no preallocation of GPU memory. Production software runs under real-time operating constraints, preallocates memory pools, and uses message-passing middleware (ROS2, DDS, custom RTOS) to keep latency predictable.

### Plain-language explanation

Think of this repository as a teaching demo. It watches a video file, pauses on every frame to look for objects, draws colorful boxes, calculates how things move, and then saves a new video. Those pauses are fine when you are experimenting on a laptop, but a self-driving car cannot afford them—it has to sense and react dozens of times every second.

Real vehicles tackle the problem with faster parts and tighter software. Cameras stream directly into special chips designed for math-heavy tasks, so the computer never wastes time copying images around. The detection brains (neural networks) are shrunk and tuned so they run in milliseconds, and motion calculations are either simplified or handled by extra sensors such as radar and IMUs. Rendering pretty overlays is optional; the car only needs the numbers (what object, where, how fast) to stay safe.

### What would need to change for real-time behavior?

1. **Low-level rewrite:** Move the detector, tracker, optical-flow, and drawing code into C++/CUDA/TensorRT so GPU kernels process frames without Python overhead, and fuse steps to minimize memory copies.
2. **Model optimization:** Export the YOLO weights to an inference engine (TensorRT, ONNX Runtime, OpenVINO), quantize to FP16/INT8, and prune layers so a single forward pass meets the target budget (often <10 ms per camera).
3. **Sensor ingest:** Replace MP4 decoding with live camera pipelines (GStreamer/V4L2/GMSL) that DMA frames straight into GPU memory, synchronized with IMU/RADAR data.
4. **Asynchronous pipeline:** Break the monolithic loop into parallel stages (decode, infer, track, publish) connected via lock-free queues so work overlaps and deadlines are deterministic.
5. **Hardware alignment:** Run on embedded automotive GPUs/ASICs with real-time schedulers, fixed memory pools, and watchdogs so perception keeps up even under thermal throttling or transient load.

Even if all of the above were implemented, the gap to production-grade autonomy would remain because OEM stacks add rigorous safety validation, redundancy, sensor fusion across multiple modalities, and integration with planning/control modules. This repository is intentionally lightweight: it demonstrates the concepts of detection, tracking, motion cues, and visualization, but it does not attempt to meet automotive reliability, latency, or safety requirements.

### If the code were optimized but the hardware stayed the same

**Technical view:**

- Software optimizations (C++/CUDA rewrite, kernel fusion, quantized models) can trim per-frame latency, but the ceiling is set by your GPU/CPU’s FLOPs, memory bandwidth, and thermals. Commodity hardware rarely sustains the <10 ms per camera budget that automotive SoCs guarantee with dedicated accelerators.
- USB/PCIe capture paths still DMA frames through system RAM with non-deterministic latency. Without vehicle-grade camera links (GMSL/FlexRay) and deterministic DMA engines, I/O jitter becomes the next bottleneck even if compute speeds up.
- Desktop operating systems cannot provide hard real-time guarantees: the scheduler may preempt perception threads, there are no watchdogs, and clock drift/interrupt handling remain uncontrolled. Optimized code executes faster on average, but worst-case latency is still unbounded.
- Sustained full utilization increases power draw and temperature; laptops/desktops throttle unpredictably, undoing part of the software gain.

**Plain-language view:**

Tuning the code makes the demo feel snappier—maybe it jumps from 5 frames per second to 25—but the same laptop/desktop hardware still can’t match the reliability or lightning-fast reaction times inside a self-driving car’s computer. Frames still have to squeeze through slower cables, the operating system can pause the app at any moment, and the machine might heat up and slow down right when you need it most. Better software helps, yet without purpose-built automotive hardware the system stays a teaching tool rather than a road-ready perception stack.
