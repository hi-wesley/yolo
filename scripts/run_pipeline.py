#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from perception.pipeline import build_default_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-object detection and tracking.")
    parser.add_argument("--source", type=str, required=True, help="Video source (path or camera index).")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLOv8 weights.")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=("bytetrack", "deepsort"),
        help="Tracker backend to use.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use for inference.")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frame count.")
    parser.add_argument("--output-video", type=str, default=None, help="Optional path to export annotated video.")
    parser.add_argument("--output-tracks", type=str, default=None, help="Optional path to export CSV track log.")
    parser.add_argument(
        "--flow-method",
        type=str,
        default="farneback",
        choices=("farneback", "raft", "none"),
        help="Optical flow method to estimate motion vectors.",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window display.")
    parser.add_argument("--no-visualization", action="store_true", help="Do not render annotations on frames.")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every Nth frame (0 = process all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow_method: Optional[str] = None if args.flow_method == "none" else args.flow_method

    pipeline = build_default_pipeline(
        detector_weights=args.weights,
        tracker_type=args.tracker,
        device=args.device,
        flow_method=flow_method,
        confidence=args.confidence,
        iou=args.iou,
    )

    source: str | int
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    pipeline.process_video(
        source=source,
        output_video=Path(args.output_video) if args.output_video else None,
        output_tracks=Path(args.output_tracks) if args.output_tracks else None,
        display=not args.no_display,
        max_frames=args.max_frames,
        visualization=not args.no_visualization,
        show_trails=True,
        show_motion=flow_method is not None,
        skip_frames=args.skip_frames,
    )


if __name__ == "__main__":
    main()

