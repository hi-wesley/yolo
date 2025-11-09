from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Union

import cv2
import numpy as np

from .datatypes import Detection, TrackState
from .detectors import BaseDetector, YoloV8Detector
from .optical_flow import compute_dense_optical_flow, estimate_track_motion
from .trackers import BaseTracker, ByteTrackTracker, DeepSortTracker
from .visualization import draw_tracks

# Typing alias for cv2.VideoCapture compatible sources.
VideoSource = Union[int, str, Path]


class PerceptionPipeline:
    """High-level orchestrator that couples detection, tracking, and visualization."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        flow_method: Optional[str] = None,
        flow_kwargs: Optional[dict] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.flow_method = flow_method
        self.flow_kwargs = flow_kwargs or {}

    def process_video(
        self,
        source: VideoSource,
        output_video: Optional[Path] = None,
        output_tracks: Optional[Path] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
        visualization: bool = True,
        show_trails: bool = True,
        show_motion: bool = True,
        skip_frames: int = 0,
    ) -> None:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

        writer = None
        output_video_path: Optional[Path] = Path(output_video) if output_video else None
        if output_video_path is not None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        track_log: List[List[Union[int, float, str]]] = []
        prev_gray: Optional[np.ndarray] = None
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if max_frames is not None and frame_index > max_frames:
                break
            if skip_frames and frame_index % (skip_frames + 1) != 1:
                continue

            detections = self.detector.predict(frame)
            tracks = self.tracker.update(detections, frame)

            if self.flow_method:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = compute_dense_optical_flow(prev_gray, gray, self.flow_method, **self.flow_kwargs)
                    estimate_track_motion(tracks, flow)
                prev_gray = gray

            for track in tracks:
                row = [
                    frame_index,
                    track.track_id,
                    track.class_id,
                    track.class_name or "",
                    *track.bbox,
                    track.confidence,
                ]
                if track.velocity:
                    row.extend([track.velocity[0], track.velocity[1]])
                track_log.append(row)

            render_frame = frame
            if visualization:
                render_frame = draw_tracks(
                    frame,
                    tracks,
                    show_trails=show_trails,
                    show_labels=True,
                    show_motion=show_motion,
                )

            if writer is not None:
                writer.write(render_frame)

            if display:
                cv2.imshow("PerceptionPipeline", render_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        if output_tracks:
            self._export_tracks(output_tracks, track_log)

    @staticmethod
    def _export_tracks(path: Path, rows: Iterable[Iterable[Union[int, float, str]]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "vx",
            "vy",
        ]
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for row in rows:
                padded = list(row)
                while len(padded) < len(fieldnames):
                    padded.append("")
                writer.writerow(padded[: len(fieldnames)])


def build_default_pipeline(
    detector_weights: str = "yolov8n.pt",
    tracker_type: str = "bytetrack",
    device: str = "cpu",
    flow_method: Optional[str] = "farneback",
    confidence: float = 0.3,
    iou: float = 0.45,
) -> PerceptionPipeline:
    detector = YoloV8Detector(weights=detector_weights, device=device, confidence=confidence, iou=iou)
    if tracker_type.lower() == "deepsort":
        tracker: BaseTracker = DeepSortTracker()
    elif tracker_type.lower() == "bytetrack":
        tracker = ByteTrackTracker(class_names=detector.class_labels())
    else:
        raise ValueError(f"Unsupported tracker_type: {tracker_type}")

    pipeline = PerceptionPipeline(
        detector=detector,
        tracker=tracker,
        flow_method=flow_method,
        flow_kwargs={},
    )
    return pipeline
