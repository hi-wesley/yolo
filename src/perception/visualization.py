from __future__ import annotations

import colorsys
from typing import Iterable, Tuple

import cv2
import numpy as np

from .datatypes import TrackState


def _id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic color mapping for a track id."""
    hue = (track_id * 0.61803398875) % 1.0  # golden ratio multiplier for separation
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(255 * b), int(255 * g), int(255 * r)  # OpenCV uses BGR order


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[TrackState],
    show_trails: bool = True,
    show_labels: bool = True,
    show_motion: bool = True,
) -> np.ndarray:
    """Overlay tracked bounding boxes, ids, and trajectories on a frame."""
    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        color = _id_to_color(track.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

        if show_labels:
            label = f"ID {track.track_id}"
            if track.class_name:
                label += f" {track.class_name}"
            if track.confidence:
                label += f" {track.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        if show_trails and track.history:
            points = list(track.history)
            for i in range(1, len(points)):
                cv2.line(
                    canvas,
                    (int(points[i - 1][0]), int(points[i - 1][1])),
                    (int(points[i][0]), int(points[i][1])),
                    color,
                    thickness=2,
                )

        if show_motion and track.velocity:
            cx, cy = track.center
            vx, vy = track.velocity
            cv2.arrowedLine(
                canvas,
                (int(cx), int(cy)),
                (int(cx + vx * 5.0), int(cy + vy * 5.0)),
                color,
                thickness=2,
                tipLength=0.3,
            )

    return canvas

