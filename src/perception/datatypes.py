from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Tuple

import numpy as np
from collections import deque


ArrayXYXY = Tuple[float, float, float, float]


@dataclass
class Detection:
    """Model-agnostic representation of a single object detection."""

    bbox: ArrayXYXY
    confidence: float
    class_id: int
    class_name: Optional[str] = None

    def as_xyxy(self) -> ArrayXYXY:
        return self.bbox

    def as_tlwh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1


@dataclass
class TrackState:
    """Current state for a tracked object."""

    track_id: int
    bbox: ArrayXYXY
    confidence: float
    class_id: int
    class_name: Optional[str] = None
    age: int = 0
    hits: int = 0
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=32))
    velocity: Optional[Tuple[float, float]] = None

    def register_observation(self, bbox: ArrayXYXY) -> None:
        self.bbox = bbox
        self.hits += 1
        center = self.center
        if self.history and len(self.history) > 0:
            prev_x, prev_y = self.history[-1]
            self.velocity = (center[0] - prev_x, center[1] - prev_y)
        self.history.append(center)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def as_xyxy(self) -> ArrayXYXY:
        return self.bbox


def detections_from_yolo_output(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: Optional[Iterable[str]] = None,
) -> List[Detection]:
    """Build Detection objects from YOLO tensors."""
    detections: List[Detection] = []
    names = list(class_names) if class_names is not None else None
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.tolist()
        name = names[int(cls)] if names is not None else None
        detections.append(
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(score),
                class_id=int(cls),
                class_name=name,
            )
        )
    return detections

