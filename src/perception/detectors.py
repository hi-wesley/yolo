from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np

from .datatypes import Detection, detections_from_yolo_output


class BaseDetector(ABC):
    """Interface for frame-level object detectors."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Return detections for a single BGR frame."""

    def class_labels(self) -> Optional[Iterable[str]]:
        return None


class YoloV8Detector(BaseDetector):
    """Thin wrapper around Ultralytics YOLOv8 models."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        classes: Optional[Iterable[int]] = None,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO  # lazy import to avoid hard dependency at module import time

        self._model = YOLO(weights)
        self._conf = confidence
        self._iou = iou
        self._device = device
        self._classes = list(classes) if classes is not None else None
        self._half = half

        # Model device and precision configuration.
        try:
            self._model.to(device)
        except AttributeError:
            # Older ultralytics versions expose model.model
            getattr(self._model, "model", self._model).to(device)
        if half and device != "cpu":
            getattr(self._model, "model", self._model).half()

    def predict(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy()
        class_names = getattr(self._model, "names", None)
        return detections_from_yolo_output(boxes, scores, class_ids, class_names)

    def class_labels(self) -> Optional[Iterable[str]]:
        return getattr(self._model, "names", None)

