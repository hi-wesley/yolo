from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

import numpy as np

from .datatypes import Detection, TrackState


class BaseTracker(ABC):
    """Interface for multi-object trackers operating on per-frame detections."""

    def __init__(self) -> None:
        self._active_tracks: Dict[int, TrackState] = {}

    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackState]:
        """Update tracker state and return active tracks."""

    def reset(self) -> None:
        self._active_tracks.clear()

    def _finalize_tracks(self, updates: Iterable[TrackState]) -> List[TrackState]:
        next_state = {track.track_id: track for track in updates}
        self._active_tracks = next_state
        return list(next_state.values())


class DeepSortTracker(BaseTracker):
    """DeepSORT tracker wrapper using the deep-sort-realtime implementation."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
        nn_budget: Optional[int] = None,
        embedder: str = "mobilenet",
    ) -> None:
        super().__init__()
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError(
                "DeepSortTracker requires the deep-sort-realtime package. "
                "Install via `pip install deep-sort-realtime`."
            ) from exc

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            embedder_gpu=False,
        )

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackState]:
        ds_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            ds_inputs.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, det.class_id))

        tracks = self._tracker.update_tracks(ds_inputs, frame=frame)

        for track_state in self._active_tracks.values():
            track_state.age += 1

        updates: List[TrackState] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            bbox = track.to_ltrb()
            state = self._active_tracks.get(track_id)
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(track.det_confidence or 0.0),
                    class_id=int(track.det_class or -1),
                )
            state.register_observation(
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            )
            state.confidence = float(track.det_confidence or state.confidence)
            state.class_id = int(track.det_class or state.class_id)
            state.age = 0
            updates.append(state)

        return self._finalize_tracks(updates)


class ByteTrackTracker(BaseTracker):
    """ByteTrack tracker wrapper relying on the Ultralytics implementation."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        try:
            from ultralytics.yolo.utils.tracker import byte_tracker
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError(
                "ByteTrackTracker requires Ultralytics YOLO tracking utilities. "
                "Install via `pip install ultralytics`."
            ) from exc

        args = byte_tracker.TrackerArgs(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
        )
        self._tracker = byte_tracker.BYTETracker(args, frame_rate=frame_rate)
        self._class_names = list(class_names) if class_names is not None else None

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackState]:
        try:
            from ultralytics.yolo.utils.tracker.byte_tracker import STrack
        except ImportError:  # pragma: no cover - runtime guard
            STrack = None  # type: ignore[assignment]

        for track_state in self._active_tracks.values():
            track_state.age += 1

        det_array = np.empty((0, 6), dtype=np.float32)
        if detections:
            rows = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                rows.append([x1, y1, x2, y2, det.confidence, det.class_id])
            det_array = np.asarray(rows, dtype=np.float32)

        img_h, img_w = frame.shape[:2]
        online_targets = self._tracker.update(det_array, (img_h, img_w), (img_h, img_w))

        updates: List[TrackState] = []
        for track in online_targets:
            if hasattr(track, "tlbr"):
                x1, y1, x2, y2 = track.tlbr
            else:
                tlwh = track.tlwh
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
            track_id = int(track.track_id)
            score = float(getattr(track, "score", 0.0))
            class_id = int(getattr(track, "cls", -1))
            name = None
            if self._class_names is not None and 0 <= class_id < len(self._class_names):
                name = self._class_names[class_id]

            state = self._active_tracks.get(track_id)
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=score,
                    class_id=class_id,
                    class_name=name,
                )
            state.class_id = class_id
            state.class_name = name
            state.confidence = score if score > 0 else state.confidence
            state.register_observation((float(x1), float(y1), float(x2), float(y2)))
            state.age = 0
            updates.append(state)

        return self._finalize_tracks(updates)
