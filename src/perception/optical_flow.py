from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .datatypes import TrackState


def compute_dense_optical_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    method: str = "farneback",
    **kwargs,
) -> np.ndarray:
    """Compute dense optical flow between two grayscale frames."""
    if method.lower() == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=kwargs.get("pyr_scale", 0.5),
            levels=kwargs.get("levels", 3),
            winsize=kwargs.get("winsize", 15),
            iterations=kwargs.get("iterations", 3),
            poly_n=kwargs.get("poly_n", 5),
            poly_sigma=kwargs.get("poly_sigma", 1.2),
            flags=kwargs.get("flags", 0),
        )
        return flow

    if method.lower() == "raft":
        try:
            from torchvision import transforms
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "RAFT optical flow requires PyTorch and Torchvision. Install them to enable this path."
            ) from exc
        try:
            from raft import RAFT
            from utils.utils import InputPadder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "RAFT model code not found. Clone https://github.com/princeton-vl/RAFT "
                "and ensure it is on PYTHONPATH."
            ) from exc

        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model_weights = kwargs.get("weights", "raft-things.pth")
        model = RAFT(kwargs.get("small", False), kwargs.get("mixed_precision", False), kwargs.get("alternate_corr", False))
        model.load_state_dict(torch.load(model_weights, map_location=device))
        model.to(device).eval()

        to_tensor = transforms.ToTensor()
        with torch.no_grad():
            image1 = to_tensor(prev_gray).unsqueeze(0).to(device)
            image2 = to_tensor(gray).unsqueeze(0).to(device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = model(image1, image2, iters=kwargs.get("iters", 20), test_mode=True)
        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        h, w = gray.shape
        return cv2.resize(flow_np, (w, h), interpolation=cv2.INTER_LINEAR)

    raise ValueError(f"Unsupported optical flow method: {method}")


def estimate_track_motion(tracks: Iterable[TrackState], flow: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Estimate per-track motion vectors given dense optical flow."""
    motion: Dict[int, Tuple[float, float]] = {}
    h, w = flow.shape[:2]
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        region = flow[y1:y2, x1:x2]
        if region.size == 0:
            continue
        mean_flow = region.mean(axis=(0, 1))
        motion[track.track_id] = (float(mean_flow[0]), float(mean_flow[1]))
        track.velocity = motion[track.track_id]
    return motion

