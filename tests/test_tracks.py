from perception.datatypes import TrackState
from perception.optical_flow import estimate_track_motion

import numpy as np


def test_track_history_and_velocity_update():
    track = TrackState(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, class_id=0)
    track.register_observation((0.0, 0.0, 10.0, 10.0))
    assert len(track.history) == 1
    assert track.velocity is None

    track.register_observation((2.0, 2.0, 12.0, 12.0))
    assert len(track.history) == 2
    assert track.velocity == (2.0, 2.0)


def test_estimate_track_motion_overrides_velocity():
    track = TrackState(track_id=5, bbox=(5.0, 5.0, 15.0, 15.0), confidence=0.8, class_id=1)
    track.register_observation(track.bbox)
    flow = np.zeros((30, 30, 2), dtype=np.float32)
    flow[5:15, 5:15, 0] = 1.5
    flow[5:15, 5:15, 1] = -0.5

    result = estimate_track_motion([track], flow)
    assert np.isclose(result[track.track_id][0], 1.5)
    assert np.isclose(result[track.track_id][1], -0.5)
    assert track.velocity == result[track.track_id]

