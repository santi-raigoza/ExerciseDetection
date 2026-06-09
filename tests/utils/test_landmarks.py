# tests/utils/test_landmarks.py
import numpy as np
import pytest
from utils.landmarks import normalize, build_window


class MockLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def make_33(overrides=None):
    """33 landmarks all at origin, with optional index overrides: {idx: (x,y,z)}."""
    lms = [MockLandmark() for _ in range(33)]
    if overrides:
        for i, (x, y, z) in overrides.items():
            lms[i] = MockLandmark(x, y, z)
    return lms


def test_normalize_returns_shape_99():
    result = normalize(make_33())
    assert result.shape == (99,)


def test_normalize_centers_hip_midpoint_at_origin():
    # Left hip (23), right hip (24), shoulders (11, 12) above hips
    lms = make_33({
        23: (0.4, 0.6, 0.0),  # left hip
        24: (0.6, 0.6, 0.0),  # right hip
        11: (0.4, 0.4, 0.0),  # left shoulder
        12: (0.6, 0.4, 0.0),  # right shoulder
    })
    result = normalize(lms)
    reshaped = result.reshape(33, 3)
    hip_mid_normalized = (reshaped[23] + reshaped[24]) / 2.0
    np.testing.assert_allclose(hip_mid_normalized, [0.0, 0.0, 0.0], atol=1e-6)


def test_normalize_scales_so_shoulder_midpoint_is_at_minus_one():
    # Hips at y=0.6, shoulders at y=0.4 → torso_height = 0.2
    # After normalization, shoulder midpoint should be at y = -0.2/0.2 = -1.0
    lms = make_33({
        23: (0.5, 0.6, 0.0),
        24: (0.5, 0.6, 0.0),
        11: (0.5, 0.4, 0.0),
        12: (0.5, 0.4, 0.0),
    })
    result = normalize(lms)
    reshaped = result.reshape(33, 3)
    shoulder_mid = (reshaped[11] + reshaped[12]) / 2.0
    np.testing.assert_allclose(shoulder_mid[:2], [0.0, -1.0], atol=1e-6)


def test_normalize_no_crash_when_torso_height_is_zero():
    # All landmarks at origin — torso height is 0, should not raise
    result = normalize(make_33())
    assert result.shape == (99,)
    assert not np.any(np.isnan(result))


def test_build_window_returns_shape_2970():
    frames = [np.ones(99) for _ in range(30)]
    assert build_window(frames).shape == (2970,)


def test_build_window_preserves_frame_order():
    # Frame i is all i's — after concatenation frame 0 starts at index 0,
    # frame 1 at index 99, frame 29 at index 2871
    frames = [np.full(99, float(i)) for i in range(30)]
    result = build_window(frames)
    assert result[0] == 0.0
    assert result[99] == 1.0
    assert result[2871] == 29.0
