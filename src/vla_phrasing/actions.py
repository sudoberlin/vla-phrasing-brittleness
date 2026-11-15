"""Conversions between a policy's raw action and what the simulator expects."""
from __future__ import annotations

import numpy as np

N_BINS = 256  # OpenVLA discretises each action dimension into 256 bins


def euler_xyz_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll, pitch, yaw (rotations about fixed x, y, z, applied in that order)
    to an axis-angle vector whose length is the angle in radians."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    v = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
    if w < 0:
        w, v = -w, -v
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros(3)
    return v / norm * (2 * np.arctan2(norm, w))


def widowx_env_action(raw) -> np.ndarray:
    """Raw 7-D policy action (dx, dy, dz, droll, dpitch, dyaw, gripper in [0, 1])
    to the WidowX action used by SimplerEnv (delta position, axis-angle
    rotation, gripper in {-1, +1} where +1 is open)."""
    raw = np.asarray(raw, dtype=float)
    rot = euler_xyz_to_rotvec(raw[3], raw[4], raw[5])
    grip = 1.0 if raw[6] > 0.5 else -1.0
    return np.concatenate([raw[:3], rot, [grip]])


def to_bins(values, low=-1.0, high=1.0, n_bins: int = N_BINS) -> np.ndarray:
    """Index of the nearest bin centre for values on [low, high]."""
    v = np.clip(np.asarray(values, dtype=float), low, high)
    idx = np.floor((v - low) / (high - low) * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)
