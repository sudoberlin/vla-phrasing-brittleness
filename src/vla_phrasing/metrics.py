"""Metrics. All of them compare a paraphrase against the canonical instruction
under an identical scene, so any difference is caused by wording."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a success rate. Sensible at small n."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def token_disagreement(tokens_a, tokens_b) -> float:
    """Fraction of action dimensions whose discrete bin differs."""
    a, b = np.asarray(tokens_a), np.asarray(tokens_b)
    return float(np.mean(a != b))


def bin_distance(tokens_a, tokens_b) -> float:
    """Mean absolute difference in bin index. Separates near misses from jumps."""
    a, b = np.asarray(tokens_a, dtype=float), np.asarray(tokens_b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def action_token_variance(tokens) -> float:
    """Variance of the bin index across a set of instructions for one
    observation, averaged over action dimensions. Input shape (P, D)."""
    t = np.asarray(tokens, dtype=float)
    if t.ndim != 2:
        raise ValueError("expected shape (num_instructions, action_dim)")
    return float(np.mean(np.var(t, axis=0)))


def first_divergence_step(tokens_a, tokens_b) -> Optional[int]:
    """First timestep at which two rollouts choose a different action.
    None if they agree for as long as both run."""
    a, b = np.asarray(tokens_a), np.asarray(tokens_b)
    n = min(len(a), len(b))
    for t in range(n):
        if np.any(a[t] != b[t]):
            return t
    return None


def trajectory_divergence(path_a, path_b) -> Dict[str, float]:
    """Distance between two end-effector paths of shape (T, 3), in the units of
    the paths. The shorter path is padded by holding its last position, which
    is what a robot that has stopped does."""
    a, b = np.asarray(path_a, dtype=float), np.asarray(path_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"mean": float("nan"), "max": float("nan"), "final": float("nan")}
    n = max(len(a), len(b))
    a = np.vstack([a, np.repeat(a[-1:], n - len(a), axis=0)])
    b = np.vstack([b, np.repeat(b[-1:], n - len(b), axis=0)])
    d = np.linalg.norm(a - b, axis=1)
    return {"mean": float(d.mean()), "max": float(d.max()), "final": float(d[-1])}
