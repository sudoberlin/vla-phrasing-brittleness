"""Environments. Each one can be reset to an exact, repeatable scene.

`SimplerEnvTask` wraps a SimplerEnv task. `DummyEnv` is a toy with no physics,
used for tests and dry runs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DUMMY_INSTRUCTIONS = {
    "widowx_spoon_on_towel": "put the spoon on the towel",
}


class DummyEnv:
    """A point that moves by the first three action dimensions. The goal
    depends only on the episode id, so a reset is exactly repeatable."""

    def __init__(self, task: str = "widowx_spoon_on_towel"):
        self.task = task
        self.pos = np.zeros(3)
        self.goal = np.zeros(3)

    def reset(self, episode_id: int) -> np.ndarray:
        rng = np.random.default_rng(episode_id)
        self.pos = np.zeros(3)
        self.goal = rng.uniform(-0.5, 0.5, size=3)
        return self._image()

    def instruction(self) -> str:
        return DUMMY_INSTRUCTIONS.get(self.task, "put the spoon on the towel")

    def step(self, env_action) -> Tuple[np.ndarray, bool]:
        self.pos = self.pos + np.asarray(env_action, dtype=float)[:3]
        return self._image(), bool(np.linalg.norm(self.pos - self.goal) < 0.05)

    def ee_position(self) -> Optional[np.ndarray]:
        return self.pos.copy()

    def _image(self) -> np.ndarray:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        delta = np.clip(self.goal - self.pos, -1, 1)
        img[:, :] = np.round((delta + 1.0) * 127.5).astype(np.uint8)
        return img


class SimplerEnvTask:
    """One SimplerEnv task. `episode_id` picks one of the task's predefined
    object layouts, so the same id gives the same scene on every reset."""

    def __init__(self, task: str):
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        self.task = task
        self.env = simpler_env.make(task)
        self._get_image = get_image_from_maniskill2_obs_dict
        self._obs = None

    def reset(self, episode_id: int) -> np.ndarray:
        self._obs, _ = self.env.reset(options={"obj_init_options": {"episode_id": int(episode_id)}})
        return self._image()

    def instruction(self) -> str:
        env = self.env if hasattr(self.env, "get_language_instruction") else self.env.unwrapped
        return env.get_language_instruction()

    def step(self, env_action) -> Tuple[np.ndarray, bool]:
        self._obs, _reward, done, _truncated, info = self.env.step(np.asarray(env_action, dtype=np.float32))
        return self._image(), bool(info.get("success", done))

    def ee_position(self) -> Optional[np.ndarray]:
        try:
            return np.asarray(self._obs["extra"]["tcp_pose"][:3], dtype=float)
        except (KeyError, TypeError):
            return None

    def _image(self) -> np.ndarray:
        return np.asarray(self._get_image(self.env, self._obs))


def make_env(name: str, task: str):
    if name == "dummy":
        return DummyEnv(task)
    if name == "simpler":
        return SimplerEnvTask(task)
    raise ValueError(f"unknown env {name!r}")
