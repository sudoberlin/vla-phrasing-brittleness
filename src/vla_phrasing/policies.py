"""Policies. A policy maps (image, instruction) to one action.

`OpenVLAPolicy` is the model under study. `DummyPolicy` is a stand-in with no
learning in it. It exists so the whole pipeline can be run and tested on a
laptop, and its numbers mean nothing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .actions import N_BINS, to_bins, widowx_env_action


@dataclass
class PolicyOutput:
    raw: np.ndarray                # action as the model outputs it, shape (7,)
    env_action: np.ndarray         # action in the simulator's convention, shape (7,)
    tokens: Optional[np.ndarray]   # discrete bin per dimension, shape (7,), if known


class DummyPolicy:
    """Reads a goal offset that DummyEnv paints into the image and steps toward
    it, plus a small fixed offset derived from a hash of the instruction text.
    Deterministic. Used for tests and dry runs only."""

    name = "dummy"

    def __init__(self, sensitivity: float = 0.004):
        self.sensitivity = sensitivity

    def act(self, image: np.ndarray, instruction: str) -> PolicyOutput:
        delta = image[0, 0, :3].astype(float) / 127.5 - 1.0
        digest = hashlib.sha256(instruction.encode("utf-8")).digest()
        noise = (np.frombuffer(digest[:3], dtype=np.uint8).astype(float) / 127.5 - 1.0)
        raw = np.zeros(7)
        raw[:3] = 0.2 * delta + self.sensitivity * noise
        raw[6] = 1.0
        return PolicyOutput(raw=raw, env_action=raw.copy(), tokens=to_bins(raw))


class OpenVLAPolicy:
    """OpenVLA through Hugging Face transformers, greedy decoding.

    Needs a CUDA GPU. `unnorm_key` selects the action statistics used to
    un-normalise the output, "bridge_orig" for the WidowX (Bridge) tasks.
    """

    name = "openvla"

    def __init__(self, model_id: str = "openvla/openvla-7b", unnorm_key: str = "bridge_orig",
                 device: str = "cuda:0", load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.unnorm_key = unnorm_key
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.vla = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
        else:
            self.vla = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs).to(device)

    @staticmethod
    def prompt(instruction: str) -> str:
        # The instruction is passed through unchanged. Its wording is the variable.
        return f"In: What action should the robot take to {instruction}?\nOut:"

    def act(self, image: np.ndarray, instruction: str) -> PolicyOutput:
        from PIL import Image

        pil = Image.fromarray(np.asarray(image)[:, :, :3].astype(np.uint8))
        inputs = self.processor(self.prompt(instruction), pil).to(self.device, dtype=self.torch.bfloat16)
        raw = np.asarray(
            self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False), dtype=float)
        return PolicyOutput(raw=raw, env_action=widowx_env_action(raw), tokens=self._tokens(raw))

    def _tokens(self, raw: np.ndarray) -> Optional[np.ndarray]:
        """Recover the bin the model chose for each dimension by inverting the
        un-normalisation. Returns None if the model does not expose its
        statistics, in which case only continuous metrics are available."""
        try:
            stats = self.vla.get_action_stats(self.unnorm_key)
            low, high = np.asarray(stats["q01"], dtype=float), np.asarray(stats["q99"], dtype=float)
            mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
            span = np.where(high - low == 0, 1.0, high - low)
            normalised = np.where(mask, 2 * (raw - low) / span - 1, raw)
            return to_bins(normalised, n_bins=N_BINS)
        except Exception:
            return None


def make_policy(name: str, **kwargs):
    if name == "dummy":
        return DummyPolicy()
    if name == "openvla":
        return OpenVLAPolicy(**kwargs)
    raise ValueError(f"unknown policy {name!r}")
