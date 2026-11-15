import json
from pathlib import Path

import numpy as np
import pytest

from vla_phrasing import analyze, run
from vla_phrasing.actions import euler_xyz_to_rotvec, to_bins, widowx_env_action
from vla_phrasing.envs import DummyEnv
from vla_phrasing.metrics import (action_token_variance, bin_distance, first_divergence_step,
                                  token_disagreement, trajectory_divergence, wilson_interval)
from vla_phrasing.paraphrases import CATEGORIES, ParaphraseSet, Paraphrase, load_set, validate
from vla_phrasing.policies import DummyPolicy

ROOT = Path(__file__).resolve().parents[1]


# ---- paraphrase files ----

@pytest.mark.parametrize("path", sorted((ROOT / "paraphrases").glob("*.yaml")), ids=lambda p: p.stem)
def test_paraphrase_files_are_valid(path):
    pset = load_set(path)
    assert pset.task == path.stem
    assert {p.category for p in pset.paraphrases} == set(CATEGORIES)


def test_validate_rejects_canonical_duplicate():
    with pytest.raises(ValueError):
        validate(ParaphraseSet("t", "put a on b", [Paraphrase("x", "verb", "Put A on B")]))


# ---- metrics ----

def test_wilson_interval():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and 0.6 < lo < 0.8


def test_token_metrics():
    a, b = [1, 2, 3, 4], [1, 2, 5, 0]
    assert token_disagreement(a, a) == 0
    assert token_disagreement(a, b) == 0.5
    assert bin_distance(a, b) == pytest.approx(1.5)
    assert action_token_variance([[0, 10], [2, 10]]) == pytest.approx(0.5)


def test_first_divergence_step():
    a = [[1, 1], [2, 2], [3, 3]]
    assert first_divergence_step(a, a) is None
    assert first_divergence_step(a, [[1, 1], [2, 9], [3, 3]]) == 1
    assert first_divergence_step(a, a[:2]) is None


def test_trajectory_divergence_pads_shorter_path():
    a = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    d = trajectory_divergence(a, a[:2])
    assert d["final"] == pytest.approx(1.0) and d["max"] == pytest.approx(1.0)
    assert d["mean"] == pytest.approx(1 / 3)
    assert trajectory_divergence(a, a)["max"] == 0


# ---- action conversion ----

def test_rotvec_single_axis():
    assert np.allclose(euler_xyz_to_rotvec(0.3, 0, 0), [0.3, 0, 0])
    assert np.allclose(euler_xyz_to_rotvec(0, -0.2, 0), [0, -0.2, 0])
    assert np.allclose(euler_xyz_to_rotvec(0, 0, 0.1), [0, 0, 0.1])
    assert np.allclose(euler_xyz_to_rotvec(0, 0, 0), 0)
    assert np.allclose(euler_xyz_to_rotvec(np.pi, 0, 0), [np.pi, 0, 0])


def test_rotvec_matches_rotation_matrix():
    def matrix(r, p, y):
        rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return rz @ ry @ rx

    def rodrigues(v):
        angle = np.linalg.norm(v)
        k = v / angle
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    rng = np.random.default_rng(0)
    for r, p, y in rng.uniform(-1.5, 1.5, size=(50, 3)):
        assert np.allclose(rodrigues(euler_xyz_to_rotvec(r, p, y)), matrix(r, p, y), atol=1e-9)


def test_rotvec_matches_scipy_if_available():
    scipy = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(1)
    for e in rng.uniform(-1.5, 1.5, size=(20, 3)):
        assert np.allclose(euler_xyz_to_rotvec(*e), scipy.Rotation.from_euler("xyz", e).as_rotvec(), atol=1e-9)


def test_widowx_env_action_and_bins():
    a = widowx_env_action([0.01, 0, -0.02, 0, 0, 0.1, 0.9])
    assert np.allclose(a, [0.01, 0, -0.02, 0, 0, 0.1, 1.0])
    assert widowx_env_action([0, 0, 0, 0, 0, 0, 0.2])[6] == -1.0
    assert list(to_bins([-1, 0, 1, 5])) == [0, 128, 255, 255]


# ---- stand-in env and policy ----

def test_dummy_env_reset_is_repeatable():
    env = DummyEnv()
    a = env.reset(3).copy()
    env.step(np.ones(7) * 0.1)
    assert np.array_equal(env.reset(3), a)
    assert not np.array_equal(env.reset(4), a)


def test_dummy_policy_is_deterministic_and_text_sensitive():
    env, pol = DummyEnv(), DummyPolicy()
    img = env.reset(0)
    a = pol.act(img, "put the spoon on the towel")
    assert np.array_equal(a.raw, pol.act(img, "put the spoon on the towel").raw)
    assert not np.array_equal(a.raw, pol.act(img, "place the spoon on the towel").raw)


# ---- end to end ----

def test_end_to_end_dry_run(tmp_path):
    common = ["--policy", "dummy", "--env", "dummy", "--episodes", "2", "--max-steps", "30",
              "--paraphrase-dir", str(ROOT / "paraphrases"), "--out", str(tmp_path)]
    run.main(["--mode", "probe", "--stride", "10"] + common)
    run.main(["--mode", "rollout"] + common)

    probe = [json.loads(l) for l in (tmp_path / "probe.jsonl").read_text().splitlines()]
    assert {r["instruction_id"] for r in probe} >= {"canonical", "verb_place"}
    frames = {(r["episode_id"], r["step"]) for r in probe}
    assert len(probe) == len(frames) * 13

    rolls = [json.loads(l) for l in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    by = {(r["episode_id"], r["instruction_id"]): r for r in rolls}
    for ep in (0, 1):
        # the control must reproduce the canonical run exactly
        assert by[(ep, "control")]["raw"] == by[(ep, "canonical")]["raw"]
        assert by[(ep, "control")]["path"] == by[(ep, "canonical")]["path"]

    text = analyze.render(tmp_path)
    assert "dry run" in text and "| control | 2 |" in text
    s = analyze.summarise_rollouts(rolls)
    assert all(s["per_category"]["control"]["never_diverged"])
    assert s["per_category"]["control"]["path_mean"] == [0.0, 0.0]
