"""Run the evaluation and write raw records to a results directory.

Two modes.

probe    Roll out the canonical instruction once, keep the camera frame every
         few steps, then ask the policy for an action on each kept frame under
         every paraphrase. The image is identical across paraphrases, so the
         only thing that differs is the text. One forward pass per query.

rollout  Run a full closed-loop episode for every paraphrase from the same
         reset. Gives task success and how far the path drifts from the
         canonical one. The canonical instruction is run twice, and the second
         run is the control: any difference between the two is noise that has
         nothing to do with wording.

Example (no GPU needed):
    python -m vla_phrasing.run --mode probe --policy dummy --env dummy --out results/dry_run
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import __version__
from .envs import make_env
from .paraphrases import ParaphraseSet, find_set
from .policies import make_policy


def _instructions(pset: ParaphraseSet, canonical: str) -> List[Dict[str, str]]:
    rows = [{"id": "canonical", "category": "canonical", "text": canonical}]
    rows += [{"id": p.id, "category": p.category, "text": p.text} for p in pset.paraphrases]
    return rows


def _rollout(env, policy, instruction: str, episode_id: int, max_steps: int, keep_frames_every: int = 0):
    image = env.reset(episode_id)
    tokens, raws, path, frames = [], [], [], []
    success = False
    for t in range(max_steps):
        if keep_frames_every and t % keep_frames_every == 0:
            frames.append((t, image))
        out = policy.act(image, instruction)
        raws.append(out.raw.tolist())
        tokens.append(None if out.tokens is None else out.tokens.tolist())
        image, success = env.step(out.env_action)
        pos = env.ee_position()
        if pos is not None:
            path.append(pos.tolist())
        if success:
            break
    return {"success": bool(success), "steps": len(raws), "tokens": tokens, "raw": raws, "path": path}, frames


def run_probe(env, policy, pset, canonical, episode_ids, max_steps, stride, out):
    with open(out / "probe.jsonl", "w") as f:
        for ep in episode_ids:
            _, frames = _rollout(env, policy, canonical, ep, max_steps, keep_frames_every=stride)
            for t, image in frames:
                for ins in _instructions(pset, canonical):
                    o = policy.act(image, ins["text"])
                    f.write(json.dumps({
                        "task": pset.task, "episode_id": ep, "step": t,
                        "instruction_id": ins["id"], "category": ins["category"], "text": ins["text"],
                        "raw": o.raw.tolist(),
                        "tokens": None if o.tokens is None else o.tokens.tolist(),
                    }) + "\n")
            print(f"[probe] {pset.task} episode {ep}: {len(frames)} frames", flush=True)


def run_rollouts(env, policy, pset, canonical, episode_ids, max_steps, out):
    rows = _instructions(pset, canonical)
    rows.insert(1, {"id": "control", "category": "control", "text": canonical})
    with open(out / "rollouts.jsonl", "w") as f:
        for ep in episode_ids:
            for ins in rows:
                rec, _ = _rollout(env, policy, ins["text"], ep, max_steps)
                rec.update({"task": pset.task, "episode_id": ep, "instruction_id": ins["id"],
                            "category": ins["category"], "text": ins["text"]})
                f.write(json.dumps(rec) + "\n")
                print(f"[rollout] {pset.task} ep {ep} {ins['id']}: "
                      f"{'success' if rec['success'] else 'fail'} in {rec['steps']} steps", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["probe", "rollout"], required=True)
    ap.add_argument("--task", default="widowx_spoon_on_towel")
    ap.add_argument("--policy", choices=["dummy", "openvla"], default="dummy")
    ap.add_argument("--env", choices=["dummy", "simpler"], default="dummy")
    ap.add_argument("--episodes", type=int, default=3, help="number of scene layouts (episode ids 0..N-1)")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--stride", type=int, default=10, help="probe mode: keep one frame every this many steps")
    ap.add_argument("--paraphrase-dir", default="paraphrases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="openvla/openvla-7b")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--load-in-4bit", action="store_true", help="quantise OpenVLA to fit a 16 GB GPU")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pset = find_set(args.task, args.paraphrase_dir)
    env = make_env(args.env, args.task)
    policy = (make_policy("openvla", model_id=args.model_id, unnorm_key=args.unnorm_key,
                          load_in_4bit=args.load_in_4bit)
              if args.policy == "openvla" else make_policy("dummy"))

    env.reset(0)
    canonical = env.instruction()
    if canonical.strip().lower() != pset.canonical.strip().lower():
        print(f"WARNING: the environment's instruction is {canonical!r} but the paraphrase file was "
              f"written for {pset.canonical!r}. Using the environment's. Check the paraphrases still match.")

    config = vars(args) | {
        "canonical_instruction": canonical, "version": __version__,
        "python": platform.python_version(), "numpy": np.__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    episode_ids = list(range(args.episodes))
    if args.mode == "probe":
        run_probe(env, policy, pset, canonical, episode_ids, args.max_steps, args.stride, out)
    else:
        run_rollouts(env, policy, pset, canonical, episode_ids, args.max_steps, out)
    print(f"done. raw records in {out}/")


if __name__ == "__main__":
    main()
