"""Turn raw records into tables.

    python -m vla_phrasing.analyze results/dry_run

Writes summary.md next to the records and prints it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .metrics import (action_token_variance, bin_distance, first_divergence_step,
                      token_disagreement, trajectory_divergence, wilson_interval)


def _read(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _fmt(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def summarise_probe(rows: List[dict]) -> Dict[str, object]:
    """Per category: how different is the action from the canonical one on the
    same frame. Plus the spread across the whole instruction set per frame."""
    frames = defaultdict(dict)
    for r in rows:
        frames[(r["task"], r["episode_id"], r["step"])][r["instruction_id"]] = r
    per_cat = defaultdict(lambda: defaultdict(list))
    per_id = defaultdict(lambda: defaultdict(list))
    variances = []
    for group in frames.values():
        base = group["canonical"]
        have_tokens = all(r["tokens"] is not None for r in group.values())
        if have_tokens:
            variances.append(action_token_variance([r["tokens"] for r in group.values()]))
        for iid, r in group.items():
            if iid == "canonical":
                continue
            l2 = float(np.linalg.norm(np.subtract(r["raw"][:6], base["raw"][:6])))
            grip = float(r["raw"][6] != base["raw"][6])
            dis = token_disagreement(r["tokens"], base["tokens"]) if have_tokens else None
            dist = bin_distance(r["tokens"], base["tokens"]) if have_tokens else None
            for store, key in ((per_cat, r["category"]), (per_id, (r["category"], iid, r["text"]))):
                store[key]["l2"].append(l2)
                store[key]["grip"].append(grip)
                store[key]["dis"].append(dis)
                store[key]["dist"].append(dist)
    return {"frames": len(frames), "token_variance": _mean(variances),
            "per_category": per_cat, "per_instruction": per_id}


def summarise_rollouts(rows: List[dict]) -> Dict[str, object]:
    episodes = defaultdict(dict)
    for r in rows:
        episodes[(r["task"], r["episode_id"])][r["instruction_id"]] = r
    per_cat = defaultdict(lambda: defaultdict(list))
    for group in episodes.values():
        base = group["canonical"]
        per_cat["canonical"]["success"].append(base["success"])
        for iid, r in group.items():
            if iid == "canonical":
                continue
            c = per_cat[r["category"]]
            c["success"].append(r["success"])
            c["flip"].append(r["success"] != base["success"])
            if r["path"] and base["path"]:
                c["path_mean"].append(trajectory_divergence(r["path"], base["path"])["mean"])
            if all(t is not None for t in r["tokens"]) and all(t is not None for t in base["tokens"]):
                step = first_divergence_step(r["tokens"], base["tokens"])
                c["never_diverged"].append(step is None)
                if step is not None:
                    c["first_step"].append(step)
    return {"episodes": len(episodes), "per_category": per_cat}


ORDER = ["canonical", "control", "verb", "object", "spatial", "qualifier", "register", "structure"]


def render(results_dir: Path) -> str:
    lines = [f"# Summary for `{results_dir.name}`", ""]
    cfg_path = results_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        lines += [f"Policy `{cfg.get('policy')}`, environment `{cfg.get('env')}`, task `{cfg.get('task')}`, "
                  f"canonical instruction \"{cfg.get('canonical_instruction')}\".", ""]
        if cfg.get("policy") == "dummy" or cfg.get("env") == "dummy":
            lines += ["**This is a dry run with stand-in components. The numbers test the pipeline and say "
                      "nothing about any real model.**", ""]
    probe = results_dir / "probe.jsonl"
    if probe.exists():
        s = summarise_probe(_read(probe))
        lines += ["## Same frame, different wording", "",
                  f"{s['frames']} frames. Mean variance of the action bin across the instruction set: "
                  f"{_fmt(s['token_variance'], 2)}.", "",
                  "| Category | Queries | Dimensions with a different bin | Mean bin distance | L2 on motion dims | Gripper flipped |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for cat in [c for c in ORDER if c in s["per_category"]]:
            d = s["per_category"][cat]
            lines.append(f"| {cat} | {len(d['l2'])} | {_fmt(_mean(d['dis']))} | {_fmt(_mean(d['dist']), 2)} | "
                         f"{_fmt(_mean(d['l2']), 4)} | {_fmt(_mean(d['grip']))} |")
        lines += ["", "### Per instruction", "",
                  "| Category | Instruction | Dimensions with a different bin | Mean bin distance |",
                  "| --- | --- | --- | --- |"]
        for (cat, _iid, text), d in sorted(s["per_instruction"].items(),
                                           key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
            lines.append(f"| {cat} | {text} | {_fmt(_mean(d['dis']))} | {_fmt(_mean(d['dist']), 2)} |")
        lines.append("")
    roll = results_dir / "rollouts.jsonl"
    if roll.exists():
        s = summarise_rollouts(_read(roll))
        lines += ["## Closed-loop rollouts", "", f"{s['episodes']} scene layouts.", "",
                  "| Category | Episodes | Success | 95% interval | Outcome differs from canonical | "
                  "Mean path distance | Mean first diverging step |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for cat in [c for c in ORDER if c in s["per_category"]]:
            d = s["per_category"][cat]
            k, n = int(sum(d["success"])), len(d["success"])
            lo, hi = wilson_interval(k, n)
            first = _fmt(_mean(d["first_step"]), 1) if d["first_step"] else ("never" if d["never_diverged"] else "n/a")
            flip = "" if cat == "canonical" else _fmt(_mean([float(x) for x in d["flip"]]), 2)
            path = "" if cat == "canonical" else _fmt(_mean(d["path_mean"]), 4)
            lines.append(f"| {cat} | {n} | {k}/{n} | {lo:.2f} to {hi:.2f} | {flip} | {path} | "
                         f"{'' if cat == 'canonical' else first} |")
        lines += ["", "The control row reruns the canonical instruction. It should match the canonical run "
                  "exactly. If it does not, the simulator or the policy is not deterministic and the other "
                  "rows cannot be read as an effect of wording.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    args = ap.parse_args(argv)
    d = Path(args.results_dir)
    text = render(d)
    (d / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
