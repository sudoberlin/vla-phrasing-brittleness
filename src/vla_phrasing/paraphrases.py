"""Paraphrase sets: one YAML file per task.

Every entry keeps the meaning of the canonical instruction and changes one
thing about its wording. The category says which thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

CATEGORIES = (
    "verb",        # different manipulation verb, same objects
    "object",      # synonym for an object, same verb
    "spatial",     # different spatial preposition or qualifier
    "qualifier",   # extra words that do not change the task
    "register",    # politeness or sentence form
    "structure",   # reordered or expanded sentence
)


@dataclass(frozen=True)
class Paraphrase:
    id: str
    category: str
    text: str


@dataclass(frozen=True)
class ParaphraseSet:
    task: str
    canonical: str
    paraphrases: List[Paraphrase]


def load_set(path) -> ParaphraseSet:
    data = yaml.safe_load(Path(path).read_text())
    items = [Paraphrase(p["id"], p["category"], p["text"]) for p in data["paraphrases"]]
    pset = ParaphraseSet(data["task"], data["canonical"], items)
    validate(pset)
    return pset


def validate(pset: ParaphraseSet) -> None:
    ids = [p.id for p in pset.paraphrases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{pset.task}: duplicate paraphrase ids")
    texts = [p.text.strip().lower() for p in pset.paraphrases]
    if len(set(texts)) != len(texts):
        raise ValueError(f"{pset.task}: duplicate paraphrase texts")
    if pset.canonical.strip().lower() in texts:
        raise ValueError(f"{pset.task}: a paraphrase repeats the canonical instruction")
    for p in pset.paraphrases:
        if p.category not in CATEGORIES:
            raise ValueError(f"{pset.task}/{p.id}: unknown category {p.category!r}")
        if not p.text.strip():
            raise ValueError(f"{pset.task}/{p.id}: empty text")


def find_set(task: str, root="paraphrases") -> ParaphraseSet:
    path = Path(root) / f"{task}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no paraphrase file for task {task!r} at {path}")
    return load_set(path)
