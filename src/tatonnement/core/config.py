"""Experiment configuration: YAML loading, dotted-path overrides, sweeps and hashing."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED = ("experiment", "periods", "market", "agents")


def load(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config is missing {missing}")
    if not isinstance(cfg["agents"], list) or not cfg["agents"]:
        raise ValueError("agents must be a non-empty list")
    for j, a in enumerate(cfg["agents"]):
        if "copy_of" in a and not (0 <= int(a["copy_of"]) < j):
            raise ValueError(f"agent {j}: copy_of must refer to an earlier agent")


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg['a']['b'][0]... from 'a.b.0'; list indices are integers."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node.setdefault(k, {})
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def expand(cfg: dict) -> list[dict]:
    """Cartesian product of cfg['sweep'] (dotted path -> list of values); names get a suffix."""
    sweep = cfg.get("sweep") or {}
    base = {k: v for k, v in cfg.items() if k != "sweep"}
    if not sweep:
        return [copy.deepcopy(base)]
    paths = list(sweep)
    out = []
    for values in itertools.product(*(sweep[p] for p in paths)):
        c = copy.deepcopy(base)
        for p, v in zip(paths, values):
            set_path(c, p, v)
        c["condition"] = ",".join(f"{p.split('.')[-1]}={v}" for p, v in zip(paths, values))
        out.append(c)
    return out


def resolved_agents(cfg: dict) -> list[dict]:
    """Agent specs with copy_of expanded (a copy is a fresh agent with the same settings)."""
    specs = []
    for a in cfg["agents"]:
        if "copy_of" in a:
            src = copy.deepcopy(specs[int(a["copy_of"])])
            src.update({k: v for k, v in a.items() if k != "copy_of"})
            specs.append(src)
        else:
            specs.append(copy.deepcopy(a))
    return specs


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
