"""Experiment configuration: YAML loading, dotted-path overrides, conditions, sweeps and hashing."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED = ("experiment", "periods", "market", "agents")


def builtin_experiments() -> dict[str, str]:
    """Name -> description of every configuration shipped in tatonnement/experiments."""
    from importlib import resources

    out = {}
    for f in sorted(resources.files("tatonnement.experiments").iterdir(), key=lambda f: f.name):
        if f.name.endswith(".yaml"):
            out[f.name[:-5]] = str(yaml.safe_load(f.read_text()).get("description", ""))
    return out


def resolve(path: str | Path) -> str:
    """The text of a configuration: a file path, or the name of a shipped experiment."""
    p = Path(path)
    if p.is_file():
        return p.read_text()
    name = p.name[:-5] if p.name.endswith(".yaml") else p.name
    if p.parent == Path(".") and name in builtin_experiments():
        from importlib import resources

        return resources.files("tatonnement.experiments").joinpath(f"{name}.yaml").read_text()
    raise FileNotFoundError(f"{path}: no such file, and not a shipped experiment "
                            f"({', '.join(builtin_experiments())})")


def load(path: str | Path) -> dict:
    cfg = yaml.safe_load(resolve(path))
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
    """One config per condition.

    cfg['conditions'] lists named conditions, each a set of dotted-path overrides
    ({name: ..., set: {path: value}}), for designs that are not a full Cartesian product.
    cfg['sweep'] (dotted path -> list of values) takes the Cartesian product of its values. With
    both, every listed condition is crossed with every sweep combination. Names join with ','.
    """
    sweep = cfg.get("sweep") or {}
    listed = cfg.get("conditions") or []
    base = {k: v for k, v in cfg.items() if k not in ("sweep", "conditions")}
    names = [c.get("name") for c in listed]
    if any(not isinstance(n, str) or not n or "/" in n for n in names) or len(set(names)) != len(names):
        raise ValueError("every entry under conditions needs a unique name without '/'")
    if not sweep and not listed:
        return [copy.deepcopy(base)]
    paths = list(sweep)
    out = []
    for entry in listed or [None]:
        for values in itertools.product(*(sweep[p] for p in paths)):
            c = copy.deepcopy(base)
            parts = []
            if entry is not None:
                for p, v in (entry.get("set") or {}).items():
                    set_path(c, p, copy.deepcopy(v))
                parts.append(entry["name"])
            for p, v in zip(paths, values):
                set_path(c, p, v)
            parts += [f"{p.split('.')[-1]}={v}" for p, v in zip(paths, values)]
            c["condition"] = ",".join(parts)
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
