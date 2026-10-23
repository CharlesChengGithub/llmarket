"""Extract golden session results from the Cheng (2026) replication outputs.

Usage: python validation/extract_paper14_goldens.py <paper14_root> [out.json]

Writes, for a fixed list of conditions and sessions, the configuration and the recorded
seed, steps, last state, last policy change and Delta of each session. tests/test_fast_golden.py
re-runs these sessions with llmarket and requires identical results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# (results file, config file, condition, sessions)
SELECTION = [
    ("output/phase1/session_summary.parquet", "configs/replication.yaml", "baseline", range(20)),
    ("output/phase1/session_summary.parquet", "configs/replication.yaml", "memoryless", range(3)),
    ("output/phase1/session_summary.parquet", "configs/replication.yaml", "cost_050", range(3)),
    ("output/phase1/session_summary.parquet", "configs/replication.yaml", "n3", range(2)),
    ("output/phase1/session_summary.parquet", "configs/replication.yaml", "low15", range(3)),
    ("output/phase2/BC_session_summary.parquet", "configs/phase2.yaml", "B_cost_fixed", range(3)),
    ("output/phase2/BC_session_summary.parquet", "configs/phase2.yaml", "B_floor1.12_nu", range(3)),
    ("output/phase2/BC_session_summary.parquet", "configs/phase2.yaml", "C_b1_grid4", range(3)),
    ("output/phase2/BC_session_summary.parquet", "configs/phase2.yaml", "C_b2_grid4", range(3)),
    ("output/phase2/BC_session_summary.parquet", "configs/phase2.yaml", "C_lambda_0.50", range(3)),
]

# Keys that affect the simulation; bookkeeping keys are dropped.
MODEL_KEYS = {"n", "c", "a", "mu", "a0", "m", "xi", "grid", "upper_bound", "k", "delta", "alpha",
              "beta", "beta_rule", "nu", "beta_effective_m", "floor", "compression", "max_steps",
              "stable_steps", "epsilon_start"}


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tests/data/paper14_goldens.json")
    tables: dict[str, pd.DataFrame] = {}
    configs: dict[str, dict] = {}
    goldens = []
    for results, config_file, name, sessions in SELECTION:
        if results not in tables:
            tables[results] = pd.read_parquet(root / results)
        if config_file not in configs:
            data = yaml.safe_load((root / config_file).read_text())
            configs[config_file] = {c["name"]: c for c in data["conditions"]}
        cfg = {k: v for k, v in configs[config_file][name].items() if k in MODEL_KEYS}
        df = tables[results]
        rows = df[(df.condition == name) & (df.session.isin(list(sessions)))].sort_values("session")
        if len(rows) != len(sessions):
            raise SystemExit(f"{name}: expected {len(sessions)} sessions, found {len(rows)}")
        goldens.append(dict(
            condition=name,
            source=results,
            config=cfg,
            sessions=[dict(session=int(r.session), seed=int(r.seed), steps=int(r.steps),
                           converged=bool(r.converged), last_state=int(r.last_state),
                           last_policy_change=int(r.last_policy_change),
                           delta_gain=float(r.delta_gain), mean_price=float(r.mean_price),
                           cycle_length=int(r.cycle_length))
                      for r in rows.itertuples()],
        ))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(source="Cheng (2026) replication outputs", goldens=goldens), indent=1))
    print(f"wrote {out}: {sum(len(g['sessions']) for g in goldens)} sessions")


if __name__ == "__main__":
    main()
