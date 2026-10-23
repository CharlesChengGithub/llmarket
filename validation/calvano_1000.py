"""Statistical validation: 1000 Calvano et al. (2020) baseline sessions against Cheng (2026).

Usage: python validation/calvano_1000.py [sessions] [--reference FILE]

Runs the shipped calvano2020 experiment on the fast engine and compares every session with the
recorded baseline of Cheng (2026), shipped as validation/calvano_1000_reference.csv (session,
steps, last state, last policy change, Delta): steps, last state and last policy change must be
identical and Delta must agree to 1e-12. Prints the mean Delta and its standard error for both,
and writes validation/calvano_1000_result.json. Takes about two minutes on a laptop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llmarket.core.config import load
from llmarket.core.experiment import run

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="?", type=int, default=1000)
    ap.add_argument("--reference", default=str(HERE / "calvano_1000_reference.csv"))
    args = ap.parse_args()
    n = args.sessions
    ref = pd.read_csv(args.reference, float_precision="round_trip").set_index("session").sort_index()
    cfg = load("calvano2020")
    started = time.perf_counter()
    res = run(cfg, out_dir=None, sessions=n)
    wall = time.perf_counter() - started
    ours = pd.DataFrame(res["conditions"][0]["sessions"]).set_index("session").sort_index()
    r = ref.loc[ours.index]
    same = {k: bool((ours[k].to_numpy() == r[k].to_numpy()).all()) for k in ("steps", "last_state", "last_policy_change")}
    max_delta_err = float(np.max(np.abs(ours["delta"].to_numpy() - r["delta"].to_numpy())))
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))  # noqa: E731
    out = dict(sessions=len(ours), identical=same, max_abs_delta_error=max_delta_err,
               mean_delta=float(ours["delta"].mean()), se_delta=se(ours["delta"]),
               paper14_mean_delta=float(r["delta"].mean()), paper14_se_delta=se(r["delta"]),
               converged_share=float(ours["converged"].mean()), wall_seconds=wall)
    print(json.dumps(out, indent=1))
    Path(__file__).with_name("calvano_1000_result.json").write_text(json.dumps(out, indent=1) + "\n")
    ok = all(same.values()) and max_delta_err <= 1e-12
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
