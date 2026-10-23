"""Command line.

  llmarket list
  llmarket run CONFIG [--out DIR] [--sessions N]
  llmarket validate CONFIG
  llmarket estimate CONFIG [--sample-sessions N] [--output-tokens N] [--json]
  llmarket cache-export --experiment NAME --out FILE [--cache PATH] [--condition NAME]

CONFIG is a YAML file or the name of a shipped experiment (see `llmarket list`).
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmarket")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the experiments shipped with the package")
    p_run = sub.add_parser("run", help="run an experiment configuration")
    p_run.add_argument("config")
    p_run.add_argument("--out", default="runs", help="output directory (default: runs)")
    p_run.add_argument("--sessions", type=int, default=None, help="override the number of sessions")
    p_val = sub.add_parser("validate", help="check a configuration without running it")
    p_val.add_argument("config")
    p_est = sub.add_parser("estimate", help="estimate LLM calls, tokens and cost offline")
    p_est.add_argument("config")
    p_est.add_argument("--sample-sessions", type=int, default=2, help="sessions simulated per condition")
    p_est.add_argument("--output-tokens", type=int, default=None, help="answer length per call (tokens)")
    p_est.add_argument("--json", action="store_true", help="print the estimate as JSON")
    p_exp = sub.add_parser("cache-export", help="copy one experiment's LLM calls into a standalone cache file")
    p_exp.add_argument("--experiment", required=True)
    p_exp.add_argument("--out", required=True)
    p_exp.add_argument("--cache", default=None, help="source cache (default: .llmarket/cache.sqlite)")
    p_exp.add_argument("--condition", default=None)
    args = parser.parse_args(argv)

    if args.command == "cache-export":
        from .llm.cache import Cache

        n = Cache(args.cache, "use").export(args.out, args.experiment, args.condition)
        print(f"exported {n} calls to {args.out}")
        return 0

    from .core import config as cfgmod

    if args.command == "list":
        for name, line in cfgmod.builtin_experiments().items():
            print(f"{name:22s} {line}")
        return 0

    if args.command == "validate":
        cfg = cfgmod.load(args.config)
        print(json.dumps(dict(experiment=cfg["experiment"], conditions=len(cfgmod.expand(cfg)),
                              hash=cfgmod.config_hash(cfg)[:12])))
        return 0
    if args.command == "estimate":
        from .llm.estimate import estimate, format_estimate

        est = estimate(cfgmod.load(args.config), args.sample_sessions, args.output_tokens)
        print(json.dumps(est, indent=1, default=str) if args.json else format_estimate(est))
        return 0
    if args.command == "run":
        from .core.experiment import run

        cfg = cfgmod.load(args.config)
        res = run(cfg, out_dir=args.out, sessions=args.sessions)
        if res.get("llm"):
            llm = res["llm"]
            spent = f"{llm['spent']:.4f} {llm['currency'] or ''}".strip()
            print(f"LLM: {llm['live_calls']} live calls, {llm['cache_hits']} cache hits, spent {spent}")
        if res["status"] != "complete":
            print(f"status: {res['status']}")
        for c in res["conditions"]:
            d = [s.get("delta") for s in c["sessions"] if s.get("delta") is not None]
            mean = sum(d) / len(d) if d else float("nan")
            print(f"{c['condition']}: {len(c['sessions'])} sessions, mean Delta = {mean:.4f}")
        print(f"output: {res['directory']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
