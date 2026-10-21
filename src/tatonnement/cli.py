"""Command line: tatonnement run CONFIG [--out DIR] [--sessions N]."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tatonnement")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run an experiment configuration")
    p_run.add_argument("config")
    p_run.add_argument("--out", default="runs", help="output directory (default: runs)")
    p_run.add_argument("--sessions", type=int, default=None, help="override the number of sessions")
    p_val = sub.add_parser("validate", help="check a configuration without running it")
    p_val.add_argument("config")
    args = parser.parse_args(argv)

    from .core import config as cfgmod

    if args.command == "validate":
        cfg = cfgmod.load(args.config)
        print(json.dumps(dict(experiment=cfg["experiment"], conditions=len(cfgmod.expand(cfg)),
                              hash=cfgmod.config_hash(cfg)[:12])))
        return 0
    if args.command == "run":
        from .core.experiment import run

        cfg = cfgmod.load(args.config)
        res = run(cfg, out_dir=args.out, sessions=args.sessions)
        for c in res["conditions"]:
            d = [s.get("delta") for s in c["sessions"] if s.get("delta") is not None]
            mean = sum(d) / len(d) if d else float("nan")
            print(f"{c['condition']}: {len(c['sessions'])} sessions, mean Delta = {mean:.4f}")
        print(f"output: {res['directory']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
