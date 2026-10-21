"""Run configured experiments: conditions x sessions, with outputs and a manifest per run."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import time
from pathlib import Path

from .. import __version__
from .config import config_hash, expand
from .record import Recorder
from .simulation import Simulation


def _scenario(name: str):
    if name in ("bertrand_logit", "bertrand"):
        from ..scenarios.bertrand import experiment as mod

        return mod
    raise ValueError(f"unknown scenario {name!r}")


def _versions() -> dict:
    out = {"tatonnement": __version__, "python": platform.python_version()}
    for mod in ("numpy", "scipy", "numba", "httpx", "pandas", "pyarrow"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        here = Path(__file__).resolve().parent
        out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                           text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        out["git_commit"] = None
    return out


def run_condition(cfg: dict, sessions: list[int], out: Path | None, keep_periods: bool = True) -> list[dict]:
    scen = _scenario(cfg["market"].get("scenario", "bertrand_logit"))
    summaries = []
    cache: dict = {}
    if scen.use_fast_engine(cfg):
        for s in sessions:
            summaries.append(scen.fast_session(cfg, s, cache))
        return summaries
    for s in sessions:
        env, agents, kw = scen.build_session(cfg, s, cache)
        rec = Recorder(every=int(cfg.get("record_every", 1)), keep=keep_periods)
        started = time.perf_counter()
        res = Simulation(env, agents, cfg["periods"], seed=int(cfg.get("seed", 0)), session=s,
                         recorder=rec, **kw).run()
        summary = dict(session=s, stop_reason=res.stop_reason, wall_seconds=time.perf_counter() - started)
        if rec.rows:
            summary.update(scen.summarize(rec, env, cfg))
        summary.update(scen.agent_summaries(agents))
        summaries.append(summary)
        if out is not None and keep_periods and rec.rows:
            rec.write(out, stem=f"periods_s{s:04d}")
    return summaries


def run(cfg: dict, out_dir: str | Path | None = "runs", sessions: int | None = None) -> dict:
    """Run every condition of cfg; return {'conditions': [...], 'directory': path or None}."""
    started = _dt.datetime.now().astimezone()
    n_sessions = int(sessions if sessions is not None else cfg.get("sessions", 1))
    h = config_hash(cfg)
    base = None
    if out_dir is not None:
        stamp = started.strftime("%Y%m%d-%H%M%S")
        base = Path(out_dir) / cfg["experiment"] / f"{stamp}-{h[:8]}"
        base.mkdir(parents=True, exist_ok=True)
    results = []
    status = "complete"
    try:
        for c in expand(cfg):
            name = c.get("condition", "main")
            out = None if base is None else base / name.replace("/", "_")
            summaries = run_condition(c, list(range(n_sessions)), out)
            results.append(dict(condition=name, sessions=summaries))
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                with (out / "sessions.jsonl").open("w") as fh:
                    for row in summaries:
                        fh.write(json.dumps(row, default=str) + "\n")
    except BaseException as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        if base is not None:
            manifest = dict(experiment=cfg["experiment"], config=cfg, config_hash=h, sessions=n_sessions,
                            started=started.isoformat(), finished=_dt.datetime.now().astimezone().isoformat(),
                            status=status, versions=_versions(),
                            conditions=[r["condition"] for r in results])
            (base / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
    return dict(conditions=results, directory=None if base is None else str(base))
