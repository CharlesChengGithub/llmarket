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


def _uses_llm(cfg: dict) -> bool:
    from .config import resolved_agents

    return any(a.get("type") == "llm" for c in expand(cfg) for a in resolved_agents(c))


def _one_session(scen, cfg: dict, s: int, cache: dict, out: Path | None, keep_periods: bool,
                 runtime: dict | None) -> dict:
    env, agents, kw = scen.build_session(cfg, s, cache, runtime)
    rec = Recorder(every=int(cfg.get("record_every", 1)), keep=keep_periods)
    started = time.perf_counter()
    res = Simulation(env, agents, cfg["periods"], seed=int(cfg.get("seed", 0)), session=s,
                     recorder=rec, **kw).run()
    summary = dict(session=s, periods=res.periods, stop_reason=res.stop_reason,
                   wall_seconds=time.perf_counter() - started)
    if rec.rows:
        summary.update(scen.summarize(rec, env, cfg))
    summary.update(scen.agent_summaries(agents))
    if out is not None and keep_periods and (rec.rows or rec.events):
        rec.write(out, stem=f"periods_s{s:04d}")
    return summary


def run_condition(cfg: dict, sessions: list[int], out: Path | None, keep_periods: bool = True,
                  runtime: dict | None = None) -> list[dict]:
    """Run the sessions of one condition. LLM runs may set llm.parallel_sessions > 1; sessions
    still produce identical results because every call is keyed by its coordinates."""
    scen = _scenario(cfg["market"].get("scenario", "bertrand_logit"))
    cache: dict = {}
    if scen.use_fast_engine(cfg):
        return [scen.fast_session(cfg, s, cache) for s in sessions]
    service = (runtime or {}).get("llm")
    workers = int((cfg.get("llm") or {}).get("parallel_sessions", 1))

    def exhausted() -> bool:
        return service is not None and service.ledger.exhausted

    summaries = []
    if workers <= 1:
        for s in sessions:
            if exhausted():
                break
            summaries.append(_one_session(scen, cfg, s, cache, out, keep_periods, runtime))
        return summaries
    from concurrent.futures import ThreadPoolExecutor

    def job(s):
        return None if exhausted() else _one_session(scen, cfg, s, cache, out, keep_periods, runtime)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(job, sessions))
    return [r for r in results if r is not None]


def run(cfg: dict, out_dir: str | Path | None = "runs", sessions: int | None = None,
        runtime: dict | None = None) -> dict:
    """Run every condition of cfg; return {'conditions': [...], 'directory': path or None, ...}.

    runtime: optional shared objects; {'llm': LLMService} overrides the service built from cfg['llm']
    (tests and `tatonnement estimate` pass a mock-backed service this way).
    """
    started = _dt.datetime.now().astimezone()
    n_sessions = int(sessions if sessions is not None else cfg.get("sessions", 1))
    h = config_hash(cfg)
    base = None
    if out_dir is not None:
        stamp = started.strftime("%Y%m%d-%H%M%S")
        base = Path(out_dir) / cfg["experiment"] / f"{stamp}-{h[:8]}"
        base.mkdir(parents=True, exist_ok=True)
    runtime = dict(runtime or {})
    own_service = False
    if runtime.get("llm") is None and _uses_llm(cfg):
        from ..llm.service import LLMService

        runtime["llm"] = LLMService.from_config(cfg.get("llm"))
        own_service = True
    service = runtime.get("llm")
    results = []
    status = "complete"
    try:
        for c in expand(cfg):
            name = c.get("condition", "main")
            out = None if base is None else base / name.replace("/", "_")
            summaries = run_condition(c, list(range(n_sessions)), out, runtime=runtime)
            results.append(dict(condition=name, sessions=summaries))
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                with (out / "sessions.jsonl").open("w") as fh:
                    for row in summaries:
                        fh.write(json.dumps(row, default=str) + "\n")
            if service is not None and service.ledger.exhausted:
                break
        stops = [s.get("stop_reason") for r in results for s in r["sessions"]]
        if service is not None and (service.ledger.exhausted or "budget_stopped" in stops):
            status = "budget_stopped"
    except BaseException as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        llm_summary = service.summary() if service is not None else None
        if llm_summary is not None:
            llm_summary["cache_totals_for_experiment"] = service.cache.totals(cfg["experiment"])
        if base is not None:
            manifest = dict(experiment=cfg["experiment"], config=cfg, config_hash=h, sessions=n_sessions,
                            started=started.isoformat(), finished=_dt.datetime.now().astimezone().isoformat(),
                            status=status, versions=_versions(),
                            conditions=[r["condition"] for r in results])
            if llm_summary is not None:
                manifest["llm"] = llm_summary
            (base / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, ensure_ascii=False))
        if own_service:
            service.close()
    return dict(conditions=results, directory=None if base is None else str(base), status=status,
                llm=llm_summary)
