"""Hooks used by tatonnement.core.experiment for the Bertrand scenario."""
from __future__ import annotations

from ...core.config import resolved_agents
from ...core.rng import kernel_seed
from .build import all_qlearning, build_session, fast_config  # noqa: F401  (re-exported)
from .metrics import summarize as _summarize


def use_fast_engine(cfg: dict) -> bool:
    """Pure Q-learning self-play without caps runs on the Numba kernel unless engine: step."""
    engine = cfg.get("engine", "auto")
    if engine == "step":
        return False
    ok = all_qlearning(cfg) and not any(r["type"] == "price_cap" for r in cfg.get("regulations", []))
    if engine == "fast" and not ok:
        raise ValueError("engine 'fast' needs Q-learning agents only, without price caps")
    return ok


def fast_session(cfg: dict, session: int, cache: dict) -> dict:
    from .qlearning.analysis import analyze
    from .qlearning.fast import prepare, run_session

    specs = resolved_agents(cfg)
    spec = specs[0]
    if any({k: v for k, v in s.items()} != spec for s in specs[1:]):
        raise ValueError("the fast engine needs identically configured Q-learning agents")
    key = ("prep", repr(sorted(spec.items())))
    if key not in cache:
        cache[key] = prepare(fast_config(cfg, spec))
    prep = cache[key]
    seed = kernel_seed(int(cfg.get("seed", 0)), session)
    out = run_session(prep, seed, alpha=float(spec.get("alpha", 0.15)),
                      max_steps=int(cfg.get("periods", 10**9)),
                      stable_steps=int(cfg.get("stop_when_stable") or 100000))
    res = analyze(prep, out["policy"], out["last_state"], full=bool(cfg.get("full_analysis", False)))
    keep = ("cycle_length", "mean_price", "delta_gain", "cs", "symmetric_constant", "at_floor_exact",
            "path_nash", "nash_onpath_oneshot", "deviation_nonprofitable", "deviation_profit_change")
    row = dict(session=session, seed=seed, engine="fast", steps=out["steps"], converged=out["converged"],
               last_state=out["last_state"], last_policy_change=out["last_policy_change"])
    row.update({k: res[k] for k in keep if k in res})
    row["delta"] = row.pop("delta_gain")
    return row


def summarize(records, env, cfg: dict) -> dict:
    return _summarize(records, env, env.n_agents, (cfg.get("metrics") or {}).get("final_window"))


def agent_summaries(agents) -> dict:
    out = {}
    for i, a in enumerate(agents):
        stats = getattr(a, "stats", None)
        if callable(stats):
            out.update({f"agent{i}_{k}": v for k, v in stats().items()})
    return out
