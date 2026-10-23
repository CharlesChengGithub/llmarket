"""Build Bertrand sessions from configuration mappings.

Market keys: n, a, a0, mu, c, F (fixed cost per period), units {alpha, beta}.
Agent types: 'rule' (see llmarket.agents.rule.RULES), 'qlearning', and 'llm' (registered
by llmarket.scenarios.bertrand.llm_pricing). Q-learning keys: alpha, beta, beta_rule, nu,
delta, k, m, xi, grid, price_map, and pretrain {stable_steps, max_steps, mode: frozen|adaptive}.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ...agents.rule import RULES
from ...core.config import resolved_agents
from ...core.rng import kernel_seed
from .demand import Economy, Units
from .env import BertrandEnv
from .qlearning.agent import QLearningAgent, QLearningPricer, kernel_initial
from .regulations import PriceCap, PriceFloor, from_config

AGENT_BUILDERS: dict[str, Callable] = {}
QL_GRID_KEYS = ("m", "xi", "grid", "upper_bound", "k", "delta", "beta", "beta_rule", "nu", "beta_effective_m", "compression")


def register_agent(kind: str):
    def deco(fn):
        AGENT_BUILDERS[kind] = fn
        return fn
    return deco


def economy(market: dict) -> Economy:
    return Economy.from_config(market)


def make_env(cfg: dict, history_limit: int | None = None) -> BertrandEnv:
    m = cfg["market"]
    units = Units(**m.get("units", {}))
    regs = [from_config(r) for r in cfg.get("regulations", [])]
    return BertrandEnv(economy(m), regulations=regs, F=m.get("F", 0.0), units=units, history_limit=history_limit)


def fast_config(cfg: dict, spec: dict) -> dict:
    """Configuration for :func:`qlearning.fast.prepare` from the market, regulations and agent spec."""
    m = cfg["market"]
    out = {k: m[k] for k in ("n", "a", "a0", "mu", "c") if k in m}
    out.update({k: spec[k] for k in QL_GRID_KEYS if k in spec})
    floors = [r for r in cfg.get("regulations", []) if r["type"] == "price_floor"]
    if any(r["type"] == "price_cap" for r in cfg.get("regulations", [])):
        raise NotImplementedError("price caps are not yet supported for Q-learning agents")
    if len(floors) > 1:
        raise ValueError("at most one price floor per experiment")
    if floors:
        f = floors[0]
        floor = {"type": f.get("kind", "constant"), "F": m.get("F", 0.0)}
        if "value" in f:
            floor["value"] = f["value"]
        if floor["type"] in ("b1", "lambda"):
            floor["lambda"] = f.get("lam", 0.1)
        out["floor"] = floor
    return out


def all_qlearning(cfg: dict) -> bool:
    return all(a.get("type") == "qlearning" and "pretrain" not in a for a in resolved_agents(cfg))


@register_agent("rule")
def _rule(spec: dict, ctx: dict):
    params = {k: v for k, v in spec.items() if k not in ("type", "rule", "name")}
    return RULES[spec["rule"]](**params)


@register_agent("qlearning")
def _qlearning(spec: dict, ctx: dict):
    from .qlearning.fast import prepare, run_session

    cfg, i, session = ctx["cfg"], ctx["index"], ctx["session"]
    key = ("prep", repr(sorted(spec.items())))
    if key not in ctx["cache"]:
        ctx["cache"][key] = prepare(fast_config(cfg, spec))
    prep = ctx["prep"] = ctx["cache"][key]
    grid_market = ctx["grid_market"]
    kwargs = dict(alpha=float(spec.get("alpha", 0.15)), price_map=spec.get("price_map", "nearest"))
    Q = None
    pre = spec.get("pretrain")
    if pre:
        pkey = ("pretrain", key, session)
        if pkey not in ctx["cache"]:
            seed = kernel_seed(int(cfg.get("seed", 0)) + 1, session)
            ctx["cache"][pkey] = run_session(prep, seed, alpha=kwargs["alpha"],
                                             max_steps=int(pre.get("max_steps", 10**9)),
                                             stable_steps=int(pre.get("stable_steps", 100000)))
        trained = ctx["cache"][pkey]
        Q = trained["Q"]
        if pre.get("mode", "frozen") == "frozen":
            kwargs.update(learn=False, explore=False)
        else:
            kwargs.update(epsilon_start=trained["steps"])
    core = QLearningAgent.from_prepared(prep, i, Q=Q, use_starts=grid_market, **kwargs)
    return QLearningPricer(core)


def build_session(cfg: dict, session: int, cache: dict | None = None, runtime: dict | None = None):
    """Return (env, agents, simulation keyword arguments) for one session.

    runtime may carry shared run-level objects, e.g. {'llm': LLMService} for all sessions of a run.
    """
    cache = {} if cache is None else cache
    specs = resolved_agents(cfg)
    n = cfg["market"].get("n", 2)
    if len(specs) != n:
        raise ValueError(f"market has n={n} firms but {len(specs)} agents are configured")
    if any(s.get("type") == "llm" for s in specs):
        from . import llm_pricing  # noqa: F401  (registers the 'llm' builder)
    grid_market = all(s.get("type") == "qlearning" for s in specs)
    long_run = cfg["periods"] > 100000
    env = make_env(cfg, history_limit=1 if long_run else cfg.get("history_limit"))
    ctx = dict(cfg=cfg, session=session, cache=cache, grid_market=grid_market, runtime=runtime)
    agents = []
    for i, spec in enumerate(specs):
        ctx["index"] = i
        if spec["type"] not in AGENT_BUILDERS:
            raise ValueError(f"unknown agent type {spec['type']!r}; known: {sorted(AGENT_BUILDERS)}")
        agents.append(AGENT_BUILDERS[spec["type"]](spec, ctx))
    sim_kwargs = dict(rng_mode=cfg.get("rng_mode", "kernel" if grid_market else "independent"),
                      stop_when_stable=cfg.get("stop_when_stable"))
    llm_agents = [a for a in agents if hasattr(a, "projected_cost")]
    if llm_agents:
        ledger = llm_agents[0].service.ledger

        def guard(t, agents=llm_agents, ledger=ledger):
            return "budget_stopped" if ledger.would_exceed(sum(a.projected_cost for a in agents)) else None

        sim_kwargs["before_period"] = guard
        sim_kwargs["max_workers"] = int((cfg.get("llm") or {}).get("concurrency", len(llm_agents)))
    if grid_market and "prep" in ctx:
        sim_kwargs["initial"] = kernel_initial(ctx["prep"])
    elif "initial_prices" in cfg:
        sim_kwargs["initial"] = np.asarray(cfg["initial_prices"], dtype=float)
    return env, agents, sim_kwargs


__all__ = ["AGENT_BUILDERS", "register_agent", "make_env", "build_session", "fast_config", "PriceCap", "PriceFloor"]
