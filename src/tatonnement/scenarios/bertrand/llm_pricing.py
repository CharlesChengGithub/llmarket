"""LLM pricing agents for the Bertrand market, and the 'llm' agent builder.

Agent keys (config): provider, model, temperature, top_p, max_tokens, extra_body, prompt (pack
name, {file: path} or a mapping), prefix (a pack prefix name or literal text), prefix_placement
(user | system), instructions (list of sentences, or a key of the pack's instruction sets),
deployment {...}, ceiling_multiplier (number or [low, high], drawn once per session),
price_range [low, high] (display units; the agent clips its own answer), retries, on_failure,
seed, record_prompt.

Everything the model sees is in display units (Units: prices x alpha, quantities x beta).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ...agents.llm_agent import LLMAgent, Parsed
from ...llm.parsing import parse_price
from ...llm.service import LLMService
from .build import register_agent
from .prompts import DHANDA_INSTRUCTIONS, PromptPack, get_pack

HISTORY_FIELDS = ("price", "rivals", "quantity", "profit")


@dataclass
class Deployment:
    """How the agent is deployed (Bichler et al. 2025 call these deployment choices).

    output: 'point' (one price); 'distribution' is planned for v0.2.
    history: 'raw' (one entry per round) or 'summary' (per own price level: count, mean quantity,
      mean profit, mean rival price).
    window: number of past rounds shown (None: the pack's default).
    rival_visible: show rivals' prices.
    interval: call the model every k periods and keep the price in between.
    decimals: rounding of displayed numbers.
    show: which history fields to show (default: the pack's).
    order: 'oldest_first' or 'newest_first'.
    own_price: 'posted' (after enforcement) or 'proposed' (what the agent asked for).
    """

    output: str = "point"
    history: str = "raw"
    window: int | None = None
    rival_visible: bool = True
    interval: int = 1
    decimals: int = 2
    show: tuple | None = None
    order: str = "oldest_first"
    own_price: str = "posted"

    @classmethod
    def from_config(cls, d: dict | None) -> "Deployment":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown deployment keys {sorted(unknown)}")
        if "show" in d and d["show"] is not None:
            d["show"] = tuple(d["show"])
        dep = cls(**d)
        if dep.output != "point":
            raise NotImplementedError("deployment.output='distribution' is planned for v0.2")
        if dep.history not in ("raw", "summary"):
            raise ValueError("deployment.history must be 'raw' or 'summary'")
        if dep.order not in ("oldest_first", "newest_first"):
            raise ValueError("deployment.order must be 'oldest_first' or 'newest_first'")
        if dep.own_price not in ("posted", "proposed"):
            raise ValueError("deployment.own_price must be 'posted' or 'proposed'")
        return dep


class PricingLLMAgent(LLMAgent):
    def __init__(self, service: LLMService, provider, model: str, pack: PromptPack | str = "fish_style",
                 deployment: Deployment | dict | None = None, prefix: str | None = None,
                 prefix_placement: str = "user", instructions: list | str | None = None,
                 ceiling_multiplier: float = 2.0, price_range: tuple | None = None, require_all_fields: bool = True,
                 pick: str | None = None, **kwargs):
        self.pack = get_pack(pack)
        self.pick = pick or self.pack.pick
        self.deployment = deployment if isinstance(deployment, Deployment) else Deployment.from_config(deployment)
        super().__init__(service, provider, model, memory_fields=tuple(self.pack.memory_fields),
                         interval=self.deployment.interval, **kwargs)
        if prefix_placement not in ("user", "system"):
            raise ValueError("prefix_placement must be 'user' or 'system'")
        name = prefix if prefix is not None else self.pack.default_prefix
        self.prefix = self.pack.prefixes.get(name, name) if name is not None else ""
        self.prefix_placement = prefix_placement
        if isinstance(instructions, str):
            instructions = DHANDA_INSTRUCTIONS[instructions] if instructions in DHANDA_INSTRUCTIONS else [instructions]
        self.instructions = list(instructions or [])
        self.ceiling_multiplier = float(ceiling_multiplier)
        rng = price_range if price_range is not None else self.pack.price_range
        self.price_range = None if rng is None else (float(rng[0]), float(rng[1]))
        self.require_all_fields = require_all_fields
        show = self.deployment.show if self.deployment.show is not None else self.pack.history_fields
        if not self.deployment.rival_visible:
            show = tuple(f for f in show if f != "rivals")
        unknown = set(show) - set(HISTORY_FIELDS)
        if unknown:
            raise ValueError(f"unknown history fields {sorted(unknown)}")
        self.show = tuple(show)
        self.window = self.deployment.window or self.pack.window

    # -- units ------------------------------------------------------------------------------
    def reset(self, ctx) -> None:
        super().reset(ctx)
        u = ctx.market["units"]
        self.alpha, self.beta = float(u["alpha"]), float(u["beta"])
        i = ctx.index
        self.cost = self.alpha * float(ctx.market["cost"][i])
        self.nash = self.alpha * float(ctx.market["nash_prices"][i])
        self.monopoly = self.alpha * float(ctx.market["monopoly_prices"][i])
        self.ceiling = self.ceiling_multiplier * self.alpha * float(max(ctx.market["monopoly_prices"]))
        self.parse_upper = 10 * self.alpha * float(max(ctx.market["monopoly_prices"]))
        init = ctx.market.get("initial_prices")
        self.start_price = float(init[i]) if init is not None else float(ctx.market["nash_prices"][i])

    def to_model_units(self, value: float) -> float:
        return float(value) / self.alpha

    def _fmt(self, x: float) -> str:
        return f"{x:.{self.deployment.decimals}f}"

    # -- prompt -------------------------------------------------------------------------------
    def history_text(self, obs) -> str:
        h = obs.history
        own_key = "posted" if self.deployment.own_price == "posted" else "proposed"
        T = len(h["posted"])
        if T == 0:
            return self.pack.empty_history
        start = max(T - self.window, 0)
        i = self.index
        rows = []
        for t in range(start, T):
            rivals = np.delete(h["posted"][t], i) * self.alpha
            rows.append(dict(round=t + 1, price=h[own_key][t][i] * self.alpha, rivals=rivals,
                             quantity=h["quantity"][t][i] * self.beta,
                             profit=h["profit"][t][i] * self.alpha * self.beta))
        if self.deployment.history == "summary":
            return self._summary(rows)
        if self.deployment.order == "newest_first":
            rows = rows[::-1]
        return "\n".join(self._row(r) for r in rows)

    def _rival_label(self, r) -> str:
        lab = self.pack.labels
        return lab.get("rival", "Competitor's price") if len(r["rivals"]) == 1 else lab.get("rivals", "Competitors' prices")

    def _row(self, r: dict) -> str:
        lab = self.pack.labels
        parts = []
        for f in self.show:
            if f == "rivals":
                parts.append((self._rival_label(r), ", ".join(self._fmt(x) for x in r["rivals"])))
            else:
                parts.append((lab.get(f, f), self._fmt(r[f])))
        if self.pack.history_style == "lines":
            return f"Round {r['round']}: " + ", ".join(f"{k} {v}" for k, v in parts)
        return f"Round {r['round']}:\n" + "\n".join(f"- {k}: {v}" for k, v in parts)

    def _summary(self, rows: list[dict]) -> str:
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(self._fmt(r["price"]), []).append(r)
        lines = [f"Summary of the last {len(rows)} rounds, grouped by the price you set:"]
        for key in sorted(groups, key=float):
            g = groups[key]
            item = [f"set in {len(g)} round{'s' if len(g) > 1 else ''}"]
            if "quantity" in self.show:
                item.append(f"average quantity sold {self._fmt(np.mean([r['quantity'] for r in g]))}")
            if "profit" in self.show:
                item.append(f"average profit {self._fmt(np.mean([r['profit'] for r in g]))}")
            if "rivals" in self.show:
                item.append(f"average {self._rival_label(g[0]).lower()} "
                            f"{self._fmt(np.mean([np.mean(r['rivals']) for r in g]))}")
            lines.append(f"- Price {key}: " + "; ".join(item))
        return "\n".join(lines)

    def memory_text(self) -> str:
        out = []
        for f in self.pack.memory_fields:
            label = self.pack.memory_labels.get(f, f)
            out.append(f"{label}:\n{self.memory.get(f) or self.pack.empty_memory}")
        return "\n".join(out)

    def messages(self, obs, retry: str | None) -> list[dict]:
        bullet = "- " if self.pack.history_style == "blocks" else ""
        regulation = "".join(f"{bullet}{d}\n" for d in obs.disclosures)
        instructions = "".join(f"{s}\n" for s in self.instructions)
        lo, hi = self.price_range if self.price_range else (float("nan"), float("nan"))
        values = dict(prefix=self.prefix if self.prefix_placement == "user" else "", cost=self._fmt(self.cost),
                      ceiling=self._fmt(self.ceiling), range_lo=self._fmt(lo), range_hi=self._fmt(hi),
                      regulation=regulation, instructions=instructions, memory=self.memory_text(),
                      history=self.history_text(obs), retry=f"{retry}\n" if retry else "", round=obs.t + 1)
        user = self.pack.user.format(**values).lstrip("\n")
        msgs = []
        system = self.pack.system.format(**values) if self.pack.system else ""
        if self.prefix_placement == "system" and self.prefix:
            system = f"{self.prefix}\n{system}".strip()
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return msgs

    def hint(self, obs, retry: bool) -> dict:
        h = obs.history
        own_last = rivals_last = None
        if len(h["posted"]):
            own_last = float(h["posted"][-1][self.index] * self.alpha)
            rivals_last = [float(x) for x in np.delete(h["posted"][-1], self.index) * self.alpha]
        lo, hi = obs.bounds.lower * self.alpha, obs.bounds.upper * self.alpha
        if self.price_range:
            lo, hi = max(lo, self.price_range[0]), min(hi, self.price_range[1])
        return dict(format=dict(fields=list(self.pack.fields), price_field=self.pack.price_field),
                    market=dict(t=obs.t, cost=self.cost, nash=self.nash, monopoly=self.monopoly, own_last=own_last,
                                rivals_last=rivals_last, lower=float(lo), upper=float(hi), retry=retry))

    # -- answers ------------------------------------------------------------------------------
    def parse(self, text: str, obs) -> Parsed:
        res = parse_price(text, self.pack.fields, self.pack.price_field, self.parse_upper, pick=self.pick)
        if res.value is None:
            return Parsed(None, res.status)
        memory = None
        if self.pack.memory_fields:
            missing = [f for f in self.pack.memory_fields if f not in res.sections]
            if missing and self.require_all_fields:
                return Parsed(None, "missing_field")
            memory = {f: res.sections.get(f, "") for f in self.pack.memory_fields}
        value = res.value
        detail = {"raw_price": value}
        if self.price_range is not None:
            clipped = min(max(value, self.price_range[0]), self.price_range[1])
            detail["range_clipped"] = clipped != value
            value = clipped
        return Parsed(value, "ok", memory, detail)

    def fallback(self, obs) -> float:
        h = obs.history
        if len(h["posted"]):
            return float(h["posted"][-1][self.index])
        return self.start_price


def ceiling_multiplier(spec, root: int, session: int) -> float:
    """Fish et al. draw the reported ceiling multiplier once per run from U[1.5, 2.5]."""
    m = spec.get("ceiling_multiplier", [1.5, 2.5])
    if isinstance(m, (int, float)):
        return float(m)
    ss = np.random.SeedSequence(int(root), spawn_key=(int(session), 2**20))
    return float(np.random.default_rng(ss).uniform(float(m[0]), float(m[1])))


_LLM_PARAMS = ("temperature", "top_p", "max_tokens", "extra_body", "presence_penalty", "frequency_penalty", "stop")


def service_for(ctx: dict) -> LLMService:
    runtime = ctx.get("runtime") or {}
    if runtime.get("llm") is not None:
        return runtime["llm"]
    cache = ctx["cache"]
    if "llm_service" not in cache:
        cache["llm_service"] = LLMService.from_config(ctx["cfg"].get("llm"))
    return cache["llm_service"]


@register_agent("llm")
def _llm(spec: dict, ctx: dict):
    cfg = ctx["cfg"]
    if any("key" in str(k).lower() for k in list(spec) + list(cfg.get("llm") or {})):
        raise ValueError("API keys do not belong in configurations: set the provider's environment variable "
                         "(for example DEEPSEEK_API_KEY) or name another one with provider.key_env")
    service = service_for(ctx)
    failure = (cfg.get("llm") or {}).get("on_parse_failure") or {}
    params = {k: spec[k] for k in _LLM_PARAMS if k in spec}
    params.setdefault("temperature", 1.0)
    return PricingLLMAgent(
        service, spec.get("provider", "mock"), spec.get("model", "mock"), pack=spec.get("prompt", "fish_style"),
        deployment=spec.get("deployment"), prefix=spec.get("prefix"),
        prefix_placement=spec.get("prefix_placement", "user"), instructions=spec.get("instructions"),
        ceiling_multiplier=ceiling_multiplier(spec, int(cfg.get("seed", 0)), ctx["session"]),
        price_range=spec.get("price_range"), require_all_fields=bool(spec.get("require_all_fields", True)),
        pick=spec.get("pick"), params=params, namespace=cfg["experiment"], condition=cfg.get("condition", "main"),
        seed_root=int(cfg.get("seed", 0)), retries=int(spec.get("retries", failure.get("retries", 10))),
        on_failure=spec.get("on_failure", failure.get("then", "stop")), seed=spec.get("seed", "auto"),
        record_prompt=bool(spec.get("record_prompt", (cfg.get("llm") or {}).get("record_prompts", False))),
    )
