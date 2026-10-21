"""Cost estimates before spending money: run the experiment offline with mock answers, assemble
every prompt exactly as a live run would, count tokens, and price them.

Prompt tokens come from tiktoken (cl100k_base) when it is installed, otherwise from the price
row's characters-per-token ratio. The estimate assumes no format retries, no provider-side prefix
caching and no hidden reasoning tokens, and reports a +/-50% range around the central value.
"""
from __future__ import annotations

import copy
import math
import threading

from .ledger import DEFAULT_CHARS_PER_TOKEN, Ledger, PriceRow, PriceTable
from .mock import MockClient
from .providers import Provider
from .service import LLMService
from .types import ChatRequest, ChatResponse


def _token_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return "tiktoken:cl100k_base", lambda text: len(enc.encode(text))
    except Exception:  # noqa: BLE001  (missing package or missing encoding file)
        return None, None


class EstimateLedger(Ledger):
    """Never refuses a call; records prompt and answer sizes by provider and model."""

    def __init__(self, table: PriceTable):
        super().__init__(table)
        self.method, self.counter = _token_counter()
        self.usage: dict[tuple, dict] = {}
        self._elock = threading.Lock()

    def row(self, provider: Provider, model: str) -> PriceRow:
        r = self.table.lookup(provider.name, model)
        if r is None:
            return PriceRow(provider.name, model, None, 0.0, 0.0, 0.0, source="unknown" if provider.paid else "unpaid")
        return r

    def check(self, request: ChatRequest, provider: Provider) -> float:
        return 0.0

    def charge(self, provider: Provider, request: ChatRequest, response: ChatResponse) -> float:
        known = self.table.lookup(provider.name, request.model)
        if self.counter is not None:
            tokens = sum(self.counter(str(m.get("content", ""))) for m in request.messages)
        else:
            cpt = known.chars_per_token if known else DEFAULT_CHARS_PER_TOKEN
            tokens = request.prompt_chars() / cpt
        with self._elock:
            u = self.usage.setdefault((provider.name, request.model), dict(calls=0, prompt_tokens=0.0,
                                                                          completion_tokens=0, paid=provider.paid))
            u["calls"] += 1
            u["prompt_tokens"] += tokens
            u["completion_tokens"] += response.completion_tokens
        return 0.0


def estimate(cfg: dict, sample_sessions: int = 2, output_tokens: int | None = None) -> dict:
    """Estimate the calls, tokens and cost of running cfg (all conditions, all sessions)."""
    from ..core.config import expand, resolved_agents
    from ..core.experiment import run_condition

    llm = cfg.get("llm") or {}
    table = PriceTable(llm.get("prices"))
    ledger = EstimateLedger(table)
    specs = [a for c in expand(cfg) for a in resolved_agents(c) if a.get("type") == "llm"]
    if not specs:
        raise ValueError("the configuration has no LLM agents")
    out_tokens = int(output_tokens or max(int(a.get("max_tokens") or 0) for a in specs) or 400)
    mock = dict(dict(policy="follow", noise=0.02), **(llm.get("mock") or {}))
    mock.update(completion_tokens=out_tokens, error_rate=0.0, fail_after=None)
    service = LLMService(ledger=ledger, client_override=MockClient(**mock))
    n_sessions = int(cfg.get("sessions", 1))
    k = max(1, min(int(sample_sessions), n_sessions))
    scale = n_sessions / k
    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("llm", {})["parallel_sessions"] = 1
    for c in expand(run_cfg):
        before = {key: dict(v) for key, v in ledger.usage.items()}
        run_condition(c, list(range(k)), None, keep_periods=True, runtime={"llm": service})
        for key, u in ledger.usage.items():  # scale this condition's sample up to all sessions
            b = before.get(key, dict(calls=0, prompt_tokens=0.0, completion_tokens=0))
            for f in ("calls", "prompt_tokens", "completion_tokens"):
                u[f] = b[f] + (u[f] - b[f]) * scale
    rows = []
    totals: dict = {}
    for (prov, model), u in sorted(ledger.usage.items()):
        price = table.lookup(prov, model)
        known = price is not None or not u["paid"]
        mid = price.cost(int(u["prompt_tokens"]), 0, int(u["completion_tokens"])) if price else (0.0 if known else math.nan)
        cur = price.currency if price else None
        rows.append(dict(provider=prov, model=model, calls=round(u["calls"]), prompt_tokens=round(u["prompt_tokens"]),
                         completion_tokens=round(u["completion_tokens"]), currency=cur, price_known=known,
                         cost_low=mid * 0.5, cost_mid=mid, cost_high=mid * 1.5,
                         price_source=price.source if price else None, price_date=price.date if price else None,
                         price_note=price.note if price else None))
        if cur and not math.isnan(mid):
            totals[cur] = totals.get(cur, 0.0) + mid
    return dict(experiment=cfg["experiment"], sessions=n_sessions, sampled_sessions=k, conditions=len(expand(cfg)),
                periods=cfg["periods"], output_tokens_per_call=out_tokens,
                token_method=ledger.method or "chars_per_token", rows=rows,
                totals={c: dict(low=v * 0.5, mid=v, high=v * 1.5) for c, v in totals.items()},
                assumptions=["no format retries", "no provider-side prefix caching (it would lower input cost)",
                             "no hidden reasoning tokens (reasoning models can use many more output tokens)",
                             "price rows as listed; DeepSeek rows are peak-hour rates"])


def format_estimate(est: dict) -> str:
    lines = [f"{est['experiment']}: {est['conditions']} condition(s) x {est['sessions']} session(s) x "
             f"{est['periods']} periods (sampled {est['sampled_sessions']} session(s) per condition; "
             f"{est['output_tokens_per_call']} output tokens per call; tokens by {est['token_method']})"]
    for r in est["rows"]:
        head = f"  {r['provider']}/{r['model']}: {r['calls']:,} calls, {r['prompt_tokens']:,} prompt + " \
               f"{r['completion_tokens']:,} output tokens"
        if not r["price_known"]:
            lines.append(head + " -- no price row: add one under llm.prices before running")
        elif r["currency"] is None:
            lines.append(head + " -- free")
        else:
            lines.append(head + f" -- {r['cost_mid']:.2f} {r['currency']} "
                         f"(range {r['cost_low']:.2f} to {r['cost_high']:.2f}; price as of {r['price_date']})")
    for cur, t in est["totals"].items():
        lines.append(f"  total: {t['mid']:.2f} {cur} (range {t['low']:.2f} to {t['high']:.2f})")
    lines.append("  assumes: " + "; ".join(est["assumptions"]))
    return "\n".join(lines)
