"""Generic LLM agent: prompt assembly is left to subclasses; this class handles calls, call
coordinates, format retries, fallbacks, memory slots and bookkeeping."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..core.types import Action, StopSession
from ..llm.service import LLMService
from ..llm.types import BudgetExceeded, CallCoords, ChatRequest, ChatResponse, UnknownPrice

FAILURE_POLICIES = ("stop", "keep_last_price")


@dataclass
class Parsed:
    value: float | None
    status: str
    memory: dict | None = None
    detail: dict | None = None


class LLMAgent:
    """Subclasses implement messages(obs, retry), parse(text, obs) and fallback(obs).

    retries: format retries after the first unusable answer (Fish et al. use 10).
    on_failure: 'stop' ends the session (Fish et al.); 'keep_last_price' repeats the last price.
    seed: 'auto' sends a per-call seed derived from the call coordinates to providers that accept
      one (reproducible local models); an integer sends that seed; None sends none.
    record_prompt: keep the full prompt in the text log (the cache keeps it in any case).
    """

    concurrent = True
    learning = False
    changed = False
    can_retry = True

    def __init__(self, service: LLMService, provider, model: str, params: dict | None = None,
                 namespace: str = "", condition: str = "", seed_root: int = 0, retries: int = 10,
                 on_failure: str = "stop", seed: int | str | None = "auto", record_prompt: bool = False,
                 interval: int = 1, memory_fields: tuple[str, ...] = ()):
        if on_failure not in FAILURE_POLICIES:
            raise ValueError(f"on_failure must be one of {FAILURE_POLICIES}")
        self.service = service
        self.provider = service.provider(provider)
        self.model = model
        self.params = dict(params or {})
        self.namespace = namespace
        self.condition = condition
        self.seed_root = int(seed_root)
        self.retries = int(retries)
        self.on_failure = on_failure
        self.seed = seed
        self.record_prompt = record_prompt
        self.interval = int(interval)
        self.memory_fields = tuple(memory_fields)
        self.projected_cost = 0.0

    # -- lifecycle ------------------------------------------------------------------------
    def reset(self, ctx) -> None:
        self.index = ctx.index
        self.session = ctx.session
        self.market = ctx.market
        self.rng = ctx.rng
        self.memory = {f: "" for f in self.memory_fields}
        self._attempts: dict[int, int] = {}
        self.totals = dict(calls=0, live_calls=0, cached_calls=0, parse_failures=0, fallbacks=0,
                           cost=0.0, prompt_tokens=0, completion_tokens=0)

    def update(self, fb) -> None:
        pass

    def stats(self) -> dict:
        return dict(self.totals)

    # -- calls --------------------------------------------------------------------------------
    def _coords(self, t: int) -> CallCoords:
        k = self._attempts.get(t, 0)
        self._attempts[t] = k + 1
        return CallCoords(self.namespace, self.condition, self.seed_root, self.session, self.index, t, k)

    def request(self, messages: list[dict], coords: CallCoords) -> ChatRequest:
        params = {k: v for k, v in self.params.items() if v is not None}
        if self.seed == "auto":
            if self.provider.supports_seed:
                params["seed"] = coords.call_seed()
        elif self.seed is not None:
            params["seed"] = int(self.seed)
        return ChatRequest(self.provider.name, self.model, tuple(messages), params)

    def ask(self, t: int, messages: list[dict], hint: dict | None = None) -> tuple[ChatResponse, ChatRequest]:
        coords = self._coords(t)
        req = self.request(messages, coords)
        try:
            ledger = self.service.ledger
            self.projected_cost = ledger.estimate(req, ledger.row(self.provider, self.model))
        except UnknownPrice:  # only live calls need a price; replayed ones are free
            self.projected_cost = 0.0
        try:
            resp = self.service.call(self.provider, req, coords, hint)
        except BudgetExceeded as exc:
            raise StopSession("budget_stopped", str(exc)) from exc
        tot = self.totals
        tot["calls"] += 1
        tot["cached_calls" if resp.cached else "live_calls"] += 1
        tot["cost"] += resp.cost
        tot["prompt_tokens"] += resp.prompt_tokens
        tot["completion_tokens"] += resp.completion_tokens
        return resp, req

    # -- subclass interface ---------------------------------------------------------------
    def messages(self, obs, retry: str | None) -> list[dict]:
        raise NotImplementedError

    def hint(self, obs, retry: bool) -> dict:
        return {}

    def parse(self, text: str, obs) -> Parsed:
        raise NotImplementedError

    def fallback(self, obs) -> float:
        raise NotImplementedError

    def to_model_units(self, value: float) -> float:
        return value

    # -- acting ---------------------------------------------------------------------------
    def act(self, obs) -> Action:
        retry = obs.retry_message
        failures = 0
        cost = 0.0
        tokens_in = tokens_out = 0
        latency = 0.0
        cached = 0
        statuses = []
        last_resp = None
        for _attempt in range(self.retries + 1):
            msgs = self.messages(obs, retry)
            resp, req = self.ask(obs.t, msgs, self.hint(obs, retry is not None or failures > 0))
            last_resp = resp
            cost += resp.cost
            tokens_in += resp.prompt_tokens
            tokens_out += resp.completion_tokens
            latency += resp.latency
            cached += int(resp.cached)
            parsed = self.parse(resp.text, obs)
            statuses.append(parsed.status)
            if parsed.value is not None:
                if parsed.memory is not None:
                    self.memory = parsed.memory
                meta = self._meta(obs, resp, msgs, statuses, failures, cost, tokens_in, tokens_out, latency, cached)
                meta.update(parsed.detail or {})
                return Action(self.to_model_units(parsed.value), meta)
            failures += 1
            self.totals["parse_failures"] += 1
        if self.on_failure == "stop":
            raise StopSession("parse_failure", f"agent {self.index}, period {obs.t}: {statuses}")
        self.totals["fallbacks"] += 1
        meta = self._meta(obs, last_resp, msgs, statuses, failures, cost, tokens_in, tokens_out, latency, cached)
        meta["fallback"] = True
        return Action(self.fallback(obs), meta)

    def _meta(self, obs, resp, msgs, statuses, failures, cost, tin, tout, latency, cached) -> dict:
        meta = dict(agent="llm", model=self.model, llm_calls=len(statuses), parse_failures=failures,
                    parse_status=statuses[-1], cached_calls=cached, cost=cost, prompt_tokens=tin,
                    completion_tokens=tout, latency=latency, response=resp.text,
                    memory=json.dumps(self.memory, ensure_ascii=False))
        if resp.reasoning:
            meta["reasoning"] = resp.reasoning
        if self.record_prompt:
            meta["prompt"] = json.dumps(msgs, ensure_ascii=False)
        return meta
