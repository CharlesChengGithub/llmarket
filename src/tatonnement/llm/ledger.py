"""Price table, cost ledger and budget guard.

Every live call is priced from its token usage. Before each call the ledger estimates its cost
from the prompt length; once spent + estimate would pass the budget the ledger is marked
exhausted, the current period is completed and the session stops ('budget_stopped'). A call that
would pass the hard limit (default 1.2 x budget) raises :class:`BudgetExceeded` immediately.
Paid providers need both a budget and a price row; otherwise calls are refused.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from importlib import resources

import yaml

from .providers import Provider
from .types import BudgetExceeded, ChatRequest, ChatResponse, UnknownPrice

DEFAULT_CHARS_PER_TOKEN = 4.0


@dataclass
class PriceRow:
    provider: str
    model: str
    currency: str | None
    input_cache_hit: float
    input_cache_miss: float
    output: float
    per_tokens: float = 1e6
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    aliases: list = field(default_factory=list)
    note: str | None = None
    date: str | None = None
    source: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "PriceRow":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        known.setdefault("currency", None)
        for k in ("input_cache_hit", "input_cache_miss", "output"):
            if known.get(k) is None:
                raise ValueError(f"price row {d.get('provider')}/{d.get('model')} needs {k}")
            known[k] = float(known[k])
        if "date" in known and known["date"] is not None:
            known["date"] = str(known["date"])
        return cls(**known)

    @property
    def free(self) -> bool:
        return self.input_cache_hit == self.input_cache_miss == self.output == 0

    def cost(self, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
        miss = max(prompt_tokens - cached_tokens, 0)
        return (cached_tokens * self.input_cache_hit + miss * self.input_cache_miss
                + completion_tokens * self.output) / self.per_tokens

    def matches(self, provider: str, model: str) -> bool:
        return self.provider == provider and (self.model in ("*", model) or model in self.aliases)


def default_rows() -> list[PriceRow]:
    text = resources.files("tatonnement.llm").joinpath("prices.yaml").read_text()
    return [PriceRow.from_dict(d) for d in yaml.safe_load(text)]


class PriceTable:
    """Default rows plus overrides; an override for the same provider/model wins."""

    def __init__(self, overrides: list[dict] | None = None, defaults: bool = True):
        self.rows = [PriceRow.from_dict(d) for d in (overrides or [])]
        if defaults:
            self.rows += default_rows()

    def lookup(self, provider: str, model: str) -> PriceRow | None:
        exact = [r for r in self.rows if r.matches(provider, model) and r.model != "*"]
        if exact:
            return exact[0]
        wild = [r for r in self.rows if r.matches(provider, model)]
        return wild[0] if wild else None


class Ledger:
    def __init__(self, table: PriceTable | None = None, budget: float | None = None,
                 currency: str | None = None, hard_factor: float = 1.2, output_tokens_guess: int = 500):
        self.table = table or PriceTable()
        self.budget = None if budget is None else float(budget)
        self.currency = currency
        self.hard_limit = None if budget is None else float(budget) * float(hard_factor)
        self.output_tokens_guess = int(output_tokens_guess)
        self.spent = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.exhausted = False
        self.used: dict[tuple, PriceRow] = {}
        self._lock = threading.Lock()

    def row(self, provider: Provider, model: str) -> PriceRow:
        r = self.table.lookup(provider.name, model)
        if r is None:
            if not provider.paid:
                r = PriceRow(provider.name, "*", None, 0.0, 0.0, 0.0, source="unpaid provider")
            else:
                raise UnknownPrice(f"no price row for {provider.name}/{model}; add one under llm.prices "
                                   "(see tatonnement/llm/prices.yaml for the format)")
        if not r.free:
            if self.budget is None:
                raise UnknownPrice(f"{provider.name}/{model} is a paid model: set llm.budget.amount")
            if self.currency is not None and r.currency is not None and r.currency != self.currency:
                raise UnknownPrice(f"budget currency {self.currency} differs from the price row currency "
                                   f"{r.currency} for {provider.name}/{model}")
        self.used[(r.provider, r.model)] = r
        return r

    def estimate(self, request: ChatRequest, row: PriceRow, output_tokens: int | None = None) -> float:
        prompt = request.prompt_chars() / row.chars_per_token
        out = output_tokens or request.params.get("max_tokens") or self.output_tokens_guess
        return row.cost(int(prompt), 0, int(out))

    def check(self, request: ChatRequest, provider: Provider) -> float:
        """Called before a live call; returns the estimate. Raises past the hard limit."""
        row = self.row(provider, request.model)
        est = self.estimate(request, row)
        if row.free or self.budget is None:
            return est
        with self._lock:
            if self.spent + est > self.hard_limit:
                self.exhausted = True
                raise BudgetExceeded(f"spent {self.spent:.4f} + estimate {est:.4f} passes the hard limit "
                                     f"{self.hard_limit:.4f} {self.currency or row.currency}")
            if self.spent + est > self.budget:
                self.exhausted = True
        return est

    def charge(self, provider: Provider, request: ChatRequest, response: ChatResponse) -> float:
        row = self.row(provider, request.model)
        cost = row.cost(response.prompt_tokens, response.cached_tokens, response.completion_tokens)
        with self._lock:
            self.spent += cost
            self.calls += 1
            self.prompt_tokens += response.prompt_tokens
            self.cached_tokens += response.cached_tokens
            self.completion_tokens += response.completion_tokens
            if self.budget is not None and self.spent >= self.budget:
                self.exhausted = True
        return cost

    def would_exceed(self, projected: float) -> bool:
        if self.budget is None:
            return False
        return self.exhausted or self.spent + projected > self.budget

    def summary(self) -> dict:
        return dict(spent=self.spent, currency=self.currency, budget=self.budget, live_calls=self.calls,
                    prompt_tokens=self.prompt_tokens, cached_prompt_tokens=self.cached_tokens,
                    completion_tokens=self.completion_tokens, exhausted=self.exhausted,
                    price_rows=[asdict(r) for r in self.used.values()])
