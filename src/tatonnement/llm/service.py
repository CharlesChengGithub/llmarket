"""LLMService: one entry point that combines the cache, the ledger and the provider clients."""
from __future__ import annotations

import threading
from typing import Any

from .cache import Cache
from .client import OpenAICompatibleClient
from .ledger import Ledger, PriceTable
from .mock import MockClient
from .providers import Provider, resolve
from .types import CacheMiss, CallCoords, ChatRequest, ChatResponse


class LLMService:
    """Thread-safe: agents of one period and sessions of one run may call it concurrently.

    clients maps provider names to client objects; missing ones are created on first use
    (MockClient for 'mock', OpenAICompatibleClient otherwise).
    """

    def __init__(self, cache: Cache | None = None, ledger: Ledger | None = None,
                 clients: dict[str, Any] | None = None, timeout: float = 120.0, mock: dict | None = None,
                 client_override: Any = None):
        self.cache = cache or Cache(mode="off")
        self.ledger = ledger or Ledger()
        self.clients = dict(clients or {})
        self.timeout = timeout
        self.mock_options = dict(mock or {})
        self.client_override = client_override  # one client for every provider (cost estimates)
        self.providers: dict[str, Provider] = {}
        self.live_calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, llm: dict | None, clients: dict | None = None) -> "LLMService":
        llm = llm or {}
        cache_cfg = llm.get("cache") or {}
        budget = llm.get("budget") or {}
        cache = Cache(cache_cfg.get("path"), cache_cfg.get("mode", "use"))
        ledger = Ledger(PriceTable(llm.get("prices")), budget=budget.get("amount"),
                        currency=budget.get("currency"), hard_factor=float(budget.get("hard_factor", 1.2)),
                        output_tokens_guess=int(budget.get("output_tokens_guess", 500)))
        return cls(cache, ledger, clients, timeout=float(llm.get("timeout", 120.0)), mock=llm.get("mock"))

    def provider(self, spec: str | dict) -> Provider:
        p = resolve(spec)
        self.providers[p.name] = p
        return p

    def client(self, provider: Provider):
        if self.client_override is not None:
            return self.client_override
        with self._lock:
            if provider.name not in self.clients:
                if provider.name == "mock":
                    self.clients[provider.name] = MockClient(**self.mock_options)
                else:
                    self.clients[provider.name] = OpenAICompatibleClient(provider, timeout=self.timeout)
            return self.clients[provider.name]

    def call(self, provider: Provider, request: ChatRequest, coords: CallCoords,
             hint: dict | None = None) -> ChatResponse:
        cached = self.cache.get(request, coords)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return cached
        if self.cache.mode == "replay":
            raise CacheMiss(f"no cached answer for {coords}")
        self.ledger.check(request, provider)
        hint = dict(hint or {}, coords=coords.as_dict())
        response = self.client(provider).complete(request, hint)
        response.cost = self.ledger.charge(provider, request, response)
        with self._lock:
            self.live_calls += 1
        row = self.ledger.row(provider, request.model)
        self.cache.put(request, coords, response, row.currency)
        return response

    def summary(self) -> dict:
        out = self.ledger.summary()
        out.update(cache_mode=self.cache.mode, cache_path=None if self.cache.mode == "off" else str(self.cache.path),
                   cache_hits=self.cache_hits, providers=[p.describe() for p in self.providers.values()])
        return out

    def close(self) -> None:
        for c in self.clients.values():
            close = getattr(c, "close", None)
            if callable(close):
                close()
        self.cache.close()
