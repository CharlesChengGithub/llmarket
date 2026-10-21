"""MockClient: answers in the agent's template with rule-based prices, without any network.

Used by tests, documentation examples and ``tatonnement estimate``. Its randomness is seeded by
the request and the call coordinates, so answers do not depend on call order or threading.

The agent passes a ``hint`` alongside each request (never part of the request itself):
  format: {"fields": [...], "price_field": "PRICE" or None (bare number)}
  market: display-unit values: cost, nash, monopoly, own_last, rivals_last (list), lower, upper,
          t, retry (True when re-asked after a rejected or unparsable answer)
  coords: the call coordinates
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from .types import ChatRequest, ChatResponse


def _midpoint(m):
    return (m["nash"] + m["monopoly"]) / 2


def _policy_follow(m, rng):
    """Match the rivals' last average price; start between Nash and monopoly."""
    if m.get("rivals_last"):
        return float(np.mean(m["rivals_last"]))
    return _midpoint(m)


POLICIES: dict[str, Callable] = {
    "follow": _policy_follow,
    "nash": lambda m, rng: m["nash"],
    "monopoly": lambda m, rng: m["monopoly"],
    "midpoint": lambda m, rng: _midpoint(m),
    "random": lambda m, rng: float(rng.uniform(m["nash"], m["monopoly"])),
    "undercut": lambda m, rng: (min(m["rivals_last"]) - 0.01 * m["nash"]) if m.get("rivals_last") else m["nash"],
}

_FILLER = ("The market data suggest that prices near the recent level keep profits stable. "
           "I will watch the competitor's response before changing course. ")


class MockClient:
    """policy: a name in POLICIES or a callable (market_hint, rng) -> display price.

    noise: sd of Gaussian noise added to the price (display units).
    completion_tokens: length of the answer in tokens (the filler text is padded to roughly
      4 characters per token so prompt and answer sizes are realistic for estimates).
    error_rate: probability of an answer without a usable price (format failure).
    fail_after: raise RuntimeError on the call after this many calls (crash injection).
    comply_on_retry: when re-asked, move the price inside the announced bounds.
    """

    def __init__(self, policy: str | Callable = "follow", noise: float = 0.0, completion_tokens: int = 120,
                 error_rate: float = 0.0, fail_after: int | None = None, comply_on_retry: bool = True,
                 seed: int = 0, chars_per_token: float = 4.0):
        self.policy = POLICIES[policy] if isinstance(policy, str) else policy
        self.policy_name = policy if isinstance(policy, str) else getattr(policy, "__name__", "custom")
        self.noise = float(noise)
        self.completion_tokens = int(completion_tokens)
        self.error_rate = float(error_rate)
        self.fail_after = fail_after
        self.comply_on_retry = comply_on_retry
        self.seed = int(seed)
        self.chars_per_token = float(chars_per_token)
        self.calls = 0

    def _rng(self, request: ChatRequest, hint: dict) -> np.random.Generator:
        coords = json.dumps(hint.get("coords", {}), sort_keys=True)
        h = hashlib.sha256((request.hash() + coords + str(self.seed)).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def complete(self, request: ChatRequest, hint: dict | None = None) -> ChatResponse:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("MockClient: injected failure")
        self.calls += 1
        hint = hint or {}
        m = hint.get("market", {})
        fmt = hint.get("format", {"fields": [], "price_field": None})
        rng = self._rng(request, hint)
        broken = rng.random() < self.error_rate
        price = float(self.policy(m, rng)) + (rng.normal(0.0, self.noise) if self.noise else 0.0)
        if self.comply_on_retry and m.get("retry"):
            lo, hi = m.get("lower"), m.get("upper")
            if lo is not None and np.isfinite(lo):
                price = max(price, lo)
            if hi is not None and np.isfinite(hi):
                price = min(price, hi)
        price_text = "I am not sure yet." if broken else f"{price:.2f}"
        text = self._render(fmt, price_text, m)
        prompt_tokens = int(round(request.prompt_chars() / self.chars_per_token))
        return ChatResponse(text=text, usage={"prompt_tokens": prompt_tokens,
                                              "completion_tokens": self.completion_tokens,
                                              "total_tokens": prompt_tokens + self.completion_tokens},
                            model=f"mock-{self.policy_name}", finish_reason="stop", latency=0.0)

    def _render(self, fmt: dict, price_text: str, m: dict) -> str:
        fields = list(fmt.get("fields") or [])
        pf = fmt.get("price_field")
        target = int(self.completion_tokens * self.chars_per_token)
        if not fields or pf is None:
            return price_text
        others = [f for f in fields if f != pf]
        body = max(target - 40, 0) // max(len(others), 1)
        lines = []
        for f in fields:
            if f == pf:
                lines.append(f"{f}: {price_text}")
            else:
                filler = (_FILLER * (body // len(_FILLER) + 1))[:body].strip()
                lines.append(f"{f}: Round {m.get('t', 0) + 1}. {filler}")
        return "\n".join(lines)
