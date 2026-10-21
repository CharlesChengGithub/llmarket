"""Requests, responses and call coordinates shared by clients, the cache and the ledger."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """A provider returned an error that retrying will not fix."""


class CacheMiss(LLMError):
    """Replay mode needs a cached response that does not exist."""


class BudgetExceeded(LLMError):
    """A paid call would break the hard spending limit."""


class UnknownPrice(LLMError):
    """A paid provider/model has no price row, so its cost cannot be controlled."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ChatRequest:
    """One chat-completion request. params holds temperature, top_p, max_tokens, seed and
    extra_body (provider-specific fields passed through unchanged)."""

    provider: str
    model: str
    messages: tuple[dict, ...]
    params: dict = field(default_factory=dict)

    def payload(self) -> dict:
        body = {"model": self.model, "messages": list(self.messages)}
        extra = {}
        for k, v in self.params.items():
            if v is None:
                continue
            if k == "extra_body":
                extra = dict(v)
            else:
                body[k] = v
        body.update(extra)
        return body

    def hash(self) -> str:
        blob = _canonical({"provider": self.provider, "model": self.model,
                           "messages": list(self.messages), "params": self.params})
        return hashlib.sha256(blob.encode()).hexdigest()

    def prompt_chars(self) -> int:
        return sum(len(str(m.get("content", ""))) for m in self.messages)


@dataclass(frozen=True)
class CallCoords:
    """Where a call sits in an experiment. Identical prompts at different coordinates are
    independent samples; the same coordinates and prompt return the cached sample."""

    namespace: str = ""
    condition: str = ""
    seed: int = 0
    session: int = 0
    agent: int = 0
    period: int = 0
    attempt: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def key(self, request_hash: str) -> str:
        return hashlib.sha256((request_hash + _canonical(self.as_dict())).encode()).hexdigest()

    def call_seed(self) -> int:
        """Deterministic per-call seed for providers that accept one."""
        return int(hashlib.sha256(_canonical(self.as_dict()).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


@dataclass
class ChatResponse:
    text: str
    reasoning: str | None = None
    usage: dict = field(default_factory=dict)
    model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    latency: float = 0.0
    cached: bool = False
    cost: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens") or 0)

    @property
    def cached_tokens(self) -> int:
        """Prompt tokens served from the provider's own prefix cache (billed at a lower rate)."""
        u = self.usage
        if u.get("prompt_cache_hit_tokens") is not None:  # DeepSeek
            return int(u["prompt_cache_hit_tokens"])
        details = u.get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or 0)

    def to_json(self) -> str:
        return _canonical({"text": self.text, "reasoning": self.reasoning, "usage": self.usage,
                           "model": self.model, "system_fingerprint": self.system_fingerprint,
                           "finish_reason": self.finish_reason})

    @classmethod
    def from_json(cls, blob: str, **kw) -> "ChatResponse":
        d = json.loads(blob)
        return cls(text=d["text"], reasoning=d.get("reasoning"), usage=d.get("usage") or {},
                   model=d.get("model"), system_fingerprint=d.get("system_fingerprint"),
                   finish_reason=d.get("finish_reason"), **kw)
