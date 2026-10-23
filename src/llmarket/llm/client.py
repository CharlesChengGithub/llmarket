"""HTTP client for OpenAI-compatible /chat/completions endpoints (sync and async).

Network errors, HTTP 429 and 5xx responses are retried with exponential backoff; these retries
are separate from an agent's format retries. Other HTTP errors raise :class:`LLMError`.
"""
from __future__ import annotations

import asyncio
import random
import time

import httpx

from .providers import Provider
from .types import ChatRequest, ChatResponse, LLMError

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def parse_completion(data: dict, latency: float) -> ChatResponse:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected response shape: {str(data)[:300]}") from exc
    return ChatResponse(
        text=msg.get("content") or "",
        reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
        usage=data.get("usage") or {},
        model=data.get("model"),
        system_fingerprint=data.get("system_fingerprint"),
        finish_reason=choice.get("finish_reason"),
        latency=latency,
    )


class OpenAICompatibleClient:
    def __init__(self, provider: Provider, timeout: float = 120.0, max_retries: int = 6,
                 backoff: float = 1.0, max_backoff: float = 60.0, transport: httpx.BaseTransport | None = None,
                 async_transport: httpx.AsyncBaseTransport | None = None):
        self.provider = provider
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self._transport = transport
        self._async_transport = async_transport
        self._client: httpx.Client | None = None
        self.http_retries = 0

    # -- helpers ----------------------------------------------------------------------------
    def _delay(self, attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    return min(float(ra), self.max_backoff)
                except ValueError:
                    pass
        return min(self.backoff * 2**attempt, self.max_backoff) * (0.5 + random.random() / 2)

    def _error(self, resp: httpx.Response) -> LLMError:
        return LLMError(f"{self.provider.name} HTTP {resp.status_code}: {resp.text[:500]}")

    def _sync(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, transport=self._transport)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- calls ------------------------------------------------------------------------------
    def complete(self, request: ChatRequest, hint: dict | None = None) -> ChatResponse:
        client = self._sync()
        body = request.payload()
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            resp = None
            try:
                resp = client.post(self.provider.endpoint, json=body, headers=self.provider.headers())
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.provider.name}: {type(exc).__name__}: {exc}") from exc
            else:
                if resp.status_code == 200:
                    return parse_completion(resp.json(), time.perf_counter() - started)
                if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise self._error(resp)
            self.http_retries += 1
            time.sleep(self._delay(attempt, resp))
        raise AssertionError("unreachable")

    async def acomplete(self, request: ChatRequest, hint: dict | None = None) -> ChatResponse:
        body = request.payload()
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._async_transport) as client:
            for attempt in range(self.max_retries + 1):
                started = time.perf_counter()
                resp = None
                try:
                    resp = await client.post(self.provider.endpoint, json=body, headers=self.provider.headers())
                except httpx.TransportError as exc:
                    if attempt == self.max_retries:
                        raise LLMError(f"{self.provider.name}: {type(exc).__name__}: {exc}") from exc
                else:
                    if resp.status_code == 200:
                        return parse_completion(resp.json(), time.perf_counter() - started)
                    if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                        raise self._error(resp)
                self.http_retries += 1
                await asyncio.sleep(self._delay(attempt, resp))
        raise AssertionError("unreachable")
