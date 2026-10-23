"""OpenAI-compatible providers. API keys are read from environment variables at call time and
are never stored in configurations, logs, caches or manifests."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .types import LLMError


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None = None
    supports_seed: bool = False
    paid: bool = True

    def api_key(self) -> str | None:
        if self.key_env is None:
            return None
        key = os.environ.get(self.key_env)
        if not key:
            raise LLMError(f"provider {self.name!r} needs the environment variable {self.key_env}")
        return key

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        key = self.api_key()
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def describe(self) -> dict:
        """What goes into manifests: no key, only where the key comes from."""
        return {"name": self.name, "base_url": self.base_url, "key_env": self.key_env}


PROVIDERS = {
    "deepseek": Provider("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": Provider("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
                     supports_seed=True),
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, supports_seed=True, paid=False),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", supports_seed=True),
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "siliconflow": Provider("siliconflow", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "mock": Provider("mock", "mock://", None, supports_seed=False, paid=False),
}


def resolve(spec: str | dict) -> Provider:
    """A provider by name, or a generic endpoint {name, base_url, key_env, paid, supports_seed}.

    A known name with a base_url override (e.g. Ollama on another host) keeps its other settings.
    """
    if isinstance(spec, str):
        if spec not in PROVIDERS:
            raise ValueError(f"unknown provider {spec!r}; known: {sorted(PROVIDERS)} or a mapping with base_url")
        return PROVIDERS[spec]
    spec = dict(spec)
    name = spec.get("name", "generic")
    base = PROVIDERS.get(name)
    if base is None and "base_url" not in spec:
        raise ValueError("a generic provider needs base_url")
    if "api_key" in spec:
        raise ValueError("put API keys in an environment variable and name it with key_env")
    return Provider(
        name=name,
        base_url=spec.get("base_url", base.base_url if base else ""),
        key_env=spec.get("key_env", base.key_env if base else None),
        supports_seed=bool(spec.get("supports_seed", base.supports_seed if base else False)),
        paid=bool(spec.get("paid", base.paid if base else True)),
    )
