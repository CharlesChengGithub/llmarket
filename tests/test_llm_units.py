"""Unit tests for the LLM layer: parsing, cache, ledger, providers and the HTTP client."""
import asyncio
import json

import httpx
import pytest

from llmarket.llm.cache import Cache
from llmarket.llm.client import OpenAICompatibleClient
from llmarket.llm.ledger import Ledger, PriceRow, PriceTable
from llmarket.llm.mock import MockClient
from llmarket.llm.parsing import parse_price, split_sections
from llmarket.llm.providers import PROVIDERS, Provider, resolve
from llmarket.llm.service import LLMService
from llmarket.llm.types import (
    BudgetExceeded,
    CacheMiss,
    CallCoords,
    ChatRequest,
    ChatResponse,
    LLMError,
    UnknownPrice,
)

FIELDS = ["OBSERVATIONS", "PLANS", "INSIGHTS", "PRICE"]


# -- parsing --------------------------------------------------------------------------------
def test_parse_template_answer():
    text = "OBSERVATIONS: prices rose.\nPLANS: hold at 1.8\nINSIGHTS: rival follows\nPRICE: 1.80"
    r = parse_price(text, FIELDS, "PRICE", upper=20)
    assert r.status == "ok" and r.value == 1.8
    assert r.sections["PLANS"] == "hold at 1.8"


def test_parse_markdown_and_echoed_template():
    text = ("Answer with OBSERVATIONS, PLANS, INSIGHTS, PRICE.\n**OBSERVATIONS**: x\n## PLANS: y\n"
            "**INSIGHTS:** z\n**PRICE**: $5.43 (up from 5.20)")
    r = parse_price(text, FIELDS, "PRICE", upper=20)
    assert r.value == 5.43
    assert r.sections["INSIGHTS"] == "z"


def test_parse_think_block_removed():
    text = "<think>PRICE: 9.99 maybe</think>OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c\nPRICE: 1.7"
    assert parse_price(text, FIELDS, "PRICE", upper=20).value == 1.7


@pytest.mark.parametrize("price, status", [("-1.5", "negative"), ("25", "out_of_range"), ("soon", "no_number")])
def test_parse_failures(price, status):
    text = f"OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c\nPRICE: {price}"
    r = parse_price(text, FIELDS, "PRICE", upper=20)
    assert r.value is None and r.status == status


def test_parse_missing_field_and_bare_number():
    assert parse_price("OBSERVATIONS: a", FIELDS, "PRICE", upper=20).status == "missing_field"
    assert parse_price("I will charge 1.65 this round.", [], None, upper=20).value == 1.65
    assert split_sections("no headings here", FIELDS) == {}


@pytest.mark.parametrize("text, value, status", [
    ("1.65", 1.65, "ok"), (" **$1.65.**\n", 1.65, "ok"), ("Price: 1.65", 1.65, "ok"),
    ("My price is 1.65.", None, "extra_text"), ("Cost is 1.00, so 1.65", None, "extra_text"),
    ("none", None, "no_number"), ("-1", None, "negative")])
def test_parse_only_mode(text, value, status):
    r = parse_price(text, [], None, upper=20, pick="only")
    assert (r.value, r.status) == (value, status)


def test_inline_headings_need_exact_case():
    prose = "The gap between my price and the rival's price: Round 6: 1.85 - 1.75. I'll set my"
    assert parse_price(prose, ["PRICE"], "PRICE", upper=20).status == "missing_field"
    assert parse_price("thinking... PRICE: 1.62", ["PRICE"], "PRICE", upper=20).value == 1.62
    assert parse_price("Price: 1.35", ["PRICE"], "PRICE", upper=20).value == 1.35


def test_parse_first_and_last():
    text = "Cost is 1.00 and the cap 1.70. My price: 1.55"
    assert parse_price(text, [], None, upper=20).value == 1.0
    assert parse_price(text, [], None, upper=20, pick="last").value == 1.55
    with pytest.raises(ValueError):
        parse_price(text, [], None, upper=20, pick="middle")


# -- requests and cache -----------------------------------------------------------------------
def _req(content="hi", **params):
    return ChatRequest("mock", "m", ({"role": "user", "content": content},), params)


def test_request_hash_and_coordinates():
    a, b = _req(temperature=1.0), _req(temperature=1.0)
    assert a.hash() == b.hash()
    assert a.hash() != _req("hi!", temperature=1.0).hash()
    assert a.hash() != _req(temperature=0.7).hash()
    c0, c1 = CallCoords("e", "main", 1, 0, 0, 5, 0), CallCoords("e", "main", 1, 0, 0, 5, 1)
    assert c0.key(a.hash()) != c1.key(a.hash())
    assert c0.call_seed() == CallCoords("e", "main", 1, 0, 0, 5, 0).call_seed() != c1.call_seed()
    assert a.payload() == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0}
    assert _req(extra_body={"enable_thinking": False}).payload()["enable_thinking"] is False


def test_cache_modes(tmp_path):
    path = tmp_path / "c.sqlite"
    req, co = _req(), CallCoords("e", "main", 0, 0, 0, 0, 0)
    resp = ChatResponse("PRICE: 1.5", usage={"prompt_tokens": 3}, model="m", latency=0.2)
    with pytest.raises(FileNotFoundError):
        Cache(path, "replay")
    c = Cache(path, "use")
    assert c.get(req, co) is None
    c.put(req, co, resp, None)
    hit = c.get(req, co)
    assert hit.cached and hit.text == "PRICE: 1.5" and hit.prompt_tokens == 3 and hit.cost == 0
    assert c.count("e") == 1
    assert c.export(tmp_path / "x.sqlite", "e") == 1
    c.close()
    r = Cache(tmp_path / "x.sqlite", "replay")
    assert r.get(req, co).text == "PRICE: 1.5"
    assert r.get(req, CallCoords("e", "main", 0, 0, 0, 1, 0)) is None
    assert Cache(path, "refresh").get(req, co) is None
    assert Cache(None, "off").get(req, co) is None


def test_service_replay_miss_raises(tmp_path):
    Cache(tmp_path / "c.sqlite", "use").close()
    svc = LLMService(Cache(tmp_path / "c.sqlite", "replay"))
    with pytest.raises(CacheMiss):
        svc.call(PROVIDERS["mock"], _req(), CallCoords())


def test_service_use_then_replay(tmp_path):
    path = tmp_path / "c.sqlite"
    hint = {"format": {"fields": ["PRICE"], "price_field": "PRICE"}, "market": {"nash": 1.5, "monopoly": 1.9}}
    svc = LLMService(Cache(path, "use"), mock={"policy": "midpoint"})
    a = svc.call(PROVIDERS["mock"], _req(), CallCoords(), hint)
    assert svc.live_calls == 1 and not a.cached
    b = svc.call(PROVIDERS["mock"], _req(), CallCoords(), hint)
    assert b.cached and b.text == a.text and svc.live_calls == 1
    svc.close()


# -- ledger -----------------------------------------------------------------------------------
PAID = Provider("paid", "mock://", None, paid=True)


def _paid_table():
    return PriceTable([dict(provider="paid", model="m", currency="CNY", input_cache_hit=0.5,
                            input_cache_miss=2.0, output=8.0)])


def test_price_arithmetic():
    row = PriceRow("p", "m", "CNY", 0.5, 2.0, 8.0)
    assert row.cost(1_000_000, 0, 0) == pytest.approx(2.0)
    assert row.cost(1_000_000, 400_000, 500_000) == pytest.approx(0.2 + 1.2 + 4.0)
    resp = ChatResponse("x", usage={"prompt_tokens": 1000, "prompt_cache_hit_tokens": 200, "completion_tokens": 50})
    assert resp.cached_tokens == 200
    resp2 = ChatResponse("x", usage={"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}})
    assert resp2.cached_tokens == 4


def test_default_price_rows():
    t = PriceTable()
    assert t.lookup("ollama", "llama3.2:3b").free
    ds = t.lookup("deepseek", "deepseek-flash")
    assert ds.currency == "CNY" and ds.source.startswith("https://")
    assert t.lookup("deepseek", "deepseek-v4-flash") is ds
    assert t.lookup("qwen", "qwen-plus") is None


def test_ledger_refuses_unknown_or_unbudgeted_paid_calls():
    with pytest.raises(UnknownPrice):
        Ledger(PriceTable(), budget=1.0).check(_req(), Provider("qwen", "x", None, paid=True))
    with pytest.raises(UnknownPrice):  # paid model but no budget
        Ledger(_paid_table()).check(ChatRequest("paid", "m", ({"role": "user", "content": "x"},)), PAID)
    with pytest.raises(UnknownPrice):  # currency mismatch
        Ledger(_paid_table(), budget=1.0, currency="USD").check(
            ChatRequest("paid", "m", ({"role": "user", "content": "x"},)), PAID)
    assert Ledger().check(_req(), PROVIDERS["ollama"]) >= 0  # free provider, no budget needed


def test_ledger_budget_and_hard_limit():
    led = Ledger(_paid_table(), budget=0.01, currency="CNY", hard_factor=1.5, output_tokens_guess=100)
    req = ChatRequest("paid", "m", ({"role": "user", "content": "x" * 4000},))  # ~1000 tokens -> 0.0028 est
    r = led.check(req, PAID)
    assert not led.exhausted and led.reserved == pytest.approx(0.0028)
    led.charge(PAID, req, ChatResponse("x", usage={"prompt_tokens": 1000, "completion_tokens": 100}), r)
    assert led.spent == pytest.approx(0.0028) and led.reserved == 0
    assert led.would_exceed(0.008) and not led.would_exceed(0.001)
    led.spent = 0.0095
    led.check(req, PAID)  # passes the budget but not the hard limit: mark exhausted, allow the call
    assert led.exhausted
    led.spent = 0.0149
    with pytest.raises(BudgetExceeded):
        led.check(req, PAID)


def test_ledger_reserves_calls_in_flight():
    led = Ledger(_paid_table(), budget=0.01, currency="CNY", hard_factor=1.0, output_tokens_guess=100)
    req = ChatRequest("paid", "m", ({"role": "user", "content": "x" * 4000},))  # estimate 0.0028
    held = [led.check(req, PAID) for _ in range(3)]  # three parallel calls, none finished yet
    assert led.reserved == pytest.approx(0.0084)
    with pytest.raises(BudgetExceeded):  # a fourth would pass the limit even though nothing is spent
        led.check(req, PAID)
    led.release(held[0])
    assert led.reserved == pytest.approx(0.0056)


# -- providers and HTTP client ------------------------------------------------------------------
def test_provider_keys_from_environment(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        PROVIDERS["deepseek"].headers()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert PROVIDERS["deepseek"].headers()["Authorization"] == "Bearer sk-test"
    assert "sk-test" not in json.dumps(PROVIDERS["deepseek"].describe())
    assert PROVIDERS["ollama"].headers() == {"Content-Type": "application/json"}
    with pytest.raises(ValueError):
        resolve({"name": "x", "base_url": "http://h/v1", "api_key": "sk"})
    remote = resolve({"name": "ollama", "base_url": "http://gpu-box:11434/v1"})
    assert remote.paid is False and remote.endpoint == "http://gpu-box:11434/v1/chat/completions"


def _ok(text="PRICE: 1.7"):
    return {"id": "x", "model": "served-model", "system_fingerprint": "fp1",
            "choices": [{"message": {"role": "assistant", "content": text, "reasoning_content": "because"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4}}


def test_client_retries_rate_limits_then_succeeds():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json=_ok())

    c = OpenAICompatibleClient(PROVIDERS["ollama"], transport=httpx.MockTransport(handler), backoff=0.0)
    r = c.complete(_req(temperature=0.7))
    assert r.text == "PRICE: 1.7" and r.reasoning == "because" and r.model == "served-model"
    assert r.system_fingerprint == "fp1" and r.prompt_tokens == 12
    assert c.http_retries == 2 and calls[0]["temperature"] == 0.7


def test_client_raises_on_client_errors():
    c = OpenAICompatibleClient(PROVIDERS["ollama"], backoff=0.0,
                               transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad model")))
    with pytest.raises(LLMError, match="400"):
        c.complete(_req())


def test_async_client():
    async def handler(request):
        return httpx.Response(200, json=_ok("PRICE: 2.0"))

    c = OpenAICompatibleClient(PROVIDERS["ollama"], async_transport=httpx.MockTransport(handler))
    assert asyncio.run(c.acomplete(_req())).text == "PRICE: 2.0"


def test_mock_client_is_order_independent():
    hint = {"format": {"fields": FIELDS, "price_field": "PRICE"},
            "market": {"nash": 1.47, "monopoly": 1.92}, "coords": {"period": 3}}
    m1, m2 = MockClient("random", seed=1), MockClient("random", seed=1)
    a = m1.complete(_req("x"), hint).text
    m2.complete(_req("y"), hint)
    assert m2.complete(_req("x"), hint).text == a
    broken = MockClient("midpoint", error_rate=1.0).complete(_req(), hint).text
    assert parse_price(broken, FIELDS, "PRICE", 20).status == "no_number"
