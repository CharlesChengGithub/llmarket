"""Smoke tests against real endpoints. Run by hand: pytest -m live -s

- Ollama: needs a local server (`ollama serve`) and the model (`ollama pull llama3.2:3b`).
- DeepSeek: needs DEEPSEEK_API_KEY and TATONNEMENT_LIVE_PAID=1; spends well under 0.1 CNY
  (6 calls, budget 0.1 CNY).
"""
import os

import httpx
import pytest

import tatonnement as tt

pytestmark = pytest.mark.live
MARKET = dict(n=2, a=2.0, mu=0.25, c=1.0, a0=0.0)


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="no local Ollama server")
def test_ollama_dhanda_style(tmp_path):
    cfg = dict(experiment="live_ollama", periods=5, sessions=1, seed=1, market=dict(MARKET),
               regulations=[dict(type="price_cap", value=1.70, enforcement="clip",
                                 text="It is ILLEGAL to set a price above {cap:.2f}.")],
               agents=[dict(type="llm", provider="ollama", model=os.environ.get("TATONNEMENT_OLLAMA_MODEL", "llama3.2:3b"),
                            prompt="dhanda_style", temperature=0.7), dict(copy_of=0)],
               llm=dict(cache=dict(mode="use", path=str(tmp_path / "c.sqlite")),
                        on_parse_failure=dict(retries=10, then="keep_last_price")))
    res = tt.run(cfg, out_dir=tmp_path)
    s = res["conditions"][0]["sessions"][0]
    print(s)
    assert res["status"] == "complete" and s["periods"] == 5
    again = tt.run(cfg, out_dir=tmp_path / "again")
    assert again["llm"]["live_calls"] == 0  # everything replayed from the cache


@pytest.mark.skipif(not (os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("TATONNEMENT_LIVE_PAID") == "1"),
                    reason="set DEEPSEEK_API_KEY and TATONNEMENT_LIVE_PAID=1 to spend a few fen")
def test_deepseek_fish_style(tmp_path):
    cfg = dict(experiment="live_deepseek", periods=3, sessions=1, seed=1,
               market=dict(MARKET, units=dict(alpha=3.2, beta=100)),
               agents=[dict(type="llm", provider="deepseek", model="deepseek-flash", prompt="fish_style",
                            prefix="explore", max_tokens=800,
                            # thinking is on by default and then ignores temperature; Fish et al. used
                            # non-reasoning models, so switch it off
                            extra_body={"thinking": {"type": "disabled"}}), dict(copy_of=0)],
               llm=dict(cache=dict(mode="use", path=str(tmp_path / "c.sqlite")), budget=dict(amount=0.1, currency="CNY")))
    res = tt.run(cfg, out_dir=tmp_path)
    s = res["conditions"][0]["sessions"][0]
    print(s, res["llm"]["spent"])
    assert res["status"] == "complete" and s["periods"] == 3 and res["llm"]["spent"] < 0.1
