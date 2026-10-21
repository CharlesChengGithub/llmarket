"""End-to-end LLM experiments with the offline MockClient."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tatonnement.core.experiment import run
from tatonnement.llm.mock import MockClient
from tatonnement.llm.service import LLMService
from tatonnement.llm.types import UnknownPrice

MARKET = dict(n=2, a=2.0, mu=0.25, c=1.0, a0=0.0)


def fish_cfg(periods=300, sessions=1, mock=None, agent=None, **llm):
    a = dict(dict(type="llm", provider="mock", model="mock", prefix="explore", record_prompt=True), **(agent or {}))
    return dict(experiment="fish_mock", periods=periods, sessions=sessions, seed=11,
                market=dict(MARKET, units=dict(alpha=3.2, beta=100)), agents=[a, dict(copy_of=0)],
                llm=dict(dict(cache=dict(mode="off"), mock=mock or dict(policy="follow", noise=0.05)), **llm))


def frames(res, condition=0):
    d = Path(res["directory"]) / res["conditions"][condition]["condition"]
    files = sorted(d.glob("periods_s*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def texts(res, session=0, condition=0):
    d = Path(res["directory"]) / res["conditions"][condition]["condition"]
    return [json.loads(line) for line in open(d / f"periods_s{session:04d}_texts.jsonl")]


def strip_wall(rows):
    return [{k: v for k, v in r.items() if k != "wall_seconds"} for r in rows]


def test_fish_structure_300_periods(tmp_path):
    res = run(fish_cfg(), out_dir=tmp_path)
    s = res["conditions"][0]["sessions"][0]
    assert res["status"] == "complete" and s["periods"] == 300 and res["llm"]["live_calls"] == 600
    df = frames(res)
    assert len(df) == 600 and (df.meta_parse_status == "ok").all()
    t = texts(res)
    prompt = json.loads(t[2 * 150]["prompt"])[0]["content"]  # period 150, agent 0
    assert "Round 51:" in prompt and "Round 50:" not in prompt and "Round 150:" in prompt
    assert "No customer would pay more than" in prompt and "Producing one unit costs 3.20" in prompt
    mem = json.loads(t[2 * 150]["memory"])
    assert set(mem) == {"PLANS", "INSIGHTS"} and mem["PLANS"].startswith("Round 151")
    manifest = json.loads((Path(res["directory"]) / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["llm"]["live_calls"] == 600


def test_runs_are_deterministic(tmp_path):
    a = run(fish_cfg(periods=60), out_dir=tmp_path / "a")
    b = run(fish_cfg(periods=60), out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(frames(a), frames(b))
    assert texts(a) == texts(b)


def test_resume_after_crash_costs_no_repeated_calls(tmp_path):
    cache = dict(mode="use", path=str(tmp_path / "cache.sqlite"))
    mock = dict(policy="follow", noise=0.05)
    cfg = fish_cfg(periods=200, mock=mock, cache=cache)
    crashing = LLMService.from_config(dict(cfg["llm"], concurrency=1),
                                      clients={"mock": MockClient(**mock, fail_after=250)})
    cfg_seq = copy.deepcopy(cfg)
    cfg_seq["llm"]["concurrency"] = 1
    with pytest.raises(RuntimeError, match="injected"):
        run(cfg_seq, out_dir=tmp_path / "crash", runtime={"llm": crashing})
    manifest = json.loads(next((tmp_path / "crash").rglob("manifest.json")).read_text())
    assert manifest["status"].startswith("failed")
    crashing.close()
    resumed = run(cfg, out_dir=tmp_path / "resume")
    assert resumed["llm"]["live_calls"] == 400 - 250 and resumed["llm"]["cache_hits"] == 250
    clean = run(fish_cfg(periods=200, mock=mock), out_dir=tmp_path / "clean")
    cols = ["session", "t", "agent", "proposed", "posted", "profit"]
    pd.testing.assert_frame_equal(frames(resumed)[cols], frames(clean)[cols])
    replay = run(dict(cfg, llm=dict(cfg["llm"], cache=dict(cache, mode="replay"))), out_dir=tmp_path / "replay")
    assert replay["llm"]["live_calls"] == 0
    assert replay["llm"]["cache_totals_for_experiment"]["None"]["calls"] == 400
    pd.testing.assert_frame_equal(frames(replay)[cols], frames(clean)[cols])


def _paid_cfg(budget, periods=300):
    cfg = fish_cfg(periods=periods, agent=dict(provider={"name": "paidmock", "base_url": "mock://", "paid": True}),
                   prices=[dict(provider="paidmock", model="mock", currency="CNY", input_cache_hit=0,
                                input_cache_miss=1000, output=1000)],
                   budget=dict(amount=budget, currency="CNY"))
    return cfg


def test_budget_stop(tmp_path):
    svc = LLMService.from_config(_paid_cfg(5.0)["llm"], clients={"paidmock": MockClient("follow")})
    res = run(_paid_cfg(5.0), out_dir=tmp_path, runtime={"llm": svc})
    s = res["conditions"][0]["sessions"][0]
    assert res["status"] == "budget_stopped" and s["stop_reason"] == "budget_stopped"
    assert 0 < s["periods"] < 300
    assert 0 < svc.ledger.spent <= 5.0
    manifest = json.loads((Path(res["directory"]) / "manifest.json").read_text())
    assert manifest["status"] == "budget_stopped" and manifest["llm"]["price_rows"][0]["provider"] == "paidmock"


def test_paid_provider_without_price_is_refused(tmp_path):
    cfg = _paid_cfg(5.0)
    cfg["llm"].pop("prices")
    svc = LLMService.from_config(cfg["llm"], clients={"paidmock": MockClient("follow")})
    with pytest.raises(UnknownPrice):
        run(cfg, out_dir=tmp_path, runtime={"llm": svc})
    assert svc.clients["paidmock"].calls == 0


def test_format_retries_are_counted(tmp_path):
    res = run(fish_cfg(periods=50, mock=dict(policy="midpoint", error_rate=0.3)), out_dir=tmp_path)
    s = res["conditions"][0]["sessions"][0]
    df = frames(res)
    failed = [json.loads(t["failed_responses"]) for t in texts(res) if "failed_responses" in t]
    assert failed and all(f[0]["status"] == "no_number" for f in failed)
    for i in (0, 1):
        fails = s[f"agent{i}_parse_failures"]
        assert fails > 0 and s[f"agent{i}_calls"] == 50 + fails
        assert df[df.agent == i].meta_parse_failures.sum() == fails
    assert s["periods"] == 50


def test_parse_failure_stops_or_keeps_last_price(tmp_path):
    stop = run(fish_cfg(periods=5, mock=dict(policy="midpoint", error_rate=1.0)), out_dir=tmp_path / "s")
    s = stop["conditions"][0]["sessions"][0]
    assert s["stop_reason"] == "parse_failure" and s["periods"] == 0 and s["agent0_calls"] == 11
    events = (Path(stop["directory"]) / "main" / "periods_s0000_events.jsonl").read_text()
    assert "I am not sure yet" in events  # the last unusable answer is kept for diagnosis
    keep = run(fish_cfg(periods=5, mock=dict(policy="midpoint", error_rate=1.0), agent=dict(retries=2),
                        on_parse_failure=dict(then="keep_last_price")), out_dir=tmp_path / "k")
    k = keep["conditions"][0]["sessions"][0]
    assert k["periods"] == 5 and k["agent0_fallbacks"] == 5 and k["agent0_calls"] == 15
    df = frames(keep)
    pn = 1.4729273733327568
    assert np.allclose(df.posted, pn, atol=1e-6)


def test_rejected_price_is_reproposed(tmp_path):
    cfg = fish_cfg(periods=4, mock=dict(policy="nash"))
    cfg["market"]["units"] = dict(alpha=1.0, beta=1.0)
    cfg["regulations"] = [dict(type="price_floor", kind="constant", value=1.6, enforcement="reject")]
    res = run(cfg, out_dir=tmp_path)
    df = frames(res)
    assert (df.rejections == 1).all() and np.allclose(df.posted, 1.6) and not df.violation_price_floor.any()
    prompt = json.loads(texts(res)[0]["prompt"])[0]["content"]
    assert "Your price 1.47 is not allowed" in prompt and "may not set a price below 1.60" in prompt
    s = res["conditions"][0]["sessions"][0]
    assert s["agent0_calls"] == 8


def dhanda_cfg(disclosure="announced", policy="monopoly"):
    agent = dict(type="llm", provider="mock", model="llama3.2:3b", prompt="dhanda_style", temperature=0.7,
                 record_prompt=True)
    return dict(experiment="dhanda_mock", periods=8, sessions=1, seed=3, market=dict(MARKET),
                regulations=[dict(type="price_cap", value=1.70, enforcement="clip", disclosure=disclosure,
                                  text="It is ILLEGAL to set a price above {cap:.2f}.")],
                agents=[agent, dict(copy_of=0)], llm=dict(cache=dict(mode="off"), mock=dict(policy=policy)))


def test_dhanda_cap_announced_and_hidden(tmp_path):
    ann = run(dhanda_cfg("announced"), out_dir=tmp_path / "a")
    hid = run(dhanda_cfg("hidden"), out_dir=tmp_path / "h")
    pa = json.loads(texts(ann)[6]["prompt"])[0]["content"]
    ph = json.loads(texts(hid)[6]["prompt"])[0]["content"]
    assert "It is ILLEGAL to set a price above 1.70." in pa and "ILLEGAL" not in ph
    assert "between 1.00 and 2.50" in pa
    assert "Round 3: your price 1.70, your profit" in pa and "rival's price 1.70" in pa
    assert "Round 1:" not in json.loads(texts(ann)[14]["prompt"])[0]["content"]  # window of 5 at t = 7
    assert np.allclose(frames(ann).posted, 1.70) and np.allclose(frames(hid).posted, 1.70)
    assert frames(ann).proposed.max() == pytest.approx(1.92, abs=0.01)


def test_mixed_market_frozen_qlearning_vs_llm(tmp_path):
    cfg = dict(experiment="mixed", periods=30, sessions=1, seed=5, market=dict(MARKET),
               agents=[dict(type="qlearning", pretrain=dict(mode="frozen", stable_steps=2000, max_steps=300000)),
                       dict(type="llm", provider="mock", model="mock", prompt="fish_style")],
               llm=dict(cache=dict(mode="off"), mock=dict(policy="follow", noise=0.02)))
    res = run(cfg, out_dir=tmp_path)
    df = frames(res)
    assert res["conditions"][0]["sessions"][0]["periods"] == 30
    from tatonnement.scenarios.bertrand.demand import Economy, standard_grid

    grid = standard_grid(Economy.from_config(MARKET))[0]
    q = df[df.agent == 0].posted.to_numpy()
    assert np.allclose(np.min(np.abs(q[:, None] - grid[None, :]), axis=1), 0)


def test_parallel_sessions_and_agents_do_not_change_results(tmp_path):
    seq = run(dict(fish_cfg(periods=20, sessions=3), llm=dict(fish_cfg()["llm"], concurrency=1)),
              out_dir=tmp_path / "seq")
    par = run(dict(fish_cfg(periods=20, sessions=3), llm=dict(fish_cfg()["llm"], parallel_sessions=3)),
              out_dir=tmp_path / "par")
    assert strip_wall(seq["conditions"][0]["sessions"]) == strip_wall(par["conditions"][0]["sessions"])
    pd.testing.assert_frame_equal(frames(seq), frames(par))


def test_interval_and_summary_history(tmp_path):
    res = run(fish_cfg(periods=10, agent=dict(deployment=dict(interval=3, history="summary", window=20))),
              out_dir=tmp_path)
    s = res["conditions"][0]["sessions"][0]
    assert s["agent0_calls"] == 4  # periods 0, 3, 6, 9
    prompts = [json.loads(t["prompt"])[0]["content"] for t in texts(res) if "prompt" in t]
    assert any("grouped by the price you set" in p and "set in 3 rounds" in p for p in prompts)


def test_seed_auto_and_prefix_in_system():
    from tatonnement.core.types import AgentContext
    from tatonnement.scenarios.bertrand.demand import Economy
    from tatonnement.scenarios.bertrand.env import BertrandEnv
    from tatonnement.scenarios.bertrand.llm_pricing import PricingLLMAgent

    env = BertrandEnv(Economy.from_config(MARKET))
    svc = LLMService()
    a = PricingLLMAgent(svc, "ollama", "llama3.2:3b", prefix="goal", prefix_placement="system")
    a.reset(AgentContext(0, 2, 0, None, env.market_info()))
    msgs = a.messages(env.observe(0), None)
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("You set the price")
    assert not msgs[1]["content"].startswith("You set the price")
    req = a.request(msgs, a._coords(0))
    assert isinstance(req.params["seed"], int)
    b = PricingLLMAgent(svc, "deepseek", "deepseek-flash")
    b.reset(AgentContext(1, 2, 0, None, env.market_info()))
    assert "seed" not in b.request(msgs, b._coords(0)).params


def test_estimate_prices_every_prompt():
    from tatonnement.llm.estimate import estimate, format_estimate

    cfg = fish_cfg(periods=30, sessions=4, agent=dict(provider="deepseek", model="deepseek-flash"))
    cfg["agents"].append(dict(type="llm", provider="qwen", model="qwen-plus"))
    cfg["market"]["n"] = 3
    est = estimate(cfg, sample_sessions=2, output_tokens=300)
    rows = {r["provider"]: r for r in est["rows"]}
    ds, qw = rows["deepseek"], rows["qwen"]
    assert ds["calls"] == 2 * 30 * 4 and qw["calls"] == 30 * 4
    assert ds["completion_tokens"] == 300 * ds["calls"]
    assert ds["cost_mid"] > 0 and ds["currency"] == "CNY" and ds["cost_high"] == pytest.approx(3 * ds["cost_low"])
    assert not qw["price_known"] and "no price row" in format_estimate(est)


def test_cli_estimate_and_cache_export(tmp_path, capsys):
    import yaml

    from tatonnement.cli import main

    cfg = fish_cfg(periods=5, cache=dict(mode="use", path=str(tmp_path / "c.sqlite")))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["estimate", str(path)]) == 0
    assert "mock/mock: 10 calls" in capsys.readouterr().out
    assert main(["run", str(path), "--out", str(tmp_path / "runs")]) == 0
    assert "10 live calls" in capsys.readouterr().out
    assert main(["cache-export", "--experiment", "fish_mock", "--out", str(tmp_path / "x.sqlite"),
                 "--cache", str(tmp_path / "c.sqlite")]) == 0
    assert "exported 10 calls" in capsys.readouterr().out


def test_keys_in_config_are_refused(tmp_path):
    with pytest.raises(ValueError, match="environment variable"):
        run(fish_cfg(periods=2, agent=dict(api_key="sk-123")), out_dir=None)


def test_shipped_example_configs_load_and_the_mock_one_runs():
    import tatonnement as tt

    root = Path(tt.__file__).parent / "experiments"
    files = sorted(root.glob("*.yaml"))
    assert {f.stem for f in files} >= {"calvano2020", "fish_style_mock", "dhanda_cap", "fish_style_deepseek",
                                       "qlearning_vs_llm"}
    for f in files:
        tt.load(f)
    cfg = tt.load(root / "fish_style_mock.yaml")
    cfg["periods"] = 5
    res = tt.run(cfg, out_dir=None, sessions=1)
    assert res["status"] == "complete" and res["llm"]["live_calls"] == 10


class _TruncatingClient(MockClient):
    """Every second answer stops at max_tokens although it already shows a price."""

    def complete(self, request, hint=None):
        resp = super().complete(request, hint)
        if (hint or {}).get("coords", {}).get("attempt", 0) == 0 and (hint or {})["coords"]["period"] % 2 == 0:
            resp.finish_reason = "length"
        return resp


def test_truncated_answer_is_a_format_failure(tmp_path):
    svc = LLMService(client_override=_TruncatingClient(policy="follow", noise=0.0))
    res = run(fish_cfg(periods=6), out_dir=tmp_path, runtime={"llm": svc})
    f = frames(res)
    even = f[f.t % 2 == 0]
    assert (even.meta_parse_failures == 1).all() and (f[f.t % 2 == 1].meta_parse_failures == 0).all()
    failed = [json.loads(r["failed_responses"]) for r in texts(res) if r["t"] % 2 == 0]
    assert all(x[0]["status"] == "truncated" and x[0]["finish_reason"] == "length" for x in failed)
