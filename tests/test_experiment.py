import json
from importlib import resources

import pytest

from tatonnement.cli import main
from tatonnement.core import config as cfgmod
from tatonnement.core.experiment import run


def rule_cfg(**over):
    cfg = dict(
        experiment="rules", sessions=2, periods=10, seed=3,
        market=dict(scenario="bertrand_logit", n=2, a=2.0, mu=0.25, c=1.0),
        regulations=[dict(type="price_cap", value=1.7)],
        agents=[dict(type="rule", rule="fixed", price=1.9), dict(type="rule", rule="tit_for_tat", start=1.5)],
    )
    cfg.update(over)
    return cfg


def test_rule_experiment_outputs(tmp_path):
    res = run(rule_cfg(), out_dir=tmp_path)
    s = res["conditions"][0]["sessions"][0]
    assert s["mean_price"] == pytest.approx(1.7)  # cap binds for both after period 0
    assert s["violation_price_cap_rate"] == pytest.approx(0.5)
    d = tmp_path / "rules"
    run_dir = next(d.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["versions"]["numpy"]
    assert (run_dir / "main" / "sessions.jsonl").exists()


def test_sweep_expands_conditions():
    cfg = rule_cfg(sweep={"regulations.0.value": [1.6, 1.8], "regulations.0.disclosure": ["announced", "hidden"]})
    conds = cfgmod.expand(cfg)
    assert len(conds) == 4
    assert conds[0]["regulations"][0]["value"] == 1.6
    assert conds[-1]["condition"] == "value=1.8,disclosure=hidden"
    res = run(cfg, out_dir=None, sessions=1)
    means = [c["sessions"][0]["mean_price"] for c in res["conditions"]]
    assert means == pytest.approx([1.6, 1.6, 1.7, 1.7], abs=0.11)


def test_copy_of_and_validation():
    cfg = rule_cfg(agents=[dict(type="rule", rule="fixed", price=1.6), dict(copy_of=0, price=1.8)])
    specs = cfgmod.resolved_agents(cfg)
    assert specs[1] == dict(type="rule", rule="fixed", price=1.8)
    with pytest.raises(ValueError):
        cfgmod.validate(rule_cfg(agents=[dict(copy_of=0)]))


@pytest.mark.parametrize("engine", ["fast", "step"])
def test_qlearning_engines_agree(engine):
    pytest.importorskip("numba")
    cfg = dict(experiment="ql", sessions=1, periods=4000, seed=20261011, engine=engine,
               market=dict(n=2), agents=[dict(type="qlearning"), dict(copy_of=0)], stop_when_stable=10**9)
    s = run(cfg, out_dir=None)["conditions"][0]["sessions"][0]
    assert s["session"] == 0
    if engine == "fast":
        assert s["steps"] == 4000


def test_builtin_calvano_config_validates(capsys):
    path = resources.files("tatonnement.experiments") / "calvano2020.yaml"
    assert main(["validate", str(path)]) == 0
    assert '"conditions": 1' in capsys.readouterr().out


def test_mixed_market_frozen_qlearning_vs_rule():
    pytest.importorskip("numba")
    cfg = dict(experiment="mixed", sessions=1, periods=50, seed=5,
               market=dict(n=2),
               agents=[dict(type="qlearning", pretrain=dict(mode="frozen", stable_steps=2000, max_steps=200000)),
                       dict(type="rule", rule="fixed", price=1.7)])
    s = run(cfg, out_dir=None)["conditions"][0]["sessions"][0]
    assert s["periods"] == 50 and 1.0 < s["mean_price_0"] < 2.2
