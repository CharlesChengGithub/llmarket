import pytest

from tatonnement.core.config import expand, resolved_agents

BASE = dict(experiment="x", periods=3, market=dict(n=2),
            agents=[dict(type="llm", instructions="neutral"), dict(copy_of=0)])


def test_listed_conditions():
    cfg = dict(BASE, conditions=[
        dict(name="neutral"),
        dict(name="aggressive", set={"agents.0.instructions": "aggressive"}),
        dict(name="cap", set={"regulations": [dict(type="price_cap", value=1.7)]})])
    out = expand(cfg)
    assert [c["condition"] for c in out] == ["neutral", "aggressive", "cap"]
    assert [a["instructions"] for a in resolved_agents(out[1])] == ["aggressive", "aggressive"]
    assert "regulations" not in out[0] and out[2]["regulations"][0]["value"] == 1.7
    assert "conditions" not in out[0]


def test_conditions_cross_sweep():
    cfg = dict(BASE, conditions=[dict(name="a"), dict(name="b")], sweep={"periods": [3, 5]})
    assert [c["condition"] for c in expand(cfg)] == ["a,periods=3", "a,periods=5", "b,periods=3", "b,periods=5"]


@pytest.mark.parametrize("bad", [[dict(name="a"), dict(name="a")], [dict(set={})], [dict(name="a/b")]])
def test_condition_names_checked(bad):
    with pytest.raises(ValueError):
        expand(dict(BASE, conditions=bad))
