"""Bit-for-bit reproduction of recorded sessions from Cheng (2026)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("numba")

from llmarket.scenarios.bertrand.qlearning.analysis import analyze
from llmarket.scenarios.bertrand.qlearning.fast import prepare, run_session, session_seed

GOLDENS = json.loads((Path(__file__).parent / "data" / "paper14_goldens.json").read_text())["goldens"]
CASES = [(g["condition"], g["config"], s) for g in GOLDENS for s in g["sessions"]]


@pytest.fixture(scope="module")
def prepared():
    return {g["condition"]: prepare(dict(g["config"])) for g in GOLDENS}


@pytest.mark.parametrize("condition,cfg,rec", CASES, ids=[f"{c}-{r['session']}" for c, _, r in CASES])
def test_session_reproduces(prepared, condition, cfg, rec):
    prep = prepared[condition]
    out = run_session(prep, rec["seed"], alpha=float(cfg.get("alpha", 0.15)))
    assert out["steps"] == rec["steps"]
    assert out["last_state"] == rec["last_state"]
    assert out["last_policy_change"] == rec["last_policy_change"]
    assert out["converged"] == rec["converged"]
    res = analyze(prep, out["policy"], out["last_state"], full=False)
    assert res["cycle_length"] == rec["cycle_length"]
    assert res["delta_gain"] == pytest.approx(rec["delta_gain"], abs=1e-12)
    assert res["mean_price"] == pytest.approx(rec["mean_price"], abs=1e-12)


def test_phase1_seed_rule():
    baseline = next(g for g in GOLDENS if g["condition"] == "baseline")
    for rec in baseline["sessions"]:
        assert session_seed(20261011, rec["session"]) == rec["seed"]
