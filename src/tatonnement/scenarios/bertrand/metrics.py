"""Session-level outcome measures for Bertrand experiments."""
from __future__ import annotations

import numpy as np


def final_window(periods: int, window: int | float | None) -> int:
    """Number of final periods to average: an int, a fraction of the session, or 10% by default."""
    if window is None:
        return max(1, int(round(0.1 * periods)))
    if isinstance(window, float) and window < 1:
        return max(1, int(round(window * periods)))
    return max(1, min(int(window), periods))


def summarize(records, env, n: int, window: int | float | None = None) -> dict:
    """Final-window prices, profits, Delta and price index; violation and fine totals.

    Delta = (mean profit - Nash profit) / (monopoly profit - Nash profit) and the price index
    = (mean price - Nash price) / (monopoly price - Nash price), both with unconstrained
    static benchmarks (0 = competitive, 1 = joint monopoly).
    """
    posted = records.array("posted", n)
    profit = records.array("profit", n)
    T = len(posted)
    w = final_window(T, window)
    p = posted[-w:]
    pi = profit[-w:]
    pn = float(np.mean(env.nash))
    pm = float(np.mean(env.monopoly))
    out = dict(
        recorded_periods=T,
        window=w,
        mean_price=float(p.mean()),
        mean_profit=float(pi.mean()),
        delta=env.profit_gain(pi.mean(axis=0)),
        price_index=float((p.mean() - pn) / (pm - pn)),
        price_sd=float(p.std()),
    )
    for i in range(n):
        out[f"mean_price_{i}"] = float(p[:, i].mean())
        out[f"mean_profit_{i}"] = float(pi[:, i].mean())
    rows = records.rows
    for key in sorted({k for r in rows for k in r if k.startswith("violation_")}):
        out[key + "_rate"] = float(np.mean([bool(r.get(key, False)) for r in rows]))
    out["fines"] = float(sum(r.get("fine", 0.0) for r in rows))
    return out
