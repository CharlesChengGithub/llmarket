"""Logit demand for differentiated Bertrand competition (Calvano et al., 2020).

The numerical code is ported from the replication code of Cheng (2026), "Do Below-Cost
Sales Bans Facilitate Algorithmic Collusion?". Expressions are kept operation-for-operation
identical so that results reproduce bit for bit; do not "simplify" them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import logsumexp


@dataclass
class Economy:
    """Logit demand q_i = exp((a_i - p_i)/mu) / (sum_j exp((a_j - p_j)/mu) + exp(a0/mu)).

    All quantities are in model units (Calvano et al. 2020: alpha = beta = 1). Display
    units for LLM agents are handled by :class:`Units`.
    """

    c: np.ndarray
    a: np.ndarray
    mu: float = 0.25
    a0: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "Economy":
        n = int(cfg.get("n", 2))
        return cls(
            np.broadcast_to(cfg.get("c", 1.0), (n,)).copy(),
            np.broadcast_to(cfg.get("a", 2.0), (n,)).copy(),
            float(cfg.get("mu", 0.25)),
            float(cfg.get("a0", 0.0)),
        )

    @property
    def n(self) -> int:
        return len(self.c)

    def quantity(self, p):
        p = np.asarray(p)
        z = (self.a - p) / self.mu
        outside = np.full((*z.shape[:-1], 1), self.a0 / self.mu)
        return np.exp(z - logsumexp(np.concatenate([z, outside], axis=-1), axis=-1, keepdims=True))

    def profit(self, p, F=0.0):
        return (np.asarray(p) - self.c) * self.quantity(p) - F

    def cs(self, p):
        """Consumer surplus (log-sum), per unit mass of consumers."""
        z = (self.a - np.asarray(p)) / self.mu
        outside = np.full((*z.shape[:-1], 1), self.a0 / self.mu)
        return self.mu * logsumexp(np.concatenate([z, outside], axis=-1), axis=-1)

    def nash(self, floor=None):
        """Static Bertrand-Nash prices, optionally with per-firm price floors."""
        f = np.full(len(self.c), -np.inf) if floor is None else np.broadcast_to(floor, self.c.shape)

        def fun(p):
            return p - np.maximum(self.c + self.mu / (1 - self.quantity(p)), f)

        sol = root(fun, np.maximum(self.c + 2 * self.mu, f), tol=1e-12)
        if np.max(np.abs(fun(sol.x))) > 1e-9:
            raise RuntimeError("Nash root residual")
        return sol.x

    def monopoly(self, floor=None):
        """Joint-profit-maximizing prices (common markup), optionally with floors."""
        f = np.full(len(self.c), -np.inf) if floor is None else np.broadcast_to(floor, self.c.shape)

        def equation(v):
            p = np.maximum(self.c + v, f)
            return v - self.mu - np.sum(self.profit(p))

        upper = 100 + np.max(self.a) + np.max(self.c) + np.max(np.where(np.isfinite(f), f, 0))
        v = brentq(equation, self.mu, upper, xtol=1e-14)
        return np.maximum(self.c + v, f)

    def br(self, p, i, floor=-np.inf):
        """Best response of firm i to the other prices in p."""
        p = np.asarray(p).copy()

        def foc(x):
            p[i] = x
            return x - self.c[i] - self.mu / (1 - self.quantity(p)[i])

        x = brentq(foc, self.c[i], max(50.0, float(np.max(self.a)) + 10.0), xtol=1e-13)
        return max(x, floor)


@dataclass(frozen=True)
class Units:
    """Display scaling used by Fish et al. (2024): prices x alpha, quantities x beta.

    Profits scale by alpha * beta. Model-unit code never sees display units; LLM prompts do.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def price_to_display(self, p):
        return np.asarray(p) * self.alpha

    def price_from_display(self, p):
        return np.asarray(p) / self.alpha

    def quantity_to_display(self, q):
        return np.asarray(q) * self.beta

    def profit_to_display(self, pi):
        return np.asarray(pi) * self.alpha * self.beta


def standard_grid(e: Economy, m: int = 15, xi: float = 0.1):
    """Calvano et al. (2020) grid: m points from p_N - xi*w to p_M + xi*w, w = p_M - p_N."""
    pn = e.nash()
    pm = e.monopoly()
    width = max(pm) - min(pn)
    lo = min(pn) - xi * width
    hi = max(pm) + xi * width
    h = (hi - lo) / (m - 1)
    return lo + np.arange(m, dtype=float) * h, lo, h


def make_grid(e: Economy, cfg: dict):
    """Price grid by name: standard, wide, extended_cost, low15 or half_step."""
    base, lo, h = standard_grid(e, int(cfg.get("m", 15)), float(cfg.get("xi", 0.1)))
    spec = cfg.get("grid", "standard")
    if spec == "standard":
        offsets = np.arange(len(base))
    elif spec == "wide":
        offsets = np.arange(-int(np.floor((lo - 0.8 * min(e.c)) / h)), len(base))
    elif spec == "extended_cost":
        offsets = np.arange(-int(np.floor((lo - min(e.c)) / h)) - 1, len(base))
    elif spec == "low15":
        return np.linspace(0.99, base[-1], 15)
    elif spec == "half_step":
        h = h / 2
        offsets = np.arange(-int(np.floor((lo - 0.8 * min(e.c)) / h)), 2 * (len(base) - 1) + 1)
    else:
        raise ValueError(spec)
    if "upper_bound" in cfg:
        top = max(int(offsets[-1]), int(np.ceil((float(cfg["upper_bound"]) - lo) / h)))
        offsets = np.arange(int(offsets[0]), top + 1)
    return lo + offsets * h


def floors_at(e: Economy, p, cfg: dict):
    """Per-firm price floors given the reference price profile p.

    Types: none; constant (cfg['value']); own_variable (marginal cost); b1 and lambda
    (c + F / max(q, lambda * q_N), lambda defaults to 0.1); b2 (c + F / max(q, q_N)).
    For the state-dependent types p is last period's price profile.
    """
    typ = cfg.get("type", "none")
    F = np.broadcast_to(cfg.get("F", 0.0), e.c.shape)
    if typ == "none":
        return np.full(len(e.c), -np.inf)
    if typ == "constant":
        return np.broadcast_to(cfg["value"], e.c.shape).copy()
    if typ == "own_variable":
        return e.c.copy()
    q = e.quantity(p)
    if typ == "b2":
        q = np.maximum(q, e.quantity(e.nash()))
    elif typ in ("b1", "lambda"):
        q = np.maximum(q, float(cfg.get("lambda", 0.1)) * e.quantity(e.nash()))
    else:
        raise ValueError(typ)
    return e.c + F / np.maximum(q, 1e-300)


def nu_value(m, n, k, beta):
    """Calvano et al.'s exploration coverage nu for grid size m, n firms, memory k."""
    return (m - 1.0) ** n / (float(m) ** (k * n + n + 1) * (-np.expm1(-beta * (n + 1))))


def beta_for_nu(m, n, k, nu):
    """Exploration decay beta that delivers coverage nu."""
    if m < 2:
        raise ValueError("nu undefined with a single action")
    v = (m - 1.0) ** n / (float(m) ** (k * n + n + 1) * nu)
    if not 0 < v < 1:
        raise ValueError("infeasible exploration coverage")
    return -np.log1p(-v) / (n + 1)
