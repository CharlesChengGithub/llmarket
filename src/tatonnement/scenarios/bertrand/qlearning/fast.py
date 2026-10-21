"""Fast path: batch self-play Q-learning in joint-action-index space (Numba).

Ported from the replication code of Cheng (2026), which reproduces Calvano et al. (2020)
Table 1 and adds price floors. The kernel's arithmetic and random-number consumption are
unchanged, so sessions reproduce bit for bit given the same seed:

* ``np.random.seed(seed)``; one uniform draw selects the initial state;
* each period, for each firm in index order, two uniform draws ``u, v``: explore if
  ``u < eps``, and if so take action ``start + int(v * (count - start))``;
* every firm's target uses its own Q before any update in that period.

Without Numba the same functions run as plain Python (slow, for tests only).
"""
from __future__ import annotations

import warnings

import numpy as np

from ..demand import Economy, beta_for_nu, floors_at, make_grid, nu_value

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


DYNAMIC_FLOORS = ("b1", "b2", "lambda")

CHECKPOINTS = np.array(
    [10**5, 25 * 10**4, 5 * 10**5, 10**6, 2 * 10**6, 5 * 10**6, 10**7, 2 * 10**7,
     5 * 10**7, 10**8, 2 * 10**8, 5 * 10**8, 10**9],
    dtype=np.int64,
)


def prepare(cfg: dict) -> dict:
    if not HAVE_NUMBA:
        warnings.warn("numba is not installed: the Q-learning kernel runs as plain Python and a Calvano "
                      "session takes hours instead of seconds (pip install numba)", RuntimeWarning, stacklevel=2)
    return _prepare(cfg)


def _prepare(cfg: dict) -> dict:
    """Precompute grids, feasible action sets, rewards and initial Q for a configuration.

    Keys of cfg: n, c, a, mu, a0 (demand); m, xi, grid, upper_bound (price grid); k (memory,
    0 or 1); delta; beta or beta_rule='nu' with nu and beta_effective_m; floor (dict for
    :func:`floors_at`, plus F for fixed costs); compression (drop infeasible actions).
    """
    e = Economy.from_config(cfg)
    n = len(e.c)
    k = int(cfg.get("k", 1))
    if k not in (0, 1):
        raise ValueError("memory k must be 0 or 1")
    grid = make_grid(e, cfg)
    floor = cfg.get("floor", {"type": "none"})
    typ = floor.get("type", "none")
    dynamic = typ in DYNAMIC_FLOORS
    if dynamic and k != 1:
        raise ValueError("state-dependent floors need k=1")
    compression = cfg.get("compression", True) and not dynamic
    initial_floor = floors_at(e, e.nash(), floor)
    ids = []
    for i in range(n):
        valid = np.flatnonzero(grid >= initial_floor[i])
        if not len(valid):
            valid = np.array([len(grid) - 1])
        ids.append(valid if compression else np.arange(len(grid)))
    counts = np.array([len(a) for a in ids], dtype=np.int64)
    if np.max(counts) > 127:
        raise ValueError("int8 action index exceeds 127")
    strides = np.array([int(np.prod(counts[i + 1:])) for i in range(n)], np.int64)
    N = int(np.prod(counts))
    S = N if k else 1
    M = int(max(counts))
    actions = np.array(np.unravel_index(np.arange(N), tuple(counts))).T.copy()
    prices = np.zeros((n, M))
    global_ids = np.full((n, M), -1, np.int16)
    for i in range(n):
        prices[i, : counts[i]] = grid[ids[i]]
        global_ids[i, : counts[i]] = ids[i]
    profiles = np.column_stack([prices[i, actions[:, i]] for i in range(n)])
    F = np.broadcast_to(floor.get("F", 0.0), (n,)).copy()
    rewards = np.ascontiguousarray(e.profit(profiles, F))
    starts = np.zeros((n, S), np.int64)
    empty = np.zeros((n, S), np.bool_)
    if dynamic:
        q = e.quantity(profiles)
        if typ == "b2":
            q = np.maximum(q, e.quantity(e.nash()))
        else:
            q = np.maximum(q, float(floor.get("lambda", 0.1)) * e.quantity(e.nash()))
        floor_values = e.c + F / np.maximum(q, 1e-300)
    else:
        floor_values = np.tile(initial_floor, (S, 1))
    for i in range(n):
        raw = np.searchsorted(prices[i, : counts[i]], floor_values[:, i], side="left")
        empty[i] = raw >= counts[i]
        starts[i] = np.minimum(raw, counts[i] - 1)
    # Initial states: self-feasible profiles; transitions can still reach other states.
    feasible = np.ones(N, dtype=bool)
    for i in range(n):
        feasible &= actions[:, i] >= starts[i, np.arange(N) if k else np.zeros(N, int)]
    initial_states = np.flatnonzero(feasible) if k else np.array([0])
    if len(initial_states) == 0:
        initial_states = np.array([N - 1 if k else 0])
    delta = float(cfg.get("delta", 0.95))
    q0 = np.zeros((n, S, M))
    # Initial Q: discounted payoff of each own action against uniformly random feasible rivals.
    if not dynamic:
        for i in range(n):
            valid = np.ones(N, dtype=bool)
            for j in range(n):
                if j != i:
                    valid &= actions[:, j] >= starts[j, 0]
            for a in range(counts[i]):
                q0[i, :, a] = rewards[valid & (actions[:, i] == a), i].mean() / (1 - delta)
    else:
        for s in range(S):
            for i in range(n):
                valid = np.ones(N, dtype=bool)
                for j in range(n):
                    if j != i:
                        valid &= actions[:, j] >= starts[j, s]
                for a in range(counts[i]):
                    q0[i, s, a] = rewards[valid & (actions[:, i] == a), i].mean() / (1 - delta)
    beta = float(cfg.get("beta", 4e-6))
    if cfg.get("beta_rule", "fixed") == "nu":
        feasible_counts = counts - starts[:, 0]
        if dynamic:
            effective_m = float(cfg.get("beta_effective_m", len(grid)))
        elif len(set(feasible_counts)) == 1:
            effective_m = int(feasible_counts[0])
        else:
            # Common beta from the geometric mean of feasible action counts.
            effective_m = float(np.prod(feasible_counts, dtype=float) ** (1 / n))
        target = float(cfg.get("nu", nu_value(15, 2, 1, 4e-6)))
        beta = beta_for_nu(effective_m, n, k, target)
    return dict(
        e=e, grid=grid, prices=prices, global_ids=global_ids, counts=counts, strides=strides,
        actions=actions, profiles=profiles, rewards=rewards, starts=starts, empty=empty,
        initial_states=initial_states, q0=q0, beta=beta, delta=delta, k=k,
        floor_values=floor_values, quantities=np.ascontiguousarray(e.quantity(profiles)),
        q_nash=e.quantity(e.nash()), F=F,
    )


@njit(cache=True)
def greedy_row(row, start, count):
    """First maximizer of row[start:count] (ties go to the lowest index)."""
    best = start
    value = row[start]
    for a in range(start + 1, count):
        if row[a] > value:
            value = row[a]
            best = a
    return best


@njit(cache=True)
def train(q0, rewards, prices, counts, strides, starts, empty, initial_states, k, alpha, beta,
          delta, max_steps, stable_steps, seed, checkpoints, epsilon_start=0):
    """Self-play Q-learning until every greedy policy is unchanged for stable_steps periods.

    Returns (Q, policy, last_state, steps, converged, checkpoint_means, empty_visits,
    last_change). checkpoint_means rows: period, then running 10,000-period means of each
    firm's price and profit.
    """
    np.random.seed(seed)
    Q = q0.copy()
    n, S, M = Q.shape
    policy = np.empty((n, S), np.int8)
    for i in range(n):
        for s in range(S):
            policy[i, s] = greedy_row(Q[i, s], starts[i, s], counts[i])
    s = initial_states[int(np.random.random() * len(initial_states))]
    stable = 0
    acts = np.empty(n, np.int64)
    targets = np.empty(n)
    window = 10000
    ring = np.zeros((window, 2 * n))
    running = np.zeros(2 * n)
    checks = np.full((len(checkpoints), 1 + 2 * n), np.nan)
    ci = 0
    empty_visits = 0
    last_change = 0
    t = 0
    for t in range(max_steps):
        eps = np.exp(-beta * (t + epsilon_start))
        joint = 0
        for i in range(n):
            u = np.random.random()
            v = np.random.random()  # both draws every period keep common random numbers
            a = int(policy[i, s])
            if u < eps:
                a = starts[i, s] + int(v * (counts[i] - starts[i, s]))
            acts[i] = a
            joint += a * strides[i]
            empty_visits += int(empty[i, s])
        sn = joint if k else 0
        for i in range(n):
            targets[i] = rewards[joint, i] + delta * Q[i, sn, policy[i, sn]]
        changed = False
        for i in range(n):
            a = acts[i]
            Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * targets[i]
            old = policy[i, s]
            new = greedy_row(Q[i, s], starts[i, s], counts[i])
            policy[i, s] = new
            if old != new:
                changed = True
        if changed:
            stable = 0
            last_change = t + 1
        else:
            stable += 1
        idx = t % window
        for i in range(n):
            running[i] += prices[i, acts[i]] - ring[idx, i]
            ring[idx, i] = prices[i, acts[i]]
            running[n + i] += rewards[joint, i] - ring[idx, n + i]
            ring[idx, n + i] = rewards[joint, i]
        if ci < len(checkpoints) and t + 1 == checkpoints[ci]:
            checks[ci, 0] = t + 1
            for j in range(2 * n):
                checks[ci, j + 1] = running[j] / min(t + 1, window)
            ci += 1
        s = sn
        if stable >= stable_steps:
            break
    return Q, policy, s, t + 1, stable >= stable_steps, checks[:ci], empty_visits, last_change


@njit(cache=True)
def cycle_from(policy, strides, k, start):
    """Deterministic greedy path from start; returns the states and joint actions of its cycle."""
    S = policy.shape[1]
    seen = np.full(S, -1, np.int64)
    path = np.empty(S, np.int64)
    joint_path = np.empty(S, np.int64)
    s = start
    t = 0
    while seen[s] < 0:
        seen[s] = t
        path[t] = s
        joint = 0
        for i in range(policy.shape[0]):
            joint += int(policy[i, s]) * strides[i]
        joint_path[t] = joint
        t += 1
        s = joint if k else 0
    begin = seen[s]
    return path[begin:t], joint_path[begin:t]


@njit(cache=True)
def mdp_qstar(policy, rewards, counts, strides, starts, k, delta, tolerance=1e-12):
    """Each firm's optimal Q against the rivals' fixed greedy policies, and its relative loss."""
    n, S = policy.shape
    M = int(max(counts))
    loss = np.zeros((n, S))
    Qstar = np.zeros((n, S, M))
    residual = 0.0
    for i in range(n):
        nexts = np.zeros((S, M), np.int64)
        r = np.zeros((S, M))
        v = np.zeros(S)
        for s in range(S):
            base = 0
            for j in range(n):
                if j != i:
                    base += int(policy[j, s]) * strides[j]
            for a in range(starts[i, s], counts[i]):
                joint = base + a * strides[i]
                nexts[s, a] = joint if k else 0
                r[s, a] = rewards[joint, i]
        diff = 0.0
        for _ in range(100000):
            diff = 0.0
            vn = np.empty(S)
            for s in range(S):
                best = -np.inf
                for a in range(starts[i, s], counts[i]):
                    value = r[s, a] + delta * v[nexts[s, a]]
                    if value > best:
                        best = value
                vn[s] = best
                diff = max(diff, abs(best - v[s]))
            v = vn
            if diff < tolerance:
                break
        residual = max(residual, diff)
        for s in range(S):
            best = -np.inf
            for a in range(M):
                Qstar[i, s, a] = -np.inf
            for a in range(starts[i, s], counts[i]):
                Qstar[i, s, a] = r[s, a] + delta * v[nexts[s, a]]
                best = max(best, Qstar[i, s, a])
            chosen = Qstar[i, s, policy[i, s]]
            loss[i, s] = (best - chosen) / max(abs(best), 1e-15)
    return Qstar, loss, residual


@njit(cache=True)
def policy_values(policy, rewards, strides, k, delta):
    """Discounted value of following the joint greedy policy from every state."""
    n, S = policy.shape
    values = np.zeros((n, S))
    done = np.zeros(S, np.bool_)
    nexts = np.zeros(S, np.int64)
    rs = np.zeros((n, S))
    for s in range(S):
        joint = 0
        for i in range(n):
            joint += int(policy[i, s]) * strides[i]
        nexts[s] = joint if k else 0
        for i in range(n):
            rs[i, s] = rewards[joint, i]
    mark = np.full(S, -1, np.int64)
    position = np.zeros(S, np.int64)
    path = np.empty(S, np.int64)
    for start in range(S):
        if done[start]:
            continue
        s = start
        t = 0
        while not done[s] and mark[s] != start:
            mark[s] = start
            position[s] = t
            path[t] = s
            t += 1
            s = nexts[s]
        prefix = t
        if not done[s]:
            begin = position[s]
            length = t - begin
            for offset in range(length):
                node = path[begin + offset]
                for i in range(n):
                    v = 0.0
                    discount = 1.0
                    for j in range(length):
                        v += discount * rs[i, path[begin + (offset + j) % length]]
                        discount *= delta
                    values[i, node] = v / (1 - discount)
                done[node] = True
            prefix = begin
        for t0 in range(prefix - 1, -1, -1):
            node = path[t0]
            for i in range(n):
                values[i, node] = rs[i, node] + delta * values[i, nexts[node]]
            done[node] = True
    return values


def run_session(prep: dict, seed: int, alpha: float = 0.15, max_steps: int = 10**9,
                stable_steps: int = 100000, epsilon_start: int = 0) -> dict:
    """Train one session and return the raw kernel outputs as a dict."""
    p = prep
    Q, policy, last, steps, conv, checks, empty_visits, last_change = train(
        p["q0"], p["rewards"], p["prices"], p["counts"], p["strides"], p["starts"], p["empty"],
        p["initial_states"], p["k"], float(alpha), p["beta"], p["delta"], int(max_steps),
        int(stable_steps), int(seed), CHECKPOINTS, int(epsilon_start),
    )
    return dict(Q=Q, policy=policy, last_state=int(last), steps=int(steps), converged=bool(conv),
                checkpoints=checks, empty_visits=int(empty_visits), last_policy_change=int(last_change))


def session_seed(entropy: int, session: int) -> int:
    """Seed used by Cheng (2026): first uint32 of SeedSequence(entropy, spawn_key=(session,))."""
    seq = np.random.SeedSequence(int(entropy), spawn_key=(int(session),))
    return int(seq.generate_state(1, dtype=np.uint32)[0])
