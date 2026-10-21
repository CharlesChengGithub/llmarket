"""Seeds: one root seed per experiment, a child per session, a grandchild per agent."""
from __future__ import annotations

import numpy as np


def session_sequence(root: int, session: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(root), spawn_key=(int(session),))


def agent_generators(root: int, session: int, n: int) -> list[np.random.Generator]:
    """Independent generators for the agents of one session (also the environment, last)."""
    children = session_sequence(root, session).spawn(n + 1)
    return [np.random.Generator(np.random.PCG64(c)) for c in children]


def kernel_seed(root: int, session: int) -> int:
    """uint32 seed in the style of Cheng (2026) for kernel-compatible shared streams."""
    return int(session_sequence(root, session).generate_state(1, dtype=np.uint32)[0])
