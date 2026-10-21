"""tatonnement: economics-first agent-based simulation with LLM, learning and rule-based agents."""

__version__ = "0.1.0"


def load(path):
    """Load and validate an experiment configuration: a YAML path or the name of a shipped experiment."""
    from .core.config import load as _load

    return _load(path)


def run(cfg, out_dir="runs", sessions=None, runtime=None):
    """Run an experiment: a mapping, a path to a YAML file, or the name of a shipped experiment."""
    from .core.experiment import run as _run

    if not isinstance(cfg, dict):
        cfg = load(cfg)
    return _run(cfg, out_dir=out_dir, sessions=sessions, runtime=runtime)


def estimate(cfg, sample_sessions=2, output_tokens=None):
    """Estimate LLM calls, tokens and cost of a configuration offline (no network)."""
    from .llm.estimate import estimate as _estimate

    if not isinstance(cfg, dict):
        cfg = load(cfg)
    return _estimate(cfg, sample_sessions, output_tokens)
