# tatonnement

Economics-first agent-based simulation in which large-language-model (LLM) agents,
reinforcement-learning agents and rule-based agents interact in the same market.

The first built-in scenario is repeated Bertrand competition with logit demand
(Calvano et al., 2020; Fish et al., 2024) under pluggable price regulation: price floors
with five cost definitions and price caps, each with separate enforcement (clip, reject,
penalty, monitor) and disclosure (announced, hidden) settings.

Status: pre-release (v0.1 in development). Not yet published.

## What works now
- Logit Bertrand market, static benchmarks under floors.
- Q-learning in two engines: a Numba batch kernel that reproduces Calvano et al. (2020) and
  Cheng (2026) session by session, bit for bit; and a step-level agent that plays against
  any other agent and is tested to be identical to the kernel.
- Rule-based agents, regulation layer, simulation loop, YAML experiments with sweeps, CLI.

LLM agents (OpenAI-compatible endpoints including DeepSeek, Qwen and Ollama, with response
caching and cost control) are next.

```bash
tatonnement validate src/tatonnement/experiments/calvano2020.yaml
tatonnement run src/tatonnement/experiments/calvano2020.yaml --sessions 20
```

License: MIT.
