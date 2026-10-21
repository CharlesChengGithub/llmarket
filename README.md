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
- LLM pricing agents on any OpenAI-compatible endpoint (DeepSeek, Qwen/DashScope, Ollama,
  OpenAI, OpenRouter, SiliconFlow, or a custom base URL):
  - prompt packs with the structure of Fish et al. (2024) and a reconstruction of the setting
    of Dhanda (2026), or your own templates from a YAML file;
  - deployment choices: raw or summarized history, window, rival visibility, decision interval;
  - SQLite response cache keyed by prompt and call coordinates (resume interrupted runs for free,
    replay a shipped cache offline, export one experiment's calls);
  - cost ledger with a price table, a budget guard and `tatonnement estimate` for offline cost
    estimates; API keys are read from environment variables only;
  - an offline mock client used by the tests and examples.

```bash
tatonnement validate src/tatonnement/experiments/calvano2020.yaml
tatonnement run src/tatonnement/experiments/calvano2020.yaml --sessions 20
tatonnement run src/tatonnement/experiments/fish_style_mock.yaml        # offline, no cost
tatonnement estimate src/tatonnement/experiments/fish_style_deepseek.yaml
```

License: MIT.
