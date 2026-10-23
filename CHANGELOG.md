# Changelog

## 0.1.1 (2026-10-23)

Renamed from tatonnement to llmarket: the package, the import name, the command and the
repository (github.com/CharlesChengGithub/llmarket; the old address redirects). Environment
variables starting with `TATONNEMENT_` now start with `LLMARKET_`, and the default cache folder
is `.llmarket/`; move an existing `.tatonnement/` folder there to keep its cached answers. Nothing
else changed: results are identical to 0.1.0.

## 0.1.0 (2026-10-23)

First public release. One scenario, algorithmic pricing in a repeated Bertrand market with
logit demand.

- **Agents.** Tabular Q-learning (Calvano et al. 2020), with a Numba kernel for markets of
  Q-learners only and a step-by-step engine for mixed markets; the two give identical action
  sequences and Q-tables. LLM pricing agents with the `fish_style` and `dhanda_style` prompt
  packs or packs of your own. Rule-based agents (fixed, nash, monopoly, random, tit_for_tat,
  grim_trigger, undercut).
- **Regulation.** Price floors (constant, marginal cost, three full-cost definitions) and price
  caps; enforcement by clipping, rejection, penalty or monitoring; announced or hidden.
- **LLM layer.** OpenAI-compatible providers (DeepSeek, Qwen, Ollama, OpenAI, OpenRouter,
  SiliconFlow, custom endpoints); a response cache keyed by request and position in the
  experiment (resume, replay, export); deterministic per-call seeds; a cost ledger with a budget
  guard; an offline cost estimator; an offline mock client.
- **Experiments.** YAML configurations with sweeps and named conditions; shipped examples
  (`tatonnement list`); parquet and JSONL outputs with a manifest.
- **Validation.** 46 sessions of Cheng (2026) reproduced bit for bit in the test suite; the
  1,000-session Calvano baseline matches exactly; a replication of Dhanda (2026) on a local
  llama3.2 3B model; a Fish-style demonstration on DeepSeek (three sessions).
