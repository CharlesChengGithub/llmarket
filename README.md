# tatonnement

An agent-based simulation toolkit built for economics: LLM agents, reinforcement learners
and rule-based agents can trade in the same market.

The first built-in scenario is algorithmic pricing. Firms play repeated Bertrand competition
with logit demand (Calvano et al., 2020; Fish et al., 2024) under price regulation that you
can switch on and off.

[中文说明](README.zh-CN.md)

Status: v0.1, an early release. Install from GitHub; not yet on PyPI.

## Why

Research on LLM pricing agents is growing quickly, but most of its code is either missing or
written once for a single paper. tatonnement aims to be a shared experimental platform for
this work. It offers:

- **Exact Q-learning baselines.** The Q-learning engine reproduces the Calvano et al. (2020)
  baseline session by session, bit for bit. It is checked against the 1,000 recorded sessions
  of Cheng (2026).
- **One market for every kind of agent.** LLM, Q-learning and rule-based agents compete
  side by side. A Q-learner can be frozen after self-play or keep learning, and different LLM
  providers can be mixed in one market.
- **Regulation as a module.**
  - Price floors come in five cost definitions: constant, marginal cost, and three full-cost
    floors.
  - Price caps are also available.
  - Enforcement is a separate setting: clip, reject and re-ask, penalty, or monitor only.
  - Disclosure is separate too: announced to the agents or hidden.
- **Deployment choices for LLM agents.** You choose between raw or summarized history, the
  length of the history window, whether rivals' prices are visible, and how often the agent
  decides.
- **Reproducible and affordable LLM runs.**
  - A response cache keyed by the prompt and the call's position in the experiment lets you
    resume an interrupted run without paying again, and replay a shipped cache offline.
  - Each call gets a deterministic seed where the provider supports one.
  - A cost ledger with a budget guard caps spending, and an offline cost estimator tells you
    the price before you run.
  - An offline mock client lets you build and test experiments without any network access.

## Install

```bash
git clone https://github.com/CharlesChengGithub/tatonnement.git
cd tatonnement
pip install -e ".[analysis]"     # pandas/pyarrow write parquet output (CSV without them)
```

Python 3.10 or newer. The core needs numpy, scipy, numba, httpx and pyyaml.

## Quick start

```bash
tatonnement list                                  # the experiments shipped with the package

# Calvano et al. (2020) baseline, 20 sessions, on the fast engine (a few seconds)
tatonnement run calvano2020 --sessions 20

# Two LLM pricing agents with the Fish et al. structure, offline mock answers, no cost
tatonnement run fish_style_mock

# What would the same experiment cost on DeepSeek? (offline; nothing is sent)
tatonnement estimate fish_style_deepseek
```

`CONFIG` can be the name of a shipped experiment or the path to your own YAML file. To start
your own, copy one of the files in `src/tatonnement/experiments/`.

The Dhanda (2026) replications run free on a local model:

```bash
ollama pull llama3.2:3b        # Ollama from https://ollama.com, about 2 GB
ollama serve                   # in another terminal, unless the Ollama app is running
tatonnement run dhanda_main    # Table 1: 2 conditions x 12 sessions x 30 rounds
tatonnement run dhanda_cap     # Table 2: 10 conditions
```

From Python:

```python
import tatonnement as tt

res = tt.run("fish_style_mock", out_dir="runs")
print(res["conditions"][0]["sessions"][0]["delta"])
```

## Experiments are YAML files

```yaml
experiment: dhanda_cap
sessions: 12
periods: 30
seed: 20261021
market: {scenario: bertrand_logit, n: 2, a: 2.0, a0: 0.0, mu: 0.25, c: 1.0}
regulations:
  - {type: price_cap, value: 1.70, enforcement: clip, disclosure: announced,
     text: "It is ILLEGAL to set a price above {cap:.2f}."}
agents:
  - {type: llm, provider: ollama, model: "llama3.2:3b", prompt: dhanda_style, temperature: 0.7}
  - {copy_of: 0}
llm:
  cache: {mode: use}
  on_parse_failure: {retries: 10, then: keep_last_price}
metrics: {final_window: 10}
sweep:                                   # Cartesian product; one condition per combination
  regulations.0.value: [1.55, 1.60, 1.70, 1.80, 1.90]
  regulations.0.disclosure: [announced, hidden]
```

For designs that are not a full Cartesian product, list named conditions, each a set of
dotted-path overrides. With both `conditions` and `sweep`, every listed condition is crossed
with every sweep combination.

```yaml
conditions:
  - {name: neutral}
  - {name: aggressive, set: {agents.0.instructions: aggressive}}
  - name: cap
    set: {regulations: [{type: price_cap, value: 1.70, enforcement: clip, disclosure: announced,
                         text: "It is ILLEGAL to set a price above {cap:.2f}."}]}
```

`tatonnement validate FILE` checks a file and counts its conditions without running anything.

### Market

`market` holds the logit-demand parameters (`n`, `a`, `a0`, `mu`, `c`). It also takes a
per-period fixed cost `F`, which the full-cost floors use, and display `units {alpha, beta}`:
LLM agents see prices multiplied by α and quantities by β, as in Fish et al. (2024). Everything
else (Q-learning, metrics, outputs) works in model units.

### Agents

| type | what it is | main keys |
|---|---|---|
| `rule` | fixed, nash, monopoly, random, tit_for_tat, grim_trigger, undercut | `rule`, rule parameters |
| `qlearning` | tabular Q-learning (Calvano et al. 2020) | `alpha`, `beta`, `delta`, `k`, `m`, `xi`, `price_map`, `pretrain {mode: frozen \| adaptive}` |
| `llm` | LLM pricing agent | `provider`, `model`, `prompt`, `prefix`, `deployment`, `temperature`, `max_tokens`, `retries`, `on_failure` |

`{copy_of: j}` adds a fresh agent with the settings of agent `j`.

When every agent is a Q-learner, the experiment runs on a Numba kernel. Otherwise it runs on
the step-by-step engine. The two engines are tested to give identical action sequences and
Q-tables for the same seed. In mixed markets the Q-learner's state is last period's prices
mapped to its grid.

### Regulations

| type | keys |
|---|---|
| `price_floor` | `kind`: `constant` (with `value`), `own_variable` (marginal cost), `b1`, `b2`, `lambda` (full-cost floors c + F / max(q_{t-1}, λ q_N), with λ = 0.1, 1 and `lam` respectively) |
| `price_cap` | `value` |

Every regulation also takes:

- `enforcement`: `clip`, `reject`, `penalty` or `monitor`;
- `disclosure`: `announced` or `hidden`;
- `text`: the sentence shown to LLM agents when the regulation is announced.

If a floor and a cap conflict, the floor prevails and the conflict is flagged in the records.

## LLM agents

### Providers

Every provider is reached through its OpenAI-compatible `/chat/completions` endpoint.

| provider | key from environment variable |
|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` |
| `qwen` (DashScope compatible mode) | `DASHSCOPE_API_KEY` |
| `ollama` (local, `http://localhost:11434/v1`) | none |
| `openai` | `OPENAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `siliconflow` | `SILICONFLOW_API_KEY` |
| any other | `provider: {name: my_endpoint, base_url: ..., key_env: MY_KEY}` |

Keys are read from the environment at call time. A configuration that contains a key is
rejected, and keys never appear in caches, logs or manifests.

DeepSeek models run in thinking mode by default. In that mode the provider ignores
`temperature`, and the hidden reasoning counts against `max_tokens`, so a small `max_tokens`
can leave the visible answer empty. To compare with work on non-reasoning models (Fish et al.
2024, for instance), switch it off and report the choice:

```yaml
agents:
  - {type: llm, provider: deepseek, model: deepseek-flash, prompt: fish_style,
     extra_body: {thinking: {type: disabled}}}
```

### Prompt packs

- `fish_style` follows the five-part structure of Fish, Gonczarowski and Shorrer (2024):
  1. a goal prefix;
  2. product information, including a price ceiling drawn once per session from
     U[1.5, 2.5] times the monopoly price;
  3. plans and insights written by the previous call;
  4. the last 100 rounds of market history;
  5. an answer template.

  The wording is ours, including the three prefixes `goal`, `explore` and
  `protect_profit`, so results are not directly comparable with the paper. To use the
  original texts, write them into a YAML prompt pack (`prompt: {file: my_pack.yaml}`). This
  package does not redistribute them.
- `dhanda_style` reconstructs the setting described by Dhanda (2026): cost, a legal range of
  [1.0, 2.5], the competitive logic, the last five rounds, and a single-number answer. The
  paper does not print its prompt, so the wording is ours. The condition sentences it quotes
  are available verbatim in `tatonnement.scenarios.bertrand.prompts` (CC BY 4.0).

Any pack can be replaced by your own templates.

### Deployment

`deployment` sets what the agent sees and how often it decides:

- `history`: `raw` gives one entry per round; `summary` groups rounds by own price level, with
  counts, mean quantity, mean profit and the mean rival price.
- `window`: how many past rounds are shown.
- `rival_visible`: whether rivals' prices appear.
- `interval`: the agent asks the model every k periods and keeps its price in between.
- `decimals`: rounding of the numbers shown.
- `show`: which history fields are shown.
- `order`: the order in which rounds are listed.
- `own_price`: `posted` (the price after enforcement) or `proposed` (what the agent asked for).

### Format failures

Answers are split by their template headings, and the price is read from the price section.
`pick` (set in the prompt pack, or per agent) decides which number counts: `first` (the default
for `fish_style`), `last`, or `only`. With `only`, the default for `dhanda_style`, the reply must
be a bare number, with at most a `Price:` label, a `$` sign or a final full stop around it. Small
models often reason aloud before answering, and the first or last number of such a reply is
frequently the cost, a cap or the rival's price rather than the decision. Strict parsing re-asks
instead of guessing. An answer counts as unusable when:

- a field is missing;
- the price is not a number, or (with `only`) other text surrounds it;
- the price is negative;
- the price is above ten times the monopoly price.

An unusable answer is re-asked up to `retries` times. After that, `on_failure: stop` ends the
session (as in Fish et al.), and `on_failure: keep_last_price` repeats the last price.

### Cache, budget, estimates

- **Cache** (`llm.cache`, SQLite, default `.tatonnement/cache.sqlite`). Each call is keyed by
  the request together with its coordinates (experiment, condition, seed, session, agent,
  period, attempt). The modes are:
  - `use`: return the cached answer if there is one, otherwise call and store it (default);
  - `replay`: cached answers only; zero cost, exact reproduction;
  - `refresh`: always call and overwrite the stored answer;
  - `off`: no caching.

  To rerun an interrupted experiment at no extra cost, run it again in `use` mode.
  `tatonnement cache-export --experiment NAME --out calls.sqlite` writes one experiment's
  calls to a file that others can replay offline.
- **Budget** (`llm.budget {amount, currency}`). Paid models need both a budget and a price row
  (`llm/prices.yaml`, or your own rows under `llm.prices`); otherwise the call is refused.
  Before each call the ledger estimates its cost:
  - if spending would pass the budget, the current period is finished and the session stops
    with status `budget_stopped`;
  - if it would pass 1.2 times the budget, the session stops at once.

  The shipped DeepSeek prices are peak-hour rates as of 2026-10-21, so recorded costs are
  upper bounds.
- **Estimates** (`tatonnement estimate FILE`). This simulates a few sessions offline, builds
  every prompt exactly as a live run would, and prices the token counts, with a ±50% range.

## Outputs

Each run writes `runs/<experiment>/<time>-<config hash>/`, containing:

- `manifest.json`: the configuration, versions, git commit, status, and LLM spending with the
  price rows used;
- one folder per condition:
  - `sessions.jsonl`: one summary per session;
  - `periods_sXXXX.parquet`: one row per period and agent: proposed and posted price,
    quantity, profit, bounds, violations, parse status, tokens and cost;
  - `periods_sXXXX_texts.jsonl`: LLM answers, reasoning and memory;
  - `periods_sXXXX_events.jsonl`: stops and other events.

## Validation

- `tests/` (`pytest`) covers demand and benchmarks, the regulations, the simulation loop,
  the bit-for-bit equivalence of the two Q-learning engines, and 46 sessions of Cheng (2026)
  reproduced exactly (baseline, memoryless, asymmetric cost, three firms, a coarse grid and all
  five floors). It also runs the whole LLM pipeline with the mock client: determinism, resume
  after a crash, replay, budget stop, format retries, rejections, and announced versus hidden
  caps.
- `validation/calvano_1000.py` runs the 1,000-session Calvano baseline and compares it with
  the recorded sessions of Cheng (2026), shipped in `validation/calvano_1000_reference.csv`.
  All 1,000 sessions match exactly (steps to convergence, final state, last policy change, and
  Δ with zero difference). Mean Δ is 0.8492 (s.e. 0.0036), and the run takes one to two
  minutes on a laptop.
- Dhanda (2026) with llama3.2 3B on a local Ollama server (`dhanda_main.yaml`, `dhanda_cap.yaml`;
  12 conditions, 12 sessions each, 8,640 calls, no unusable answers). The headline result carries
  over: an announced cap becomes a focal point. Prices settle just below announced caps of
  1.70, 1.80 and 1.90 (1.67, 1.74 and 1.86, against 1.68, 1.78 and 1.87 in the paper), and hidden
  caps leave prices where they are without a cap. The unregulated level does not: with our
  reconstructed prompt, neutral agents price below Nash (Δ = −0.86, against +0.11), and
  aggressive agents go further (Δ = −1.75, against −0.49). Letting the model reason before it
  answers moves the neutral level towards Nash (Δ = −0.46) and leaves the cap result unchanged.
- Fish-style pricing on DeepSeek (`fish_style_deepseek`: deepseek-flash with thinking off, the
  `explore` prefix, 3 sessions of 300 periods; 1,800 calls, every answer usable, 19.79 CNY at
  peak-hour rates, 16 minutes). This is a demonstration, not a replication: one model and three
  sessions. The sessions part ways. In one, the two firms climb to about 5.73 and hold there
  (Δ = 0.89 over the last 50 periods; in the units the agents see, Nash is 4.71 and monopoly
  6.16). In the other two, prices settle near 4.5, a little below Nash (Δ = −0.21 and −0.28). In
  one of these the agents match each other at 4.54 and write that they are defending the
  equilibrium against a price war; in the other they keep undercutting each other by a few
  cents. Answers averaged 375 output tokens, so the default estimate (27.6 CNY at 1,000 tokens
  per answer) was an upper bound.
- Live smoke tests (`pytest -m live -s`, run by hand) cover a local Ollama server with
  llama3.2 3B (Dhanda-style, then a zero-call replay from the cache) and DeepSeek (Fish-style,
  six calls, budget 0.1 CNY).

## References

- Calvano, E., Calzolari, G., Denicolò, V., Pastorello, S. (2020). Artificial intelligence,
  algorithmic pricing, and collusion. *American Economic Review* 110(10).
- Fish, S., Gonczarowski, Y. A., Shorrer, R. I. (2024). Algorithmic collusion by large
  language models. arXiv:2404.00806.
- Dhanda, K. (2026). A price cap can act as a focal point for LLM pricing agents: evidence
  from a repeated Bertrand duopoly. Research Square, doi:10.21203/rs.3.rs-10183486/v1.
- Cheng, C. (2026). Do below-cost sales bans facilitate algorithmic collusion? Working paper.

## License

MIT. See `LICENSE`.
