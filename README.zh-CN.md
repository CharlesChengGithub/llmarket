# tatonnement

面向经济学的多主体模拟工具包：大模型主体、强化学习主体和规则型主体可以在同一个市场里交易。

第一个内置场景是算法定价：企业重复进行 Bertrand 价格竞争，需求为 logit 形式（Calvano 等，2020；Fish 等，2024），价格规制可以按需开关。

[English](README.md)

状态：v0.1，早期版本。从 GitHub 安装，尚未上传 PyPI。

## 为什么做这个

研究大模型定价主体的论文增长很快，但多数没有公开代码，公开的也大多是为单篇论文写的一次性脚本。tatonnement 想成为这一领域共用的实验平台，提供以下功能。

- **精确的 Q-learning 基准。** Q-learning 引擎按 session 逐位复现 Calvano 等（2020）的基准实验，并以 Cheng（2026）记录的 1000 个 session 作为验收标准。
- **各类主体同场竞争。** 大模型、Q-learning 和规则型主体可以同场竞争。Q-learning 主体自我对弈收敛后，可以冻结，也可以继续学习；一个市场里也可以混用不同服务商的模型。
- **规制模块可插拔。**
  - 底价有五种成本口径：常数、边际成本，以及三种完全成本口径。
  - 也可以设价格上限。
  - 执行方式单独设定：截断、拒绝后重报、罚款、只记录。
  - 是否告知主体也单独设定：告知，或不告知。
- **大模型主体的部署方式可选。** 历史可以给原始记录或摘要；历史窗口长度、对手价格是否可见、每隔几期决策一次，都可以设定。
- **大模型实验可复现、费用可控。**
  - 回答缓存按"提示词 + 调用在实验中的位置"建键。中断后重跑已完成的调用不再花钱；别人拿到导出的缓存文件，可以离线重放。
  - 服务商支持时，每次调用都使用确定的 seed。
  - 费用台账带预算闸门，花费不会超出预算；离线估算命令可以在开跑前算出大致费用。
  - 离线 mock 客户端不联网，可以搭建和测试实验。

## 安装

```bash
git clone https://github.com/CharlesChengGithub/tatonnement.git
cd tatonnement
pip install -e ".[analysis]"     # pandas/pyarrow 用于输出 parquet（不装就输出 CSV）
```

需要 Python 3.10 及以上。核心依赖为 numpy、scipy、numba、httpx 和 pyyaml。

## 快速上手

```bash
tatonnement list                                  # 列出随包提供的实验

# Calvano 等（2020）基准，20 个 session，快速引擎（几秒钟）
tatonnement run calvano2020 --sessions 20

# Fish 等结构的两家大模型定价主体，mock 离线回答，零费用
tatonnement run fish_style_mock

# 同一实验用 DeepSeek 要花多少钱？（离线估算，不发送任何请求）
tatonnement estimate fish_style_deepseek
```

`CONFIG` 可以是随包实验的名字，也可以是自己的 YAML 文件路径。写自己的实验时，可以从 `src/tatonnement/experiments/` 里复制一个改。

Dhanda（2026）的复现用本地模型，免费：

```bash
ollama pull llama3.2:3b        # 先从 https://ollama.com 安装 Ollama；模型约 2 GB
ollama serve                   # 另开一个终端运行；已打开 Ollama App 时不需要
tatonnement run dhanda_main    # 表 1：2 个条件 × 12 个 session × 30 期
tatonnement run dhanda_cap     # 表 2：10 个条件
```

在 Python 中调用：

```python
import tatonnement as tt

res = tt.run("fish_style_mock", out_dir="runs")
print(res["conditions"][0]["sessions"][0]["delta"])
```

## 用 YAML 文件描述实验

配置示例见英文 README 或 `src/tatonnement/experiments/` 下的文件。

`sweep` 对列出的参数取笛卡尔积，每种组合是一个条件。设计不是完整笛卡尔积时（例如"中性、激进、中性加上限"三个条件），用 `conditions` 逐个列出有名字的条件，每个条件是一组按点号路径的改动；`conditions` 和 `sweep` 同时出现时，两者交叉。`tatonnement validate FILE` 只检查配置、统计条件数，不运行实验。

- **市场**（`market`）
  - logit 需求参数：`n`、`a`、`a0`、`mu`、`c`。
  - 每期固定成本 `F`，供完全成本底价使用。
  - 显示单位 `units {alpha, beta}`：大模型看到的价格乘以 α、数量乘以 β，与 Fish 等（2024）一致。Q-learning、指标和输出一律用模型单位。
- **主体**（`agents`）
  - `rule`：规则型，包括 fixed、nash、monopoly、random、tit_for_tat、grim_trigger、undercut。
  - `qlearning`：表格型 Q-learning；`pretrain` 设预训练，自我对弈收敛后冻结或继续学习。
  - `llm`：大模型定价主体。
  - `{copy_of: j}`：按第 j 个主体的设定新建一个主体。

  全是 Q-learning 时自动走 Numba 内核，否则走逐期引擎。同一种子下，两种引擎的动作序列和 Q 表经测试完全相同。混合市场里，Q-learning 的状态是上期各家价格映射到它的网格上。
- **规制**（`regulations`）
  - `price_floor` 的 `kind` 取以下之一：
    - `constant`：常数底价，配合 `value`；
    - `own_variable`：不得低于边际成本；
    - `b1`、`b2`、`lambda`：完全成本底价 c + F / max(q_{t−1}, λ·q_N)，λ 分别为 0.1、1 和 `lam`。
  - `price_cap`：价格上限。
  - 每条规制都设 `enforcement`（clip、reject、penalty、monitor）、`disclosure`（announced、hidden），以及告知时给大模型看的 `text`。
  - 底价高于上限时按底价执行，记录里会标出冲突。

## 大模型主体

- **服务商**：deepseek、qwen（DashScope 兼容模式）、ollama（本地）、openai、openrouter、siliconflow，也可以接自定义的 OpenAI 兼容接口。密钥只从环境变量读取（例如 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`）。配置文件里出现密钥会直接报错；缓存、日志和运行清单里也都不会出现密钥。
  - DeepSeek 默认开启思考模式。这时服务商会忽略 `temperature`，隐藏的推理也计入 `max_tokens`，`max_tokens` 设小了，可见回答可能是空的。要和非推理模型的研究（例如 Fish 等，2024）比较，应关闭思考模式并在文中说明：`extra_body: {thinking: {type: disabled}}`。
- **提示词包**
  - `fish_style`：沿用 Fish、Gonczarowski 和 Shorrer（2024）的五段结构：目标前缀、产品信息（价格天花板 = U[1.5, 2.5] × 垄断价，每个 session 抽一次）、上一期写下的计划和心得、最近 100 期市场记录、回答模板。措辞是本包自己写的，三个前缀 goal、explore、protect_profit 也不是原文，所以结果不能和原论文直接比较。要用原文，请自己把原文写成 YAML 提示词包，通过 `prompt: {file: ...}` 加载；本包不分发原文。
  - `dhanda_style`：按 Dhanda（2026）的描述重建：成本、合法价格区间 [1.0, 2.5]、竞争逻辑、最近 5 期记录、只回答一个数字。原文没有给出提示词，措辞是本包写的；论文引用的条件句原样收在 `prompts` 模块里（CC BY 4.0）。
- **部署方式**（`deployment`）：
  - `history`：原始记录，或摘要（按本方价格水平分组，给出次数、平均销量、平均利润和对手平均价格）；
  - `window`：历史窗口长度；
  - `rival_visible`：对手价格是否可见；
  - `interval`：每隔几期决策一次；
  - `decimals`：显示位数；
  - `show`：显示哪些字段；
  - `order`：先列最早还是最近的记录；
  - `own_price`：本方价格显示实际执行价还是提出价。
- **格式错误**：回答按模板字段切分，再从价格字段里取价格。取哪个数字由 `pick` 决定（在提示词包里设，也可以对单个主体设）：`first` 取第一个数字（`fish_style` 的默认）；`last` 取最后一个；`only` 要求回答只有一个数字，最多带 `Price:`、`$` 或句末句号（`dhanda_style` 的默认）。小模型常常先写一段推理再给价格，这种回答的第一个或最后一个数字往往是成本、上限或对手价格，不是它定的价，所以严格模式宁可重问，也不猜。以下情况视为格式错误：缺字段；价格不是数字，或（`only` 模式下）数字前后还有别的文字；价格为负；价格高于 10 倍垄断价。出错就重问，最多 `retries` 次；仍失败时，`on_failure: stop` 结束该 session（Fish 的做法），`keep_last_price` 沿用上期价格。
- **缓存**：SQLite 文件，默认 `.tatonnement/cache.sqlite`。键为"请求 + 调用坐标"，坐标包括实验、条件、种子、session、主体、期和第几次尝试。四种模式：
  - `use`（默认）：有缓存就读，没有就调用并写入；
  - `replay`：只读缓存，零费用、精确复现；
  - `refresh`：重新调用并覆盖；
  - `off`：不用缓存。

  中断后用 `use` 模式重跑，已完成的部分不再花钱。`tatonnement cache-export` 把某个实验用到的调用导出成单独文件，可以随复现包一起发布。
- **预算**：付费模型必须同时设预算和价目行，否则拒绝调用。每次调用前先估算费用：
  - 会超过预算时，跑完当期就停，状态记为 `budget_stopped`；
  - 会超过预算的 1.2 倍时，立即停止。

  内置的 DeepSeek 价格是 2026-10-21 的忙时价，所以记录的费用是上限。
- **估算**：`tatonnement estimate FILE` 离线模拟几个 session，按真实运行的方式组装每一期的提示词，统计 token 数并给出费用，区间为 ±50%。

## 输出

每次运行写到 `runs/<实验名>/<时间>-<配置哈希>/` 下，内容包括：

- `manifest.json`：配置、软件版本、git 提交、运行状态，以及大模型花费和所用的价目行；
- 每个条件一个文件夹，里面有：
  - `sessions.jsonl`：每个 session 的汇总指标；
  - 每期每个主体一行的长表：提出价、执行价、销量、利润、上下限、违规标志、解析状态、token 数和费用；
  - 大模型回答原文和记忆的文本记录；
  - 事件记录。

## 验收

- `pytest` 覆盖以下内容：
  - 需求函数和基准价格、各类规制、模拟循环；
  - 两种 Q-learning 引擎逐位等价；
  - 逐位复现 Cheng（2026）的 46 个 session：基准、无记忆、成本不对称、三家、粗网格和五种底价；
  - 用 mock 跑通整条大模型管线：确定性、崩溃后续跑、重放、预算停止、格式重试、拒绝后重报、告知与不告知的上限。
- `validation/calvano_1000.py` 跑 Calvano 基准的 1000 个 session，与 Cheng（2026）的记录逐个比较（记录随仓库提供：`validation/calvano_1000_reference.csv`）。结果：1000 个全部逐位一致，包括收敛步数、最终状态、最后一次策略变化，Δ 的差为 0。平均 Δ = 0.8492（标准误 0.0036），笔记本上一两分钟跑完。
- 用本地 Ollama 的 llama3.2 3B 复现 Dhanda（2026）（`dhanda_main.yaml`、`dhanda_cap.yaml`；12 个条件，各 12 个 session，8,640 次调用，没有不合格的回答）。主要结论复现了：告知的上限成为聚点。告知上限为 1.70、1.80、1.90 时，价格停在上限下方，分别为 1.67、1.74、1.86（论文为 1.68、1.78、1.87）；不告知的上限不改变价格。无规制时的水平没有复现：用我们重建的提示词，中性主体的价格低于 Nash（Δ = −0.86，论文 +0.11），激进主体更低（Δ = −1.75，论文 −0.49）。允许模型先推理再报价，中性水平向 Nash 靠近（Δ = −0.46），上限的结果不变。
- DeepSeek 上的 Fish 结构示例（`fish_style_deepseek`：deepseek-flash，关闭思考模式，`explore` 前缀，3 个 session × 300 期；1,800 次调用，回答全部可用，按忙时价花费 19.79 元，用时 16 分钟）。这只是演示，不是复现：只有一个模型、三个 session。三个 session 走向不同。一个 session 里两家把价格抬到 5.73 左右并一直保持（最后 50 期 Δ = 0.89；按主体看到的单位，Nash 价格为 4.71，垄断价格为 6.16）。另外两个停在 4.5 附近，略低于 Nash（Δ = −0.21 和 −0.28）：其中一个两家都定 4.54，回答里写着要守住这个均衡、避免价格战；另一个两家一直相互压价几分钱。回答平均 375 个输出 token，所以默认估算（按每次 1,000 token 算，27.6 元）是上限。
- 真实接口的冒烟测试（`pytest -m live -s`，手动运行）：本地 Ollama 的 llama3.2 3B（Dhanda 结构，再从缓存零调用重放），以及 DeepSeek（Fish 结构，6 次调用，预算 0.1 元）。

## 许可证

MIT，见 `LICENSE`。
