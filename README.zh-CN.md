# tatonnement

面向经济学的多主体模拟工具包：大模型主体、强化学习主体和规则型主体可以在同一个市场里交易。

第一个内置场景是算法定价：企业重复进行 Bertrand 价格竞争，需求为 logit 形式（Calvano 等，2020；Fish 等，2024），价格规制可以按需开关。

[English](README.md)

状态：预发布（v0.1 开发中，尚未上传 PyPI）。

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
git clone <仓库地址>        # 尚未公开
cd tatonnement
pip install -e ".[fast,analysis]"     # numba 用于 Q-learning 快速引擎，pandas/pyarrow 用于输出
```

需要 Python 3.10 及以上。核心依赖只有 numpy、scipy、httpx 和 pyyaml。

## 快速上手

```bash
# Calvano 等（2020）基准，20 个 session，快速引擎（几秒钟）
tatonnement run src/tatonnement/experiments/calvano2020.yaml --sessions 20

# Fish 等结构的两家大模型定价主体，mock 离线回答，零费用
tatonnement run src/tatonnement/experiments/fish_style_mock.yaml

# 同一实验用 DeepSeek 要花多少钱？（离线估算，不发送任何请求）
tatonnement estimate src/tatonnement/experiments/fish_style_deepseek.yaml
```

在 Python 中调用：

```python
import tatonnement as tt

res = tt.run("src/tatonnement/experiments/fish_style_mock.yaml", out_dir="runs")
print(res["conditions"][0]["sessions"][0]["delta"])
```

## 用 YAML 文件描述实验

配置示例见英文 README 或 `src/tatonnement/experiments/` 下的文件。

`sweep` 对列出的参数取笛卡尔积，每种组合是一个条件。`tatonnement validate FILE` 只检查配置、统计条件数，不运行实验。

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
- **格式错误**：回答按模板字段切分，价格取价格字段里的第一个数字。以下情况视为格式错误：缺字段、价格不是数字、价格为负、价格高于 10 倍垄断价。出错就重问，最多 `retries` 次；仍失败时，`on_failure: stop` 结束该 session（Fish 的做法），`keep_last_price` 沿用上期价格。
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
- `validation/calvano_1000.py` 跑 Calvano 基准的 1000 个 session，与 Cheng（2026）的记录逐个比较。结果：1000 个全部逐位一致，包括收敛步数、最终状态、最后一次策略变化，Δ 的差为 0。平均 Δ = 0.8492（标准误 0.0036），笔记本上约 100 秒跑完。
- 真实接口的冒烟测试（`pytest -m live`）正在补充。

## 许可证

MIT，见 `LICENSE`。
