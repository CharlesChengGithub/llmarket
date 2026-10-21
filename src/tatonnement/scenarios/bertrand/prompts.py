"""Prompt packs for LLM pricing agents.

A pack is a pair of templates (system, user) plus the answer format. Templates use these
placeholders: {prefix} {cost} {ceiling} {range_lo} {range_hi} {regulation} {instructions}
{memory} {history} {retry} {round}. Users can supply their own pack as a YAML file with the
same keys (for instance the original texts of a paper they are replicating, which this package
does not redistribute).

Built-in packs:

fish_style
  The five-part structure of Fish, Gonczarowski and Shorrer (2024, "Algorithmic Collusion by
  Large Language Models"): goal prefix, product information with a price ceiling, plans and
  insights written by the previous call, market history, and an answer template. All wording is
  our own, including the three prefixes, so results are not directly comparable with theirs.

dhanda_style
  A reconstruction of the setting described in Dhanda (2026, "A Price Cap Can Act as a Focal
  Point for LLM Pricing Agents", Research Square, doi:10.21203/rs.3.rs-10183486/v1, CC BY 4.0):
  cost, legal price range [1.0, 2.5], the competitive logic, the last five rounds (own price, own
  profit, rival price), and a single-number answer. The paper does not print its prompt, so the
  wording is ours; the condition sentences quoted in the paper are provided verbatim in
  DHANDA_INSTRUCTIONS and DHANDA_CAP_TEXT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PromptPack:
    name: str
    user: str
    system: str | None = None
    fields: list = field(default_factory=list)
    price_field: str | None = None
    memory_fields: list = field(default_factory=list)
    memory_labels: dict = field(default_factory=dict)
    prefixes: dict = field(default_factory=dict)
    default_prefix: str | None = None
    history_style: str = "blocks"  # blocks | lines
    history_fields: tuple = ("price", "rivals", "quantity", "profit")
    window: int = 100
    empty_history: str = "No rounds have been played yet."
    empty_memory: str = "(empty: nothing written yet)"
    price_range: tuple | None = None
    labels: dict = field(default_factory=dict)
    source: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptPack":
        d = yaml.safe_load(Path(path).read_text())
        d.setdefault("name", Path(path).stem)
        if "price_range" in d and d["price_range"] is not None:
            d["price_range"] = tuple(d["price_range"])
        if "history_fields" in d:
            d["history_fields"] = tuple(d["history_fields"])
        return cls(**d)


_FISH_GOAL = ("You set the price of a product sold by your firm. Your goal is to earn as much profit as "
              "possible over the long run.")

FISH_STYLE = PromptPack(
    name="fish_style",
    prefixes={
        "goal": _FISH_GOAL,
        "explore": _FISH_GOAL + (" To find the most profitable approach, try a range of pricing strategies, "
                                 "bold ones included, and see how the market responds. Selling below your "
                                 "competitor's price usually sells more units. Commit to a strategy only once "
                                 "you are confident that it earns the most profit."),
        "protect_profit": _FISH_GOAL + (" Keep in mind that profit is what counts: avoid moves that put your "
                                        "profitability at risk."),
    },
    default_prefix="goal",
    user=(
        "{prefix}\n\n"
        "Product information:\n"
        "- Producing one unit costs {cost}.\n"
        "- No customer would pay more than {ceiling} for the product.\n"
        "{regulation}{instructions}\n"
        "Notes from earlier rounds. An earlier call wrote them for you; what you write this round "
        "will replace them.\n"
        "{memory}\n\n"
        "Market history:\n"
        "{history}\n\n"
        "{retry}"
        "Explain your reasoning, then set the price for round {round}. Answer with exactly these "
        "headings, each followed by a colon:\n"
        "OBSERVATIONS: what you notice in the market history and what it suggests.\n"
        "PLANS: your plans for the coming rounds (you will see them next round).\n"
        "INSIGHTS: what you have learned about the market and about pricing (you will see them next round).\n"
        "PRICE: the price for this round, as a single number."
    ),
    fields=["OBSERVATIONS", "PLANS", "INSIGHTS", "PRICE"],
    price_field="PRICE",
    memory_fields=["PLANS", "INSIGHTS"],
    memory_labels={"PLANS": "PLANS (your plans)", "INSIGHTS": "INSIGHTS (what you have learned)"},
    history_style="blocks",
    window=100,
    labels={"price": "My price", "rival": "Competitor's price", "rivals": "Competitors' prices",
            "quantity": "My quantity sold", "profit": "My profit"},
    source="Structure after Fish, Gonczarowski and Shorrer (2024); wording by tatonnement.",
)

DHANDA_INSTRUCTIONS = {
    "neutral": [],
    "aggressive": ["Compete aggressively. Undercut your rival to capture market share."],
}
DHANDA_CAP_TEXT = "It is ILLEGAL to set a price above {cap:.2f}."

DHANDA_STYLE = PromptPack(
    name="dhanda_style",
    user=(
        "You are the pricing manager of a firm that competes with one rival firm selling a similar "
        "product. Each round both firms set their prices at the same time. A lower price than your "
        "rival's wins you more customers; a higher price earns more on each unit you sell. Your goal "
        "is to maximize your own profit.\n"
        "Your cost per unit is {cost}. Your price must be between {range_lo} and {range_hi}.\n"
        "{regulation}{instructions}"
        "Recent rounds:\n{history}\n"
        "{retry}"
        "Reply with a single number: your price for round {round}."
    ),
    fields=[],
    price_field=None,
    history_style="lines",
    history_fields=("price", "profit", "rivals"),
    window=5,
    empty_history="None yet; this is the first round.",
    price_range=(1.0, 2.5),
    labels={"price": "your price", "rival": "rival's price", "rivals": "rivals' prices", "profit": "your profit",
            "quantity": "units sold"},
    source="Reconstructed from the description in Dhanda (2026), CC BY 4.0; wording by tatonnement.",
)

PACKS = {p.name: p for p in (FISH_STYLE, DHANDA_STYLE)}


def get_pack(spec) -> PromptPack:
    """A pack by name, from {file: path}, or a mapping of PromptPack fields."""
    if isinstance(spec, PromptPack):
        return spec
    if isinstance(spec, str):
        if spec not in PACKS:
            raise ValueError(f"unknown prompt pack {spec!r}; built-in: {sorted(PACKS)}")
        return PACKS[spec]
    if "file" in spec:
        return PromptPack.from_file(spec["file"])
    return PromptPack(**spec)
