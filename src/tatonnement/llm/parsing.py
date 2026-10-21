"""Parse template answers: split by field headings, read the price from the price section.

A heading is a field name at the start of a line (any case), optionally decorated with Markdown
('**PRICE**:', '## PRICE:'), followed by a colon; when no line starts with a heading, headings
are searched inside lines, in the exact case of the template. When a field appears more than once (models
sometimes echo the template first) the last occurrence counts. Reasoning blocks
(<think>...</think>) are removed before parsing.

Which number is the price: pick='first' (the first number in the section), 'last', or 'only'
(the section must be just a number, with at most a 'Price:' label, a '$' sign, Markdown emphasis
or a final full stop around it; anything else is the status 'extra_text'). Use 'only' for
single-number prompts: small models often reason aloud, and the first or last number of such a
reply is frequently the cost, a cap or a rival's price rather than the decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_ONLY = re.compile(r"^[\s*_`\"']*(?:price[\s*_]*[:=]?)?[\s*_`\"']*\$?\s*(-?\d+(?:\.\d+)?)[\s*_`\"'.]*$", re.I)
PICKS = ("first", "last", "only")


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    if "</think>" in text.lower():  # an opening tag was cut off by the provider
        text = text[text.lower().rfind("</think>") + len("</think>"):]
    return text


def split_sections(text: str, fields: Sequence[str]) -> dict[str, str]:
    if not fields:
        return {}
    names = "|".join(re.escape(f) for f in fields)
    pattern = re.compile(rf"^[ \t>#*_\-]*({names})[ \t*_]*:", re.I | re.M)
    hits = list(pattern.finditer(text))
    if not hits:  # headings run together on one line; exact case only, so prose ("the rival's price: ...") never matches
        pattern = re.compile(rf"\b({names})[ \t*_]*:")
        hits = list(pattern.finditer(text))
    out: dict[str, str] = {}
    canon = {f.lower(): f for f in fields}
    for j, m in enumerate(hits):
        end = hits[j + 1].start() if j + 1 < len(hits) else len(text)
        out[canon[m.group(1).lower()]] = text[m.end():end].strip().strip("*_ ").strip()
    return out


@dataclass
class ParsedPrice:
    value: float | None
    status: str  # ok | missing_field | no_number | extra_text | negative | out_of_range
    sections: dict


def parse_price(text: str, fields: Sequence[str], price_field: str | None, upper: float,
                pick: str = "first") -> ParsedPrice:
    """Price from an answer. With price_field None the whole answer is the price section."""
    clean = strip_reasoning(text)
    sections = split_sections(clean, fields) if price_field else {}
    if price_field:
        if price_field not in sections:
            return ParsedPrice(None, "missing_field", sections)
        section = sections[price_field]
    else:
        section = clean
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
    nums = _NUMBER.findall(section)
    if not nums:
        return ParsedPrice(None, "no_number", sections)
    if pick == "only":
        m = _ONLY.match(section)
        if m is None:
            return ParsedPrice(None, "extra_text", sections)
        value = float(m.group(1))
    else:
        value = float(nums[0] if pick == "first" else nums[-1])
    if value < 0:
        return ParsedPrice(None, "negative", sections)
    if value > upper:
        return ParsedPrice(None, "out_of_range", sections)
    return ParsedPrice(value, "ok", sections)
