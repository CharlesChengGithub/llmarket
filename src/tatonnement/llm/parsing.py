"""Parse template answers: split by field headings, read the price as the first number.

A heading is a field name at the start of a line, optionally decorated with Markdown
('**PRICE**:', '## PRICE:'), followed by a colon. When a field appears more than once (models
sometimes echo the template first) the last occurrence counts. Reasoning blocks
(<think>...</think>) are removed before parsing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


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
    if not hits:
        pattern = re.compile(rf"\b({names})[ \t*_]*:", re.I)
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
    status: str  # ok | missing_field | no_number | negative | out_of_range
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
    nums = _NUMBER.findall(section)
    if not nums:
        return ParsedPrice(None, "no_number", sections)
    value = float(nums[0] if pick == "first" else nums[-1])
    if value < 0:
        return ParsedPrice(None, "negative", sections)
    if value > upper:
        return ParsedPrice(None, "out_of_range", sections)
    return ParsedPrice(value, "ok", sections)
