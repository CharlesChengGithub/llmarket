"""Long-format period records (one row per session x period x agent) and event logs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_SCALAR = (str, int, float, bool, type(None))


def _flat_meta(meta: dict) -> dict:
    """Keep scalar metadata; serialize anything else as JSON text."""
    out = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            v = v.item()
        out[f"meta_{k}"] = v if isinstance(v, _SCALAR) else json.dumps(v, ensure_ascii=False, default=str)
    return out


class Recorder:
    """Collects rows in memory.

    every: keep every k-th period (1 = all). Long Q-learning sessions use a large k or
    keep=False to avoid storing millions of rows; the last period is always kept.
    text_fields: metadata keys stored in a separate text log rather than the table
    (LLM responses, memory files).
    """

    def __init__(self, every: int = 1, keep: bool = True, text_fields: tuple[str, ...] = ("response", "memory", "prompt")):
        self.every = max(int(every), 1)
        self.keep = keep
        self.text_fields = set(text_fields)
        self.rows: list[dict] = []
        self.texts: list[dict] = []
        self.events: list[dict] = []

    def log_period(self, session, t, posted, proposed, quantity, profit, bounds, flags, fines, extra=None):
        if not self.keep or t % self.every:
            return
        for i in range(len(posted)):
            f = flags[i]
            meta = dict(f.get("meta") or {})
            texts = {k: meta.pop(k) for k in list(meta) if k in self.text_fields}
            if texts:
                self.texts.append(dict(session=session, t=t, agent=i, **texts))
            row = dict(session=session, t=t, agent=i, proposed=float(proposed[i]), posted=float(posted[i]),
                       quantity=float(quantity[i]), profit=float(profit[i]), lower=float(bounds[i].lower),
                       upper=float(bounds[i].upper), fine=float(fines[i]))
            row.update({k: v for k, v in f.items() if k.startswith("violation_") or k == "rejections"})
            row.update(_flat_meta(meta))
            self.rows.append(row)

    def event(self, **kw):
        self.events.append(kw)

    def frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def array(self, column: str, n_agents: int) -> np.ndarray:
        """Column as a (periods, agents) array for one session's records."""
        vals = np.array([r[column] for r in self.rows], dtype=float)
        return vals.reshape(-1, n_agents)

    def write(self, directory: str | Path, stem: str = "periods") -> Path:
        """Write the table as parquet (CSV without pyarrow) plus JSONL text and event logs."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd  # noqa: F401
            import pyarrow  # noqa: F401

            path = d / f"{stem}.parquet"
            self.frame().to_parquet(path, index=False)
        except ImportError:
            import csv

            path = d / f"{stem}.csv"
            keys = sorted({k for r in self.rows for k in r})
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        for name, items in (("texts", self.texts), ("events", self.events)):
            if items:
                with (d / f"{stem}_{name}.jsonl").open("w") as fh:
                    for item in items:
                        fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return path
