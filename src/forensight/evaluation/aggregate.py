"""Turn frame-level or segment-level scores into one score per video / audio file.

The per-frame scores are also kept: they become the anomaly timeline in the
explainable report.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable

import numpy as np


def aggregate(keys: Iterable[Hashable], scores: Iterable[float], labels: Iterable[int],
              how: str = "mean") -> tuple[list[Hashable], np.ndarray, np.ndarray]:
    """Group scores by key (e.g. video id). Returns (keys, aggregated scores, labels).

    how: "mean" (default, robust) or "max" (sensitive to short manipulated segments)
         or "top_mean" (mean of the top 25%: a middle ground).
    """
    s_by, l_by = defaultdict(list), {}
    for k, s, y in zip(keys, scores, labels):
        s_by[k].append(float(s))
        if k in l_by and l_by[k] != int(y):
            raise ValueError(f"Inconsistent labels for {k!r}")
        l_by[k] = int(y)
    out_keys = sorted(s_by, key=str)
    agg = []
    for k in out_keys:
        v = np.sort(np.asarray(s_by[k]))
        if how == "mean":
            agg.append(v.mean())
        elif how == "max":
            agg.append(v.max())
        elif how == "top_mean":
            agg.append(v[-max(1, len(v) // 4):].mean())
        else:
            raise ValueError(f"unknown aggregation {how!r}")
    return out_keys, np.asarray(agg), np.asarray([l_by[k] for k in out_keys])
