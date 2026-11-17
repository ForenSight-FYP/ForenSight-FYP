"""Identity-disjoint train/val/test splits.

Deepfake datasets reuse the same person in many clips. If one person appears in
both training and test data, the model can memorise faces or voices and the test
score is inflated. Splitting by group (identity / speaker / source video pair)
prevents that leakage.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Sequence, TypeVar

T = TypeVar("T")


def group_split(
    items: Sequence[T],
    groups: Sequence[Hashable],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Split items so that every group lands in exactly one of train/val/test."""
    if len(items) != len(groups):
        raise ValueError("items and groups must have the same length")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    by_group: dict[Hashable, list[T]] = defaultdict(list)
    for item, g in zip(items, groups):
        by_group[g].append(item)

    keys = sorted(by_group, key=str)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    parts = (keys[:n_train], keys[n_train:n_train + n_val], keys[n_train + n_val:])
    return tuple([x for k in part for x in by_group[k]] for part in parts)  # type: ignore[return-value]
