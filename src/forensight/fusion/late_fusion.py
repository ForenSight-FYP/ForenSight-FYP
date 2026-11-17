"""Late fusion of module scores into one calibrated fake probability.

Each module (video, audio, av) outputs a fake score in [0, 1]. Some inputs lack a
modality (e.g. an audio-only file has no video or AV score). We therefore fit one
logistic-regression model per *subset* of modalities seen in training, and at
inference pick the model matching the modalities that are present.

Logistic regression is deliberately simple: its coefficients show how much each
module contributed to the final decision, which feeds the explainable report.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

MODALITIES = ("video", "audio", "av")
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class _SubsetModel:
    modalities: tuple[str, ...]
    coef: np.ndarray
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = _logit(x) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LateFusion:
    """Fit on a table of module scores (NaN = modality missing) and labels."""

    min_samples: int = 20
    C: float = 1.0
    models: dict[tuple[str, ...], _SubsetModel] = field(default_factory=dict)

    def fit(self, scores: Mapping[str, np.ndarray], labels: np.ndarray) -> LateFusion:
        labels = np.asarray(labels).astype(int)
        cols = {m: np.asarray(scores[m], dtype=float) for m in MODALITIES if m in scores}
        self.models.clear()
        for r in range(1, len(cols) + 1):
            for subset in combinations(sorted(cols, key=MODALITIES.index), r):
                # rows where every modality in the subset is available
                mask = np.all([~np.isnan(cols[m]) for m in subset], axis=0)
                y = labels[mask]
                if mask.sum() < self.min_samples or len(np.unique(y)) < 2:
                    continue
                x = np.stack([cols[m][mask] for m in subset], axis=1)
                clf = LogisticRegression(C=self.C).fit(_logit(x), y)
                self.models[subset] = _SubsetModel(subset, clf.coef_[0].copy(), float(clf.intercept_[0]))
        if not self.models:
            raise ValueError("No modality subset had enough labelled samples of both classes to fit.")
        return self

    def predict_one(self, scores: Mapping[str, float | None]) -> dict:
        """Fuse one sample. Returns the probability plus per-module contributions."""
        present = tuple(m for m in MODALITIES if scores.get(m) is not None and not np.isnan(scores[m]))
        if not present:
            raise ValueError("At least one module score is required.")
        model = self.models.get(present)
        if model is None:
            # fall back to the largest fitted subset contained in what we have
            candidates = [s for s in self.models if set(s) <= set(present)]
            if not candidates:
                raise ValueError(f"No fusion model available for modalities {present}")
            model = self.models[max(candidates, key=len)]
        x = np.array([[scores[m] for m in model.modalities]], dtype=float)
        prob = float(model.predict(x)[0])
        contrib = dict(zip(model.modalities, (_logit(x)[0] * model.coef).tolist()))
        return {"probability": prob, "modalities_used": list(model.modalities), "contributions": contrib}

    def predict(self, scores: Mapping[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(scores.values())))
        return np.array([
            self.predict_one({m: float(scores[m][i]) for m in scores})["probability"] for i in range(n)
        ])

    # ---- persistence (plain JSON so it is easy to inspect) ----
    def save(self, path: str | Path) -> None:
        data = [
            {"modalities": list(m.modalities), "coef": m.coef.tolist(), "intercept": m.intercept}
            for m in self.models.values()
        ]
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LateFusion:
        obj = cls()
        for d in json.loads(Path(path).read_text()):
            key = tuple(d["modalities"])
            obj.models[key] = _SubsetModel(key, np.array(d["coef"]), float(d["intercept"]))
        return obj
