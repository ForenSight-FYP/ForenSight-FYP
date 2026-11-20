"""Evaluation metrics for ForenSight.

Convention: labels 0 = real/bona fide, 1 = fake/spoof; scores are fake scores
(higher = more likely fake).

- AUC: standard for video deepfake detection.
- EER: standard for speech anti-spoofing (ASVspoof). Reported as a fraction (0.05 = 5%).
- Threshold metrics (accuracy, precision, recall, F1, FPR, FNR) at a chosen threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _check(labels, scores):
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} differ in shape")
    if not np.isfinite(scores).all():
        raise ValueError(f"{int((~np.isfinite(scores)).sum())} of {scores.size} scores are NaN/inf")
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("labels must be 0 (real) or 1 (fake)")
    if len(np.unique(labels)) < 2:
        raise ValueError("need both real and fake samples to compute ranking metrics")
    return labels, scores


def auc(labels, scores) -> float:
    labels, scores = _check(labels, scores)
    return float(roc_auc_score(labels, scores))


def eer(labels, scores) -> tuple[float, float]:
    """Equal error rate and the threshold where FPR == FNR.

    Uses linear interpolation between ROC points so the value matches the usual
    ASVspoof-style computation closely.
    """
    labels, scores = _check(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    # first index where FPR >= FNR
    idx = int(np.argmax(diff >= 0))
    if idx == 0:
        return float(fpr[0]), float(thr[0])
    # interpolate between idx-1 and idx
    d0, d1 = diff[idx - 1], diff[idx]
    w = d0 / (d0 - d1) if d0 != d1 else 0.0
    eer_value = fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1])
    thr_lo, thr_hi = thr[idx - 1], thr[idx]
    if not np.isfinite(thr_lo):  # roc_curve's first threshold can be inf
        thr_lo = thr_hi
    threshold = thr_lo + w * (thr_hi - thr_lo)
    return float(eer_value), float(threshold)


@dataclass
class ThresholdMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)


def threshold_metrics(labels, scores, threshold: float = 0.5) -> ThresholdMetrics:
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    def safe(n, d):
        return float(n / d) if d else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=safe(tp + tn, len(labels)),
        precision=precision,
        recall=recall,
        f1=safe(2 * precision * recall, precision + recall),
        fpr=safe(fp, fp + tn),
        fnr=safe(fn, fn + tp),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def full_report(labels, scores, threshold: float | None = None) -> dict:
    """All metrics the proposal commits to. If threshold is None, the EER threshold is used."""
    e, e_thr = eer(labels, scores)
    thr = e_thr if threshold is None else threshold
    report = {"auc": auc(labels, scores), "eer": e, "eer_threshold": e_thr}
    report.update(threshold_metrics(labels, scores, thr).as_dict())
    return report
