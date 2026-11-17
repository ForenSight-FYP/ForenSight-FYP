import numpy as np
import pytest

from forensight.evaluation import metrics


def test_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert metrics.auc(y, s) == 1.0
    e, thr = metrics.eer(y, s)
    assert e == pytest.approx(0.0, abs=1e-9)
    assert 0.3 <= thr <= 0.7


def test_random_scores_eer_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    s = rng.random(20_000)
    e, _ = metrics.eer(y, s)
    assert e == pytest.approx(0.5, abs=0.02)
    assert metrics.auc(y, s) == pytest.approx(0.5, abs=0.02)


def test_eer_matches_bruteforce_on_gaussians():
    rng = np.random.default_rng(1)
    real = rng.normal(0, 1, 5000)
    fake = rng.normal(2, 1, 5000)
    y = np.r_[np.zeros(5000), np.ones(5000)]
    s = np.r_[real, fake]
    e, _ = metrics.eer(y, s)
    # brute force: sweep thresholds, take point where FPR and FNR are closest
    thr = np.sort(s)
    fpr = np.array([(real >= t).mean() for t in thr])
    fnr = np.array([(fake < t).mean() for t in thr])
    i = np.argmin(np.abs(fpr - fnr))
    assert e == pytest.approx((fpr[i] + fnr[i]) / 2, abs=0.003)
    # theory for two unit Gaussians 2 apart: EER = Phi(-1) ~ 0.1587
    assert e == pytest.approx(0.1587, abs=0.01)


def test_threshold_metrics_counts():
    y = [0, 0, 1, 1]
    s = [0.2, 0.6, 0.4, 0.9]
    m = metrics.threshold_metrics(y, s, 0.5)
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5 and m.f1 == 0.5
    assert m.fpr == 0.5 and m.fnr == 0.5 and m.accuracy == 0.5


def test_full_report_keys():
    r = metrics.full_report([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    for k in ("auc", "eer", "precision", "recall", "f1", "fpr", "fnr"):
        assert k in r


def test_single_class_rejected():
    with pytest.raises(ValueError):
        metrics.auc([1, 1, 1], [0.1, 0.2, 0.3])
