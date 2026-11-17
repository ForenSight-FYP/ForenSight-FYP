import numpy as np
import pytest

from forensight.fusion.late_fusion import LateFusion


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)

    def module(strength):
        # informative but noisy fake scores in (0, 1)
        z = strength * (2 * y - 1) + rng.normal(0, 1, n)
        return 1 / (1 + np.exp(-z))

    video, audio, av = module(1.0), module(1.0), module(0.8)
    audio[: n // 4] = np.nan  # some samples have no audio (e.g. silent clips)
    return {"video": video, "audio": audio, "av": av}, y


def test_fused_beats_single_module():
    from forensight.evaluation.metrics import auc

    s, y = _synthetic()
    full = ~np.isnan(s["audio"])
    fusion = LateFusion().fit(s, y)
    fused = fusion.predict({k: v[full] for k, v in s.items()})
    assert auc(y[full], fused) > max(auc(y[full], s[m][full]) for m in s) + 0.02


def test_missing_modality_uses_subset_model():
    s, y = _synthetic()
    fusion = LateFusion().fit(s, y)
    out = fusion.predict_one({"video": 0.9, "audio": None, "av": 0.8})
    assert out["modalities_used"] == ["video", "av"]
    assert 0.0 <= out["probability"] <= 1.0
    assert set(out["contributions"]) == {"video", "av"}


def test_audio_only_input():
    s, y = _synthetic()
    fusion = LateFusion().fit(s, y)
    hi = fusion.predict_one({"audio": 0.95})["probability"]
    lo = fusion.predict_one({"audio": 0.05})["probability"]
    assert hi > 0.5 > lo


def test_save_load_roundtrip(tmp_path):
    s, y = _synthetic()
    fusion = LateFusion().fit(s, y)
    p = tmp_path / "fusion.json"
    fusion.save(p)
    loaded = LateFusion.load(p)
    q = {"video": 0.7, "audio": 0.6, "av": 0.4}
    assert loaded.predict_one(q)["probability"] == pytest.approx(fusion.predict_one(q)["probability"])


def test_no_scores_rejected():
    s, y = _synthetic()
    with pytest.raises(ValueError):
        LateFusion().fit(s, y).predict_one({"video": None})
