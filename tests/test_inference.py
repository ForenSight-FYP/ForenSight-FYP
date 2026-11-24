"""Tests for the analyse function (src/forensight/inference.py). No downloads, no GPU:
a randomly initialised small backbone stands in for the trained EfficientNet-B4."""

import numpy as np
import pytest

# CI installs only the light dependencies (no PyTorch): skip these tests there, like
# test_models.py does. They run in Colab, where the full [dl] extras are installed.
torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from forensight.inference import AUDIO_THRESHOLD, VIDEO_THRESHOLD, Analyser, AnalysisResult, prepare_face  # noqa: E402


def test_prepare_face_matches_training_preprocessing():
    face = np.full((100, 120, 3), (0, 128, 255), dtype=np.uint8)  # BGR
    x = prepare_face(face, image_size=64)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    # RGB order after conversion: R=255, G=128, B=0, then ImageNet normalisation
    assert x[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)
    assert x[2, 0, 0] == pytest.approx((0.0 - 0.406) / 0.225, abs=1e-4)


def test_verdict_rules():
    assert AnalysisResult("a.mp4").verdict == "UNKNOWN"
    assert AnalysisResult("a.mp4", video_score=VIDEO_THRESHOLD + 0.01).verdict == "FAKE"
    assert AnalysisResult("a.mp4", video_score=0.01, audio_score=0.01).verdict == "REAL"
    assert AnalysisResult("a.mp4", video_score=0.01, audio_score=AUDIO_THRESHOLD).verdict == "FAKE"
    assert "-> FAKE" in AnalysisResult("a.mp4", video_score=0.9).summary()


@pytest.fixture
def faceless_video(tmp_path):
    import cv2

    path = tmp_path / "noise.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
    rng = np.random.default_rng(0)
    for _ in range(20):
        out.write(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    out.release()
    return path


@pytest.fixture
def small_checkpoint(tmp_path):
    from forensight.models.video import FrameDetector

    path = tmp_path / "best.pt"
    torch.save(FrameDetector("efficientnet_b0", pretrained=False).state_dict(), path)
    return path


class _StubCropper:
    """Stands in for FaceCropper (which downloads a detector model): returns a fixed
    crop, or None to simulate frames with no face."""

    def __init__(self, face: bool):
        self.face = face

    def crop(self, frame, size=380):
        return np.zeros((size, size, 3), dtype=np.uint8) if self.face else None


def _analyser(ckpt, face: bool):
    return Analyser(ckpt, backbone="efficientnet_b0", image_size=64, num_frames=4,
                    use_audio=False, device="cpu", cropper=_StubCropper(face))


def test_video_with_faces_gives_a_score(faceless_video, small_checkpoint):
    r = _analyser(small_checkpoint, face=True).analyse(faceless_video)
    assert r.frames_sampled == 4 and r.faces_found == 4
    assert 0.0 <= r.video_score <= 1.0 and len(r.frame_scores) == 4
    assert r.verdict in ("REAL", "FAKE")


def test_video_without_face_is_reported_not_crashed(faceless_video, small_checkpoint):
    r = _analyser(small_checkpoint, face=False).analyse(faceless_video)
    assert r.frames_sampled == 4 and r.faces_found == 0
    assert r.video_score is None and r.verdict == "UNKNOWN"
    assert any("no face" in n for n in r.notes)


def test_unsupported_and_missing_files(tmp_path):
    a = Analyser(None, use_audio=False, device="cpu")
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert "unsupported" in a.analyse(txt).notes[0]
    with pytest.raises(FileNotFoundError):
        a.analyse(tmp_path / "missing.mp4")
