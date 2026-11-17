"""Shape checks for the deep-learning models. Skipped where torch/timm are not installed
(e.g. CI), run them on Colab or a machine with the `dl` extra."""

import pytest

torch = pytest.importorskip("torch")


def test_melcnn_shapes():
    pytest.importorskip("torchaudio")
    from forensight.models.audio import MelCNN

    m = MelCNN().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 64_600))
    assert out.shape == (2,)


def test_frame_detector_shapes():
    pytest.importorskip("timm")
    from forensight.models.video import FrameDetector

    m = FrameDetector("efficientnet_b0", pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2,)


def test_aasist_wrapper_converts_to_fake_logit():
    from forensight.models.audio import AASISTWrapper

    class Fake(torch.nn.Module):
        def forward(self, x):  # official AASIST returns (embedding, [spoof, bonafide] logits)
            return None, torch.tensor([[2.0, -1.0], [-3.0, 1.0]])

    out = AASISTWrapper(Fake())(torch.zeros(2, 10))
    assert torch.allclose(out, torch.tensor([3.0, -4.0]))
