"""Audio detectors.

1. MelCNN: a small log-mel spectrogram CNN. Trains in minutes on Colab, gives a
   first baseline quickly, and works directly with spectrogram saliency maps.
2. AASIST: the state-of-the-art raw-waveform anti-spoofing model. We do not
   re-implement it; scripts/setup_aasist.sh clones the official code
   (github.com/clovaai/aasist) and its pretrained weights, and `load_aasist`
   wraps it so it outputs a fake score like every other ForenSight module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn


class MelCNN(nn.Module):
    def __init__(self, sample_rate: int = 16_000, n_mels: int = 80, channels=(32, 64, 128, 128)):
        super().__init__()
        import torchaudio

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, win_length=400, hop_length=160, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c_in, 1))

    def spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) waveform -> (B, 1, n_mels, frames) normalised log-mel."""
        spec = self.to_db(self.melspec(wav)).unsqueeze(1)
        return (spec - spec.mean(dim=(2, 3), keepdim=True)) / (spec.std(dim=(2, 3), keepdim=True) + 1e-5)

    def forward_spec(self, spec: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(spec)).squeeze(1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Returns fake logits of shape (B,)."""
        return self.forward_spec(self.spectrogram(wav))


class AASISTWrapper(nn.Module):
    """Adapts the official AASIST model to ForenSight's convention.

    The official code labels bona fide as 1 and spoof as 0, and its evaluation score
    is the bona fide logit. We return a *fake* logit instead: logit(spoof) - logit(bonafide),
    so sigmoid(output) is the probability of spoof.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _, out = self.model(wav)
        return out[:, 0] - out[:, 1]


def load_aasist(repo_dir: str | Path = "third_party/aasist", variant: str = "AASIST",
                weights: str | Path | None = None, device: str = "cpu") -> AASISTWrapper:
    """Load the official AASIST (or AASIST-L) model with pretrained weights."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "models").exists():
        raise FileNotFoundError(f"AASIST code not found in {repo_dir}. Run scripts/setup_aasist.sh first.")
    sys.path.insert(0, str(repo_dir))
    from importlib import import_module

    conf = json.loads((repo_dir / "config" / f"{variant}.conf").read_text())
    model = import_module(f"models.{conf['model_config']['architecture']}").Model(conf["model_config"])
    weights = Path(weights) if weights else repo_dir / "models" / "weights" / f"{variant}.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    return AASISTWrapper(model).to(device).eval()
