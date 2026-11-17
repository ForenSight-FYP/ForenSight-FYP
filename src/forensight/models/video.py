"""Video (frame-level) detector: an ImageNet-pretrained CNN with one output logit.

EfficientNet-B4 at 380x380 is a strong, widely used FF++ baseline. The backbone is
created through timm so it can be swapped from the config (e.g. "xception41",
"efficientnet_b0" for quick experiments on a small GPU).
"""

from __future__ import annotations

import torch
from torch import nn


class FrameDetector(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fake logits of shape (B,). Apply sigmoid for a fake probability."""
        return self.head(self.dropout(self.backbone(x))).squeeze(1)

    def gradcam_target_layer(self) -> nn.Module:
        """Last convolutional block: where Grad-CAM heatmaps are taken from (FYP2)."""
        for name in ("conv_head", "act2"):
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
        return list(self.backbone.children())[-2]
