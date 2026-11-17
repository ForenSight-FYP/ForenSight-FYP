"""Train the frame-level video detector on preprocessed face crops.

    python -m forensight.training.train_video --config configs/video_effnet.yaml
    python -m forensight.training.train_video --config configs/video_effnet.yaml train.epochs=1  # smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from forensight.config import load_config
from forensight.data.datasets import FaceFrameDataset
from forensight.models.video import FrameDetector
from forensight.training.common import evaluate, fit, pick_device, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))

    d, t = cfg["data"], cfg["train"]
    frames = Path(d["frames_dir"])
    mk = lambda split, train: FaceFrameDataset(
        frames / split, d["image_size"], train=train,
        max_frames_per_video=d.get("frames_per_video_train") if train else None)
    loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=t["batch_size"], shuffle=shuffle, num_workers=t.get("num_workers", 2),
        pin_memory=True, drop_last=shuffle)

    train_ds, val_ds, test_ds = mk("train", True), mk("val", False), mk("test", False)
    print(f"frames: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

    model = FrameDetector(cfg["model"]["backbone"], pretrained=cfg["model"].get("pretrained", True))
    out_dir = Path(cfg["output_dir"])
    fit(model, loader(train_ds, True), loader(val_ds, False), cfg, out_dir, group_val_by_key=True)

    device = pick_device()
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test = evaluate(model.to(device), loader(test_ds, False), device, group_by_key=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2))
    print("TEST (video-level):", json.dumps({k: round(v, 4) for k, v in test.items()}))


if __name__ == "__main__":
    main()
