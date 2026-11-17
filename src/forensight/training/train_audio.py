"""Train the MelCNN audio baseline on ASVspoof 2019 LA, evaluate on 2021 DF.

    python -m forensight.training.train_audio --config configs/audio_melcnn.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from forensight.config import load_config
from forensight.data.datasets import ASVspoofDataset
from forensight.models.audio import MelCNN
from forensight.training.common import evaluate, fit, pick_device, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 42))
    d, t = cfg["data"], cfg["train"]

    train_ds = ASVspoofDataset(d["train_protocol"], d["train_audio"], train=True)
    val_ds = ASVspoofDataset(d["dev_protocol"], d["dev_audio"])
    loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=t["batch_size"], shuffle=shuffle, num_workers=t.get("num_workers", 2),
        pin_memory=True, drop_last=shuffle)
    print(f"utterances: train {len(train_ds)}, dev {len(val_ds)}")

    model = MelCNN()
    out_dir = Path(cfg["output_dir"])
    fit(model, loader(train_ds, True), loader(val_ds, False), cfg, out_dir)

    device = pick_device()
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    for name, e in (cfg.get("eval_sets") or {}).items():
        ds = ASVspoofDataset(e["protocol"], e["audio"], subset=e.get("subset"),
                             only_existing=e.get("only_existing", False))
        res = evaluate(model.to(device), loader(ds, False), device, group_by_key=False)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
        print(f"{name}: EER {res['eer'] * 100:.2f}%  AUC {res['auc']:.4f}")


if __name__ == "__main__":
    main()
