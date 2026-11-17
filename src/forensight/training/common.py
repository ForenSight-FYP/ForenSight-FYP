"""Shared training loop: binary classification with fake logits.

Handles class imbalance (FF++ has 4 fakes per real; ASVspoof has ~9 spoofs per
bona fide) with pos_weight, mixed precision on GPU, best-checkpoint saving on the
validation metric, and a JSON log of every epoch.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from forensight.evaluation import metrics
from forensight.evaluation.aggregate import aggregate


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Returns (keys, fake probabilities, labels) for every sample in the loader."""
    model.eval()
    keys, probs, labels = [], [], []
    for x, y, k in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
        keys.extend(k)
    return keys, np.concatenate(probs), np.concatenate(labels).astype(int)


def evaluate(model, loader, device, group_by_key: bool) -> dict:
    keys, probs, labels = predict(model, loader, device)
    if group_by_key:  # frames -> videos
        keys, probs, labels = aggregate(keys, probs, labels, how="mean")
    return metrics.full_report(labels, probs)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: dict,
        out_dir: str | Path, group_val_by_key: bool = False) -> dict:
    t = cfg["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model.to(device)

    labels = train_loader.dataset.labels()
    n_fake, n_real = int(labels.sum()), int(len(labels) - labels.sum())
    pos_weight = torch.tensor(n_real / max(n_fake, 1), device=device)  # down-weights the majority fakes
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=t["lr"], total_steps=t["epochs"] * len(train_loader), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    monitor, mode = t.get("monitor", "auc"), t.get("monitor_mode", "max")
    best, history = None, []

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, (x, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % t.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        val = evaluate(model, val_loader, device, group_val_by_key)
        entry = {"epoch": epoch, "train_loss": running / len(train_loader),
                 "seconds": round(time.time() - start, 1), **{f"val_{k}": v for k, v in val.items()}}
        history.append(entry)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}))

        score = val[monitor]
        if best is None or (score > best if mode == "max" else score < best):
            best = score
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  new best val {monitor} = {score:.4f} -> saved best.pt")
        torch.save(model.state_dict(), out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"best_" + monitor: best, "history": history}
