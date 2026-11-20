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


class NonFiniteModelError(RuntimeError):
    """The model's weights or outputs became NaN/inf (usually a mixed-precision overflow)."""


def model_is_finite(model: nn.Module) -> bool:
    """True if every parameter and buffer (incl. BatchNorm running stats) is finite."""
    return all(torch.isfinite(t).all().item()
               for t in (*model.parameters(), *model.buffers()) if t.is_floating_point())


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Returns (keys, fake probabilities, labels) for every sample in the loader.

    Runs in fp16 on GPU for speed. A batch that overflows to NaN/inf in fp16 is
    recomputed in fp32; if it is still not finite, the model itself is broken and
    NonFiniteModelError is raised instead of passing NaN on to the metrics.
    """
    model.eval()
    keys, probs, labels = [], [], []
    n_fallback = 0
    for x, y, k in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x).float()
        if not torch.isfinite(logits).all():
            n_fallback += 1
            logits = model(x.float()).float()   # outside autocast: full fp32
            if not torch.isfinite(logits).all():
                raise NonFiniteModelError(
                    "model outputs are NaN/inf even in fp32: the weights or BatchNorm statistics are broken")
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
        keys.extend(k)
    if n_fallback:
        print(f"  note: {n_fallback} batch(es) overflowed in fp16 and were recomputed in fp32")
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
    best, history, start_epoch = None, [], 1

    # Resume: a run interrupted by a Colab disconnect continues from its last finished epoch.
    ckpt_path = out_dir / "checkpoint.pt"
    if t.get("resume", True) and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, history, start_epoch = ck["best"], ck["history"], ck["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best val {monitor} so far: {best})")

    # BatchNorm running stats are updated in the forward pass, so GradScaler (which only
    # skips the optimizer step) cannot protect them from an fp16 overflow. Keep a copy of
    # the last good values and put it back after a non-finite batch.
    buffers = [b for b in model.buffers() if b.is_floating_point()]

    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        start, running, n_ok, n_skipped = time.time(), 0.0, 0, 0
        good_buffers = [b.detach().clone() for b in buffers]
        for step, (x, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x).float(), y)
            if not torch.isfinite(loss):
                n_skipped += 1
                with torch.no_grad():
                    for b, g in zip(buffers, good_buffers):
                        b.copy_(g)
                sched.step()   # keep the LR schedule aligned with the step count
                continue
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            with torch.no_grad():
                for b, g in zip(buffers, good_buffers):
                    g.copy_(b)
            running += loss.item()
            n_ok += 1
            if step % t.get("log_every", 50) == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / max(n_ok, 1):.4f}")
        if n_skipped:
            print(f"  note: skipped {n_skipped} batch(es) with a NaN/inf loss this epoch")

        if not model_is_finite(model):
            raise NonFiniteModelError(
                f"model weights became NaN/inf during epoch {epoch}; nothing was saved for this epoch. "
                f"checkpoint.pt still holds epoch {epoch - 1}: run the cell again to resume from there.")
        val = evaluate(model, val_loader, device, group_val_by_key)
        entry = {"epoch": epoch, "train_loss": running / max(n_ok, 1), "skipped_batches": n_skipped,
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
        tmp = out_dir / "checkpoint.pt.tmp"   # written then renamed, so a disconnect never leaves a broken file
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history}, tmp)
        tmp.replace(ckpt_path)

    return {"best_" + monitor: best, "history": history}
