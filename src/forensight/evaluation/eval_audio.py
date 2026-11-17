"""Evaluate a pretrained audio model (AASIST from the official repo, or a MelCNN
checkpoint) on one or more ASVspoof evaluation sets.

    python -m forensight.evaluation.eval_audio --config configs/audio_aasist.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from forensight.config import load_config
from forensight.data.datasets import ASVspoofDataset
from forensight.evaluation import metrics
from forensight.training.common import pick_device, predict


def build_model(cfg: dict, device):
    m = cfg["model"]
    if m["name"] == "aasist":
        from forensight.models.audio import load_aasist

        return load_aasist(m.get("repo_dir", "third_party/aasist"), m.get("variant", "AASIST"),
                           m.get("weights"), device=str(device))
    if m["name"] == "melcnn":
        from forensight.models.audio import MelCNN

        model = MelCNN()
        model.load_state_dict(torch.load(m["checkpoint"], map_location=device))
        return model.to(device).eval()
    raise ValueError(f"unknown audio model {m['name']!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    device = pick_device()
    model = build_model(cfg, device)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(cfg.get("chunk_size", 0) or 0)
    for name, e in cfg["eval_sets"].items():
        ds = ASVspoofDataset(e["protocol"], e["audio"], subset=e.get("subset"),
                             only_existing=e.get("only_existing", False))
        loader_kw = dict(batch_size=cfg.get("batch_size", 32), num_workers=2)
        if chunk_size:
            keys, probs, labels = _score_in_chunks(model, ds, name, out_dir, chunk_size, loader_kw, device)
        else:
            keys, probs, labels = predict(model, DataLoader(ds, **loader_kw), device)
        res = metrics.full_report(labels, probs)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(res, indent=2))
        np.savez_compressed(out_dir / f"{name}_scores.npz", keys=np.array(keys), probs=probs, labels=labels)
        print(f"{name}: n={len(ds)}  EER {res['eer'] * 100:.2f}%  AUC {res['auc']:.4f}")


def _score_in_chunks(model, ds, name, out_dir, chunk_size, loader_kw, device):
    """Score ds chunk by chunk, saving each chunk; already-saved chunks are skipped."""
    import time

    from torch.utils.data import Subset

    from forensight.evaluation import chunked

    all_keys = [u.utt_id for u in ds.utts]
    run_dir = out_dir / f"{name}_chunks_{chunked.key_hash(all_keys)}"
    ranges = chunked.chunk_ranges(len(ds), chunk_size)
    done = sum(chunked.chunk_path(run_dir, i).exists() for i in range(len(ranges)))
    print(f"{name}: {len(ds)} clips in {len(ranges)} chunks of {chunk_size}; {done} already saved in {run_dir}")
    for i, (s, t) in enumerate(ranges):
        if chunked.chunk_path(run_dir, i).exists():
            continue
        start = time.time()
        keys, probs, labels = predict(model, DataLoader(Subset(ds, range(s, t)), **loader_kw), device)
        chunked.save_chunk(run_dir, i, keys, probs, labels)
        print(f"  chunk {i + 1}/{len(ranges)} saved ({t - s} clips, {time.time() - start:.0f}s)", flush=True)
    return chunked.merge_chunks(run_dir, len(ranges), all_keys)

if __name__ == "__main__":
    main()
