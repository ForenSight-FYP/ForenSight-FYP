"""Resumable, chunked scoring for large evaluation sets.

Scoring ASVspoof 2021 DF (~611k clips) takes hours, longer than a Colab session may
last. Scores are written to disk one chunk at a time; a re-run skips chunks that are
already saved and only computes the rest. The chunk folder name includes a hash of
the exact list of utterances, so adding more downloaded parts later starts a fresh,
consistent set of chunks instead of mixing old and new orderings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def key_hash(keys: Sequence[str]) -> str:
    h = hashlib.sha1()
    for k in keys:
        h.update(k.encode())
        h.update(b"\n")
    return h.hexdigest()[:10]


def chunk_ranges(n: int, size: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def chunk_path(run_dir: Path, i: int) -> Path:
    return Path(run_dir) / f"chunk_{i:05d}.npz"


def save_chunk(run_dir: Path, i: int, keys, probs, labels) -> None:
    """Write via a temporary file and rename, so an interrupted save never leaves a
    half-written chunk that a later run would trust."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f".tmp_chunk_{i:05d}.npz"
    np.savez_compressed(tmp, keys=np.asarray(keys), probs=np.asarray(probs), labels=np.asarray(labels))
    tmp.replace(chunk_path(run_dir, i))


def merge_chunks(run_dir: Path, n_chunks: int, expected_keys: Sequence[str]):
    """Concatenate all chunks and check they cover exactly the expected utterances, in order."""
    keys, probs, labels = [], [], []
    for i in range(n_chunks):
        p = chunk_path(run_dir, i)
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; re-run to finish scoring")
        d = np.load(p, allow_pickle=False)
        keys.extend(d["keys"].tolist())
        probs.append(d["probs"])
        labels.append(d["labels"])
    if list(keys) != list(expected_keys):
        raise ValueError("saved chunks do not match the current utterance list; delete the chunk folder and re-run")
    return keys, np.concatenate(probs), np.concatenate(labels).astype(int)
