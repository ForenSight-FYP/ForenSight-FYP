"""FaceForensics++ index builder.

Expected layout after downloading with the official FF++ script (compression c23):

    <root>/original_sequences/youtube/c23/videos/071.mp4
    <root>/manipulated_sequences/Deepfakes/c23/videos/071_054.mp4
    <root>/manipulated_sequences/Face2Face/c23/videos/071_054.mp4
    ...
    <splits_dir>/train.json  val.json  test.json   (from the FF++ GitHub repo, dataset/splits/)

Each split file is a list of [target, source] id pairs, e.g. [["071", "054"], ...].
A pair contributes the two original videos (071, 054) and the manipulated videos
"071_054" and "054_071" for every method. Using the official splits keeps identities
separate between train/val/test and makes results comparable with published work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forensight import FAKE, REAL

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


@dataclass(frozen=True)
class VideoItem:
    path: Path
    label: int
    method: str  # "original" or the manipulation method
    video_id: str


def _pairs(splits_dir: Path, split: str) -> list[tuple[str, str]]:
    with open(splits_dir / f"{split}.json", encoding="utf-8") as f:
        return [(a, b) for a, b in json.load(f)]


def build_index(
    root: str | Path,
    splits_dir: str | Path,
    split: str,
    compression: str = "c23",
    methods: tuple[str, ...] = METHODS,
    check_exists: bool = True,
) -> list[VideoItem]:
    root, splits_dir = Path(root), Path(splits_dir)
    items: list[VideoItem] = []
    for a, b in _pairs(splits_dir, split):
        for vid in (a, b):
            p = root / "original_sequences" / "youtube" / compression / "videos" / f"{vid}.mp4"
            items.append(VideoItem(p, REAL, "original", vid))
        for m in methods:
            for vid in (f"{a}_{b}", f"{b}_{a}"):
                p = root / "manipulated_sequences" / m / compression / "videos" / f"{vid}.mp4"
                items.append(VideoItem(p, FAKE, m, vid))
    if check_exists:
        missing = [i.path for i in items if not i.path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(items)} FF++ videos missing, e.g. {missing[0]}. "
                "Check the root path and compression level."
            )
    return items
