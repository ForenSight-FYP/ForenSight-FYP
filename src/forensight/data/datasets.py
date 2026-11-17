"""PyTorch datasets (require the `dl` extra: torch, torchvision, soundfile)."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from forensight.data import asvspoof
from forensight.data.audio_preprocess import NUM_SAMPLES, load_audio, pad_or_crop

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceFrameDataset(Dataset):
    """Face crops written by video_preprocess.py, read from <split_dir>/index.csv."""

    def __init__(self, split_dir: str | Path, image_size: int = 380, train: bool = False,
                 max_frames_per_video: int | None = None):
        self.root = Path(split_dir)
        self.image_size = image_size
        self.train = train
        with open(self.root / "index.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if max_frames_per_video:
            kept, count = [], {}
            for r in rows:
                n = count.get(r["video_id"], 0)
                if n < max_frames_per_video:
                    kept.append(r)
                    count[r["video_id"]] = n + 1
            rows = kept
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def labels(self) -> np.ndarray:
        return np.array([int(r["label"]) for r in self.rows])

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Light, artefact-preserving augmentation: flips, brightness, and JPEG
        # recompression (teaches robustness to the compression seen on social media).
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        if np.random.rand() < 0.3:
            img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            q = int(np.random.randint(40, 95))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return np.ascontiguousarray(img)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.root / r["path"]), cv2.IMREAD_COLOR)
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        if self.train:
            img = self._augment(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img.transpose(2, 0, 1))
        return x, torch.tensor(float(r["label"])), r["video_id"]


class ASVspoofDataset(Dataset):
    """Raw-waveform ASVspoof dataset (works for 2019 LA and 2021 DF)."""

    def __init__(self, protocol: str | Path, audio_dir: str | Path, train: bool = False,
                 num_samples: int = NUM_SAMPLES, ext: str = ".flac", subset: str | None = None):
        utts = asvspoof.read_protocol(protocol)
        if subset:
            utts = [u for u in utts if u.subset == subset]
        self.utts = utts
        self.audio_dir = Path(audio_dir)
        self.train = train
        self.num_samples = num_samples
        self.ext = ext

    def __len__(self):
        return len(self.utts)

    def labels(self) -> np.ndarray:
        return np.array([u.label for u in self.utts])

    def __getitem__(self, i):
        u = self.utts[i]
        x = load_audio(asvspoof.audio_path(self.audio_dir, u.utt_id, self.ext))
        x = pad_or_crop(x, self.num_samples, random_crop=self.train)
        return torch.from_numpy(x), torch.tensor(float(u.label)), u.utt_id
