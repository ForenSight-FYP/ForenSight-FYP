"""Analyse one media file: the core of the ForenSight product (Step 2).

    from forensight.inference import Analyser
    analyser = Analyser(video_checkpoint=".../runs/video_effnet_b4/best.pt")
    result = analyser.analyse("clip.mp4")
    print(result.summary())

Video path: sample frames -> crop the largest face -> EfficientNet-B4 -> mean fake score.
Audio path: decode the soundtrack (ffmpeg) -> 4 s windows -> AASIST -> mean fake score.
Both use exactly the preprocessing used in training/evaluation, so the scores match
the numbers in docs/experiments.md. Per-frame and per-window scores are kept: they
become the timeline in the explainable report (Step 6).

Models are loaded once in Analyser.__init__ and reused for every file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# Decision thresholds on the fake probability.
# Video: picked on the FF++ validation set (Sync Hub, 20 Nov), not on the test set.
VIDEO_THRESHOLD = 0.228
# Audio: provisional. AASIST's raw scores are not calibrated; Step 3 (fusion) replaces
# this with a threshold learned on the 2019 LA dev set.
AUDIO_THRESHOLD = 0.5


@dataclass
class AnalysisResult:
    file: str
    video_score: float | None = None      # fake probability, mean over face frames
    audio_score: float | None = None      # fake probability, mean over 4 s windows
    frames_sampled: int = 0
    faces_found: int = 0
    frame_scores: list[float] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    audio_window_scores: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def video_fake(self) -> bool | None:
        return None if self.video_score is None else self.video_score >= VIDEO_THRESHOLD

    @property
    def audio_fake(self) -> bool | None:
        return None if self.audio_score is None else self.audio_score >= AUDIO_THRESHOLD

    @property
    def verdict(self) -> str:
        """Simple rule until Step 3: FAKE if any available module says fake."""
        flags = [f for f in (self.video_fake, self.audio_fake) if f is not None]
        if not flags:
            return "UNKNOWN"
        return "FAKE" if any(flags) else "REAL"

    def summary(self) -> str:
        v = "n/a" if self.video_score is None else f"{self.video_score:.2f}"
        a = "n/a" if self.audio_score is None else f"{self.audio_score:.2f}"
        line = (f"{Path(self.file).name} -> video {v} (faces {self.faces_found}/{self.frames_sampled}), "
                f"audio {a} -> {self.verdict}")
        return line + "".join(f"\n   note: {n}" for n in self.notes)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prepare_face(face_bgr: np.ndarray, image_size: int = 380) -> np.ndarray:
    """Same steps as FaceFrameDataset (no augmentation): resize, BGR->RGB, /255,
    ImageNet normalisation, HWC->CHW."""
    import cv2

    from forensight.data.datasets import IMAGENET_MEAN, IMAGENET_STD

    if face_bgr.shape[0] != image_size or face_bgr.shape[1] != image_size:
        face_bgr = cv2.resize(face_bgr, (image_size, image_size))
    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


class Analyser:
    def __init__(self, video_checkpoint: str | Path | None, backbone: str = "efficientnet_b4",
                 image_size: int = 380, num_frames: int = 16, face_scale: float = 1.3,
                 use_audio: bool = True, aasist_dir: str | Path = "third_party/aasist",
                 device: str | None = None, batch_size: int = 16, cropper=None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size, self.num_frames, self.batch_size = image_size, num_frames, batch_size

        self.video_model, self.cropper = None, None
        if video_checkpoint is not None:
            from forensight.data.video_preprocess import FaceCropper
            from forensight.models.video import FrameDetector

            model = FrameDetector(backbone=backbone, pretrained=False)
            state = torch.load(video_checkpoint, map_location=self.device)
            model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
            self.video_model = model.to(self.device).eval()
            self.cropper = cropper or FaceCropper(scale=face_scale)

        self.audio_model = None
        if use_audio:
            from forensight.models.audio import load_aasist

            self.audio_model = load_aasist(aasist_dir, device=self.device)

    # ---- video -------------------------------------------------------------------
    def _score_video(self, path: Path, res: AnalysisResult) -> None:
        import torch

        from forensight.data.video_preprocess import read_frames

        frames = read_frames(path, self.num_frames)
        res.frames_sampled = len(frames)
        faces, idxs = [], []
        for idx, frame in frames:
            face = self.cropper.crop(frame, size=self.image_size)
            if face is not None:
                faces.append(prepare_face(face, self.image_size))
                idxs.append(idx)
        res.faces_found = len(faces)
        if not frames:
            res.notes.append("could not read any video frames")
            return
        if not faces:
            res.notes.append("no face found in the sampled frames: video model skipped")
            return
        scores = []
        with torch.no_grad():
            for i in range(0, len(faces), self.batch_size):
                x = torch.from_numpy(np.stack(faces[i:i + self.batch_size])).to(self.device)
                scores.append(_sigmoid(self.video_model(x).float().cpu().numpy()))
        s = np.concatenate(scores)
        res.frame_scores, res.frame_indices = s.round(4).tolist(), idxs
        res.video_score = float(s.mean())  # same "mean" aggregation as the test metric

    # ---- audio -------------------------------------------------------------------
    def _score_audio(self, path: Path, res: AnalysisResult) -> None:
        import torch

        from forensight.data.audio_preprocess import SAMPLE_RATE, load_audio, segments

        try:
            wav = load_audio(path)
        except Exception:
            res.notes.append("no readable audio track: audio model skipped")
            return
        if len(wav) < SAMPLE_RATE // 2 or float(np.abs(wav).max(initial=0.0)) < 1e-4:
            res.notes.append("audio is shorter than 0.5 s or silent: audio model skipped")
            return
        windows = segments(wav)
        scores = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                x = torch.from_numpy(windows[i:i + self.batch_size]).to(self.device)
                scores.append(_sigmoid(self.audio_model(x).float().cpu().numpy()))
        s = np.concatenate(scores)
        res.audio_window_scores = s.round(4).tolist()
        res.audio_score = float(s.mean())

    # ---- public ------------------------------------------------------------------
    def analyse(self, path: str | Path) -> AnalysisResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        res = AnalysisResult(file=str(path))
        ext = path.suffix.lower()
        if ext in VIDEO_EXTS and self.video_model is not None:
            self._score_video(path, res)
        if (ext in VIDEO_EXTS or ext in AUDIO_EXTS) and self.audio_model is not None:
            self._score_audio(path, res)
        if ext not in VIDEO_EXTS | AUDIO_EXTS:
            res.notes.append(f"unsupported file type {ext!r}")
        return res


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Analyse video/audio files for deepfakes.")
    p.add_argument("files", nargs="+")
    p.add_argument("--video-checkpoint",
                   default="/content/drive/MyDrive/forensight/runs/video_effnet_b4/best.pt")
    p.add_argument("--no-audio", action="store_true")
    a = p.parse_args(argv)
    analyser = Analyser(a.video_checkpoint, use_audio=not a.no_audio)
    for f in a.files:
        print(analyser.analyse(f).summary())


if __name__ == "__main__":
    main()
