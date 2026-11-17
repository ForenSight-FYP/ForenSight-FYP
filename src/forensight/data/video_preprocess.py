"""Video preprocessing: sample frames uniformly and crop the largest face.

Run once per dataset to turn videos into face-crop images on disk, so training
does not decode video every epoch:

    python -m forensight.data.video_preprocess --config configs/video_effnet.yaml

Output layout: <out_dir>/<split>/<label>/<method>/<video_id>/<frame_idx>.jpg
plus <out_dir>/<split>/index.csv listing every crop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def sample_frame_indices(total: int, num: int) -> list[int]:
    """Evenly spaced frame indices across the video (all frames if it is short)."""
    if total <= 0:
        return []
    if total <= num:
        return list(range(total))
    return np.linspace(0, total - 1, num).round().astype(int).tolist()


def read_frames(path: str | Path, num: int) -> list[tuple[int, np.ndarray]]:
    """Return [(frame_index, BGR frame)] for evenly sampled frames."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in sample_frame_indices(total, num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


def expand_box(x: int, y: int, w: int, h: int, scale: float, img_w: int, img_h: int):
    """Grow a face box around its centre (faces need context: hairline, jaw, edges
    are where blending artefacts show) and clip it to the image."""
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * scale
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(img_w, cx + side / 2))
    y1 = int(min(img_h, cy + side / 2))
    return x0, y0, x1, y1


class FaceCropper:
    """Largest-face detector using MediaPipe (falls back to OpenCV Haar if MediaPipe
    is unavailable, which is fine for smoke tests but not for real experiments)."""

    def __init__(self, scale: float = 1.3, min_confidence: float = 0.5):
        self.scale = scale
        try:
            import mediapipe as mp

            self._mp = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=min_confidence
            )
            self._haar = None
        except Exception:  # pragma: no cover - depends on environment
            self._mp = None
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        boxes = []
        if self._mp is not None:
            res = self._mp.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                b = d.location_data.relative_bounding_box
                boxes.append((int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h)))
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = [tuple(map(int, b)) for b in self._haar.detectMultiScale(gray, 1.1, 5)]
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])

    def crop(self, frame_bgr: np.ndarray, size: int = 380) -> np.ndarray | None:
        box = self.detect(frame_bgr)
        if box is None:
            return None
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = expand_box(*box, self.scale, w, h)
        if x1 <= x0 or y1 <= y0:
            return None
        return cv2.resize(frame_bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def preprocess_split(items, out_dir: Path, num_frames: int, size: int, cropper: FaceCropper) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for item in items:
        for idx, frame in read_frames(item.path, num_frames):
            face = cropper.crop(frame, size)
            if face is None:
                skipped += 1
                continue
            rel = Path(str(item.label)) / item.method / item.video_id / f"{idx:05d}.jpg"
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"path": str(rel), "label": item.label, "method": item.method,
                         "video_id": item.video_id, "frame": idx})
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "method", "video_id", "frame"])
        writer.writeheader()
        writer.writerows(rows)
    return skipped


def main(argv=None):
    from tqdm import tqdm  # noqa: F401  (progress bars in long runs)

    from forensight.config import load_config
    from forensight.data import ffpp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    cropper = FaceCropper(scale=d.get("face_scale", 1.3))
    for split in ("train", "val", "test"):
        items = ffpp.build_index(d["ffpp_root"], d["ffpp_splits"], split, d.get("compression", "c23"))
        skipped = preprocess_split(items, Path(d["frames_dir"]) / split, d["num_frames"],
                                   d["image_size"], cropper)
        print(f"[{split}] {len(items)} videos processed, {skipped} frames without a face skipped")


if __name__ == "__main__":
    main()
