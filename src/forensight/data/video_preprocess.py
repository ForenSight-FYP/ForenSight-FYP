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


_MODEL_URLS = {
    "mediapipe": "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "yunet": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
}


def _model_file(name: str) -> str:
    """Download a detector model once into ~/.cache/forensight and return its path."""
    import urllib.request

    url = _MODEL_URLS[name]
    dest = Path.home() / ".cache" / "forensight" / url.rsplit("/", 1)[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return str(dest)


class FaceCropper:
    """Largest-face detector. Tries, in order: MediaPipe's legacy ``solutions`` API (older
    releases), MediaPipe Tasks ``FaceDetector`` (newer releases removed ``solutions``),
    OpenCV YuNet, then OpenCV Haar. ``backend`` says which one is in use; Haar is only
    good enough for smoke tests."""

    def __init__(self, scale: float = 1.3, min_confidence: float = 0.5, backends=None):
        self.scale = scale
        self.min_confidence = min_confidence
        self.backend = None
        self._det = None
        errors = []
        for name in backends or ("mediapipe", "mediapipe-tasks", "yunet", "haar"):
            try:
                self._det = getattr(self, "_init_" + name.replace("-", "_"))()
                self.backend = name
                break
            except Exception as e:  # depends on installed library versions
                errors.append(f"{name}: {type(e).__name__}: {e}")
        if self.backend is None:
            raise RuntimeError("No face detector could be loaded:\n  " + "\n  ".join(errors))
        self.load_errors = errors

    def _init_mediapipe(self):
        import mediapipe as mp

        return mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=self.min_confidence
        )

    def _init_mediapipe_tasks(self):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_model_file("mediapipe")),
            min_detection_confidence=self.min_confidence,
        )
        return vision.FaceDetector.create_from_options(opts)

    def _init_yunet(self):
        return cv2.FaceDetectorYN.create(_model_file("yunet"), "", (320, 320), self.min_confidence)

    def _init_haar(self):
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if det.empty():
            raise RuntimeError("Haar cascade file not found")
        return det

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        boxes = []
        if self.backend == "mediapipe":
            res = self._det.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                b = d.location_data.relative_bounding_box
                boxes.append((int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h)))
        elif self.backend == "mediapipe-tasks":
            import mediapipe as mp

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            for d in res.detections or []:
                b = d.bounding_box
                boxes.append((int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)))
        elif self.backend == "yunet":
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(frame_bgr)
            boxes = [tuple(int(v) for v in f[:4]) for f in (faces if faces is not None else [])]
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = [tuple(map(int, b)) for b in self._det.detectMultiScale(gray, 1.1, 5)]
        boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
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


_FIELDS = ["path", "label", "method", "video_id", "frame"]


def _video_dir(item) -> Path:
    return Path(str(item.label)) / item.method / item.video_id


def preprocess_split(items, out_dir: Path, num_frames: int, size: int, cropper: FaceCropper,
                     log_every: int = 100) -> int:
    """Write face crops for every video in items. Resumable: a video whose crop folder
    already has a ``.done`` marker is not processed again, so an interrupted run only
    does the videos that are left."""
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped, start = [], 0, time.time()
    for n, item in enumerate(items, 1):
        vdir = out_dir / _video_dir(item)
        if not (vdir / ".done").exists():
            vdir.mkdir(parents=True, exist_ok=True)
            for idx, frame in read_frames(item.path, num_frames):
                face = cropper.crop(frame, size)
                if face is None:
                    skipped += 1
                    continue
                cv2.imwrite(str(vdir / f"{idx:05d}.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (vdir / ".done").touch()
        for jpg in sorted(vdir.glob("*.jpg")):
            rows.append({"path": str(_video_dir(item) / jpg.name), "label": item.label,
                         "method": item.method, "video_id": item.video_id, "frame": int(jpg.stem)})
        if n % log_every == 0 or n == len(items):
            print(f"  {n}/{len(items)} videos ({time.time() - start:.0f}s)", flush=True)
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return skipped


def archive_split(split_dir: Path, archive: Path) -> None:
    """Pack one finished split into a single .tar in Drive (one big file copies far faster
    to and from Drive than tens of thousands of small JPEGs). Written via a temporary name
    so a half-copied archive is never mistaken for a finished one."""
    import tarfile

    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(archive.name + ".part")
    with tarfile.open(tmp, "w") as tar:
        tar.add(split_dir, arcname=split_dir.name)
    tmp.replace(archive)


def restore_split(archive: Path, frames_dir: Path) -> None:
    import tarfile

    frames_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(frames_dir, filter="data")
        except TypeError:  # Python versions without extraction filters
            tar.extractall(frames_dir)


def main(argv=None):
    from forensight.config import load_config
    from forensight.data import ffpp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    frames_dir = Path(d["frames_dir"])
    archive_dir = Path(d["frames_archive_dir"]) if d.get("frames_archive_dir") else None
    cropper = None
    for split in ("train", "val", "test"):
        local = frames_dir / split
        archive = archive_dir / f"{split}.tar" if archive_dir else None
        if (local / "index.csv").exists():
            print(f"[{split}] face crops already on local disk")
            continue
        if archive and archive.exists():
            print(f"[{split}] restoring face crops from {archive}")
            restore_split(archive, frames_dir)
            continue
        if cropper is None:
            cropper = FaceCropper(scale=d.get("face_scale", 1.3))
            print("face detector:", cropper.backend)
            if cropper.backend == "haar":
                print("WARNING: Haar is a weak detector; fix MediaPipe/YuNet before the real run:")
                for err in cropper.load_errors:
                    print("  ", err)
        items = ffpp.build_index(d["ffpp_root"], d["ffpp_splits"], split, d.get("compression", "c23"))
        print(f"[{split}] cropping faces from {len(items)} videos")
        skipped = preprocess_split(items, local, d["num_frames"], d["image_size"], cropper)
        print(f"[{split}] done; {skipped} frames without a face skipped")
        if archive:
            archive_split(local, archive)
            print(f"[{split}] saved {archive}")


if __name__ == "__main__":
    main()
