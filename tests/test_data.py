import json

import cv2
import numpy as np
import pytest

from forensight import FAKE, REAL
from forensight.config import apply_overrides
from forensight.data import asvspoof, ffpp
from forensight.data.audio_preprocess import NUM_SAMPLES, pad_or_crop, resample, segments
from forensight.data.splits import group_split
from forensight.data.video_preprocess import FaceCropper, expand_box, read_frames, sample_frame_indices
from forensight.evaluation.aggregate import aggregate


# ---------- splits ----------
def test_group_split_is_identity_disjoint():
    items = list(range(1000))
    groups = [i // 10 for i in items]  # 100 identities, 10 clips each
    tr, va, te = group_split(items, groups, seed=1)
    g = lambda xs: {x // 10 for x in xs}
    assert not (g(tr) & g(va) or g(tr) & g(te) or g(va) & g(te))
    assert len(tr) + len(va) + len(te) == 1000
    assert len(g(tr)) == 80


# ---------- ASVspoof protocols ----------
def test_parse_2019_la_lines():
    b = asvspoof.parse_line("LA_0079 LA_T_1138215 - - bonafide")
    s = asvspoof.parse_line("LA_0079 LA_T_1271820 - A01 spoof")
    assert (b.label, b.attack, b.utt_id) == (REAL, None, "LA_T_1138215")
    assert (s.label, s.attack) == (FAKE, "A01")


def test_parse_2021_df_line():
    u = asvspoof.parse_line(
        "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -")
    assert (u.speaker, u.utt_id, u.label, u.attack, u.subset) == ("LA_0023", "DF_E_2000011", FAKE, "A14", "progress")


def test_parse_rejects_unlabelled_line():
    with pytest.raises(ValueError):
        asvspoof.parse_line("LA_0001 LA_T_1 - A01 unknown")


# ---------- FF++ index ----------
def test_ffpp_index(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "test.json").write_text(json.dumps([["071", "054"]]))
    items = ffpp.build_index(tmp_path, tmp_path / "splits", "test", check_exists=False)
    assert sum(i.label == REAL for i in items) == 2
    assert sum(i.label == FAKE for i in items) == 2 * len(ffpp.METHODS)
    assert any(i.video_id == "054_071" and i.method == "Face2Face" for i in items)
    with pytest.raises(FileNotFoundError):
        ffpp.build_index(tmp_path, tmp_path / "splits", "test", check_exists=True)


# ---------- audio ----------
def test_pad_repeats_short_clip():
    x = np.arange(10, dtype=np.float32)
    y = pad_or_crop(x, 25)
    assert len(y) == 25 and y[10] == 0 and y[24] == 4


def test_crop_long_clip():
    x = np.arange(100_000, dtype=np.float32)
    assert len(pad_or_crop(x)) == NUM_SAMPLES
    rng = np.random.default_rng(0)
    assert len(pad_or_crop(x, random_crop=True, rng=rng)) == NUM_SAMPLES


def test_segments_cover_whole_clip():
    x = np.arange(200_000, dtype=np.float32)
    seg = segments(x)
    assert seg.shape[1] == NUM_SAMPLES
    assert seg[-1, -1] == x[-1]


def test_resample_length():
    x = np.zeros(44_100, dtype=np.float32)
    assert len(resample(x, 44_100)) == 16_000


# ---------- video ----------
def test_sample_frame_indices():
    assert sample_frame_indices(5, 32) == [0, 1, 2, 3, 4]
    idx = sample_frame_indices(300, 4)
    assert idx[0] == 0 and idx[-1] == 299 and len(idx) == 4


def test_expand_box_clips_to_image():
    assert expand_box(0, 0, 100, 100, 1.3, 120, 120) == (0, 0, 115, 115)


def test_read_frames_from_synthetic_video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    w.release()
    frames = read_frames(path, 5)
    assert len(frames) == 5
    assert frames[0][1].shape == (48, 64, 3)


# ---------- aggregation & config ----------
def test_aggregate_by_video():
    keys, s, y = aggregate(["a", "a", "b", "b"], [0.2, 0.4, 0.9, 0.7], [0, 0, 1, 1])
    assert keys == ["a", "b"] and np.allclose(s, [0.3, 0.8]) and list(y) == [0, 1]
    _, s_max, _ = aggregate(["a", "a"], [0.2, 0.4], [0, 0], how="max")
    assert s_max[0] == 0.4


def test_config_overrides():
    cfg = apply_overrides({"train": {"epochs": 8}}, ["train.epochs=1", "model.backbone=efficientnet_b0"])
    assert cfg == {"train": {"epochs": 1}, "model": {"backbone": "efficientnet_b0"}}


def test_filter_existing_keeps_only_downloaded_clips(tmp_path):
    (tmp_path / "DF_E_1.flac").write_bytes(b"")
    utts = [asvspoof.parse_line(f"LA_0001 DF_E_{i} nocodec asvspoof A01 spoof notrim eval x - - - -") for i in (1, 2)]
    kept = asvspoof.filter_existing(utts, tmp_path)
    assert [u.utt_id for u in kept] == ["DF_E_1"]


def test_chunked_scoring_resume_and_merge(tmp_path):
    from forensight.evaluation import chunked

    keys = [f"DF_E_{i}" for i in range(10)]
    ranges = chunked.chunk_ranges(len(keys), 4)
    assert ranges == [(0, 4), (4, 8), (8, 10)]
    run = tmp_path / chunked.key_hash(keys)
    for i, (s, t) in enumerate(ranges):
        chunked.save_chunk(run, i, keys[s:t], np.linspace(0, 1, t - s), [i % 2] * (t - s))
    k, p, y = chunked.merge_chunks(run, len(ranges), keys)
    assert k == keys and len(p) == 10 and y.dtype.kind == "i"
    assert not list(run.glob(".tmp_*"))  # no half-written files left behind
    assert chunked.key_hash(keys) != chunked.key_hash(keys[:-1])
    with pytest.raises(ValueError):
        chunked.merge_chunks(run, len(ranges), keys[::-1])
    chunked.chunk_path(run, 1).unlink()
    with pytest.raises(FileNotFoundError):
        chunked.merge_chunks(run, len(ranges), keys)


def test_preprocess_split_resumes_and_archives(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from forensight.data import video_preprocess as vp

    calls = []
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    monkeypatch.setattr(vp, "read_frames", lambda path, n: calls.append(path) or [(0, frame), (5, frame)])
    cropper = SimpleNamespace(crop=lambda f, size: np.zeros((size, size, 3), dtype=np.uint8))
    items = [SimpleNamespace(path=f"v{i}.mp4", label=i % 2, method="m", video_id=f"id{i}") for i in range(3)]

    out = tmp_path / "frames" / "train"
    vp.preprocess_split(items, out, 2, 16, cropper)
    assert len(calls) == 3
    # second run: every video has a .done marker, so nothing is decoded again
    vp.preprocess_split(items, out, 2, 16, cropper)
    assert len(calls) == 3
    rows = (out / "index.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 3 * 2

    archive = tmp_path / "drive" / "train.tar"
    vp.archive_split(out, archive)
    assert archive.exists() and not archive.with_name("train.tar.part").exists()
    restored = tmp_path / "restored"
    vp.restore_split(archive, restored)
    assert (restored / "train" / "index.csv").read_text() == (out / "index.csv").read_text()


def test_face_cropper_falls_back_and_reports_backend():
    class StubCropper(FaceCropper):
        def _init_stub(self):  # stands in for a real detector; reports one 50x40 face
            return None

        def detect(self, frame_bgr):
            return (10, 20, 50, 40)

    cropper = StubCropper(backends=("nope", "stub"))
    assert cropper.backend == "stub"
    assert cropper.load_errors and cropper.load_errors[0].startswith("nope")
    assert cropper.crop(np.zeros((240, 320, 3), np.uint8), size=64).shape == (64, 64, 3)


def test_face_cropper_raises_when_nothing_loads():
    with pytest.raises(RuntimeError, match="No face detector"):
        FaceCropper(backends=("nope",))
