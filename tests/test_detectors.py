from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytest

from backend.services import device as device_mod
from backend.services.detectors import (
    BALL,
    COCO_CLASS_MAP,
    PERSON,
    Detections,
    Detector,
    GroundTruthDetector,
    TiledBallDetector,
    UltralyticsDetector,
    class_map_from_names,
    iou_matrix,
    nms,
    tensorrt_engine_path,
    tile_grid,
)
from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS = REPO_ROOT / "yolov8n.pt"


@pytest.fixture(scope="module")
def truth(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("detectors")
    spec = SyntheticMatchSpec(width=1280, height=720, duration_s=3.0, fps=25.0, seed=5, goals=[])
    return generate_synthetic_match(root / "m.mp4", spec)


# ----------------------------------------------------------------------
# Detections / box helpers
# ----------------------------------------------------------------------


def test_detections_validation_and_helpers() -> None:
    det = Detections([[0, 0, 10, 20], [5, 5, 7, 7]], [0.9, 0.4], [PERSON, BALL], frame_index=3)
    assert len(det) == 2
    assert det.xyxy.dtype == np.float32 and det.cls.dtype == np.int64
    assert len(det.of_class(BALL)) == 1
    assert det.scaled(2.0).xyxy[0].tolist() == [0, 0, 20, 40]
    assert det.translated(100, 10).xyxy[1].tolist() == [105, 15, 107, 17]
    assert len(Detections.concat([det, Detections.empty()])) == 2
    with pytest.raises(ValueError):
        Detections([[0, 0, 1, 1]], [0.5, 0.6], [0])


def test_iou_and_nms() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    ious = iou_matrix(boxes, boxes)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(81 / 119, rel=1e-4)
    assert ious[0, 2] == 0.0
    keep = nms(boxes, np.array([0.5, 0.9, 0.3]), 0.5)
    assert keep.tolist() == [1, 2]
    assert iou_matrix(np.zeros((0, 4)), boxes).shape == (0, 3)


def test_class_map_from_model_names() -> None:
    coco = {0: "person", 1: "bicycle", 32: "sports ball"}
    assert class_map_from_names(coco) == COCO_CLASS_MAP
    custom = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    assert class_map_from_names(custom) == {0: BALL, 1: PERSON, 2: PERSON, 3: PERSON}


def test_tensorrt_engine_cache_name() -> None:
    path = tensorrt_engine_path("/models/yolov8s.pt", 1280, True, 16)
    assert path == Path("/models/yolov8s_1280_fp16_b16.engine")
    assert tensorrt_engine_path("m.pt", 960, False, 1).name == "m_960_fp32_b1.engine"


# ----------------------------------------------------------------------
# GroundTruthDetector
# ----------------------------------------------------------------------


def test_ground_truth_detector_is_deterministic_and_batch_independent(truth) -> None:
    det = GroundTruthDetector(truth.tracking, seed=4)
    assert isinstance(det, Detector)
    frames = [np.zeros((360, 640, 3), np.uint8)] * 6
    a = det.detect_batch(frames, frame_indices=list(range(10, 16)))
    b = det.detect_batch(frames[:2], frame_indices=[10, 11]) + det.detect_batch(frames[2:], frame_indices=[12, 13, 14, 15])
    for x, y in zip(a, b):
        assert np.array_equal(x.xyxy, y.xyxy) and np.array_equal(x.conf, y.conf)
    assert a[0].frame_index == 10


def test_ground_truth_detector_noise_model(truth) -> None:
    clean = GroundTruthDetector(truth.tracking, dropout=0.0, jitter_px=0.0, false_positive_rate=0.0,
                                ball_dropout=0.0, occlusion_iou=None)
    frame = np.zeros((720, 1280, 3), np.uint8)
    det = clean.detect_batch([frame], frame_indices=[0])[0]
    persons = det.of_class(PERSON)
    assert len(persons) == len(truth.tracking.players)
    first = truth.tracking.players[1]
    assert np.any(np.all(np.isclose(persons.xyxy, [first.x1[0], first.y1[0], first.x2[0], first.y2[0]]), axis=1))
    assert len(det.of_class(BALL)) == 1

    # Detections follow the input frame's pixel space (proxy at half size).
    half = clean.detect_batch([np.zeros((360, 640, 3), np.uint8)], frame_indices=[0])[0]
    assert np.allclose(np.sort(half.of_class(PERSON).xyxy, axis=0), np.sort(persons.xyxy, axis=0) / 2, atol=1e-3)

    noisy = GroundTruthDetector(truth.tracking, dropout=0.3, jitter_px=2.0, false_positive_rate=0.2, seed=1)
    frames = [frame] * 40
    dets = noisy.detect_batch(frames, frame_indices=list(range(40)))
    confs = np.concatenate([d.of_class(PERSON).conf for d in dets])
    n_true = 40 * len(truth.tracking.players)
    n_hi = int((confs >= 0.5).sum())
    assert 0.55 * n_true < n_hi < 0.8 * n_true  # ~30 % dropout (+ occlusion)
    assert int((confs < 0.5).sum()) > 0  # some low-confidence false positives
    assert all(set(d.cls.tolist()) <= {PERSON, BALL} for d in dets)


# ----------------------------------------------------------------------
# TiledBallDetector
# ----------------------------------------------------------------------


class _BlobDetector:
    """Fake detector: bright blobs. Balls (small blobs) are only found on
    inputs no wider than ``ball_visible_max_w`` (mimics a tiny ball vanishing
    when a 4K frame is letterboxed down to the model size)."""

    def __init__(self, ball_visible_max_w: int = 1300) -> None:
        self.batch = 4
        self.max_w = ball_visible_max_w
        self.calls: List[int] = []

    def info(self) -> Dict[str, object]:
        return {"name": "blob"}

    def detect_batch(self, frames: List[np.ndarray], *, frame_indices: Sequence[int]) -> List[Detections]:
        import cv2

        self.calls.append(len(frames))
        out = []
        for frame, idx in zip(frames, frame_indices):
            gray = frame[:, :, 0]
            n, _labels, stats, _c = cv2.connectedComponentsWithStats((gray > 127).astype(np.uint8))
            boxes, confs, classes = [], [], []
            for k in range(1, n):
                x, y, w, h, area = stats[k]
                is_ball = w < 20 and h < 20
                if is_ball and frame.shape[1] > self.max_w:
                    continue
                boxes.append([x, y, x + w, y + h])
                confs.append(0.8 if is_ball else 0.9)
                classes.append(BALL if is_ball else PERSON)
            out.append(Detections(np.asarray(boxes, np.float32).reshape(-1, 4), confs, classes, idx))
        return out


def test_tile_grid_covers_frame_with_overlap() -> None:
    tiles = tile_grid(3840, 2160, 1280, 0.15)
    xs = sorted({t[0] for t in tiles})
    assert xs[0] == 0 and xs[-1] + 1280 == 3840
    assert all(b - a <= int(1280 * 0.85) for a, b in zip(xs, xs[1:]))
    assert all(x1 - x0 == 1280 and y1 - y0 == 1280 for x0, y0, x1, y1 in tiles)
    assert tile_grid(640, 360, 1280, 0.15) == [(0, 0, 640, 360)]


def test_tiled_ball_detector_finds_tiny_ball_once_in_frame_coordinates() -> None:
    frame = np.zeros((1080, 2400, 3), np.uint8)
    frame[300:400, 100:140] = 255  # person-sized blob
    tiles = tile_grid(2400, 1080, 1280, 0.15)
    # Put the ball inside the overlap of the first two tiles.
    overlap_x0 = tiles[1][0]
    bx, by = overlap_x0 + 40, 700
    frame[by:by + 8, bx:bx + 8] = 255

    base = _BlobDetector()
    plain = base.detect_batch([frame], frame_indices=[7])[0]
    assert len(plain.of_class(BALL)) == 0  # the full frame misses the ball

    tiled = TiledBallDetector(base, tile=1280, overlap=0.15)
    det = tiled.detect_batch([frame], frame_indices=[7])[0]
    balls = det.of_class(BALL)
    assert len(balls) == 1  # seen by two tiles, merged by NMS
    assert balls.xyxy[0].tolist() == [bx, by, bx + 8, by + 8]
    persons = det.of_class(PERSON)
    assert len(persons) == 1  # persons come from the full-frame pass only
    assert persons.xyxy[0].tolist() == [100, 300, 140, 400]
    assert det.frame_index == 7
    assert tiled.info()["ball_tiles"]["tile"] == 1280


def test_tiled_ball_detector_passthrough_for_small_frames() -> None:
    base = _BlobDetector(ball_visible_max_w=10_000)
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[100:106, 100:106] = 255
    det = TiledBallDetector(base, tile=1280).detect_batch([frame], frame_indices=[0])[0]
    assert len(det.of_class(BALL)) == 1
    assert base.calls == [1]  # no tile pass


# ----------------------------------------------------------------------
# Device selection / ffmpeg capability probes
# ----------------------------------------------------------------------


def test_select_device_cpu_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(device_mod.DEVICE_ENV, raising=False)
    cpu = device_mod.select_device("cpu")
    assert cpu.kind == "cpu" and cpu.torch_device == "cpu"
    assert cpu.recommended_batch(1280) == 1
    monkeypatch.setenv(device_mod.DEVICE_ENV, "cpu")
    assert device_mod.select_device("cuda").kind == "cpu"
    monkeypatch.setenv(device_mod.DEVICE_ENV, "cuda:1")
    info = device_mod.select_device("auto")
    assert info.requested == "cuda:1"
    assert info.kind in {"cuda", "mps", "cpu"}  # falls back when CUDA is absent


def test_recommended_batch_heuristics() -> None:
    def cuda(vram: float) -> device_mod.DeviceInfo:
        return device_mod.DeviceInfo(kind="cuda", name="gpu", vram_gb=vram, supports_half=True)

    assert cuda(8).recommended_batch(1280) == 8
    assert cuda(16).recommended_batch(1280) == 16
    assert cuda(24).recommended_batch(1280) == 24
    assert cuda(128).recommended_batch(1280) == 32  # DGX Spark unified memory
    assert cuda(16).recommended_batch(640) > cuda(16).recommended_batch(1280)
    assert cuda(8).recommended_batch(1536) >= 1
    mps = device_mod.DeviceInfo(kind="mps", name="m")
    assert mps.recommended_batch(1280) == 4
    assert cuda(12).torch_device == "cuda:0"


def test_encoder_and_hwaccel_preferences() -> None:
    prefs = device_mod.encoder_preferences()
    assert prefs[-1] == "libx264"
    assert prefs.index("libx264") > 0
    assert device_mod.hwaccel_preferences()
    if shutil.which("ffmpeg"):
        enc = device_mod.pick_encoder()
        assert enc in prefs
        # Cached: the second call must not re-probe.
        assert device_mod.pick_encoder() == enc
        assert device_mod.pick_encoder.cache_info().hits >= 1
        assert device_mod.ffmpeg_version() >= (4, 0)
        accel = device_mod.pick_hwaccel()
        assert accel is None or accel in device_mod.hwaccel_preferences()


# ----------------------------------------------------------------------
# Real YOLO smoke test (opt-in: needs ultralytics + repo weights)
# ----------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None or not YOLO_WEIGHTS.is_file(),
                    reason="ultralytics or yolov8n.pt unavailable")
def test_ultralytics_detector_smoke_on_proxy_frames(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from backend.services.frame_source import FrameReader, build_proxy

    spec = SyntheticMatchSpec(width=1280, height=720, duration_s=1.0, fps=25.0, seed=2, goals=[])
    generate_synthetic_match(tmp_path / "m.mp4", spec)
    proxy = build_proxy(tmp_path / "m.mp4", tmp_path / "run", height=360, thumbs=False)
    with FrameReader(proxy.path, end_frame=10) as reader:
        items = list(reader)
    det = UltralyticsDetector(str(YOLO_WEIGHTS), imgsz=640, conf=0.2, device="cpu", batch=4)
    out = det.detect_batch([it.frame for it in items], frame_indices=[it.index for it in items])
    assert len(out) == len(items) == 10
    for d, it in zip(out, items):
        assert isinstance(d, Detections)
        assert d.frame_index == it.index
        assert set(d.cls.tolist()) <= {PERSON, BALL}
        if len(d):
            assert d.xyxy[:, 2].max() <= proxy.width + 1 and d.xyxy[:, 3].max() <= proxy.height + 1
    info = det.info()
    assert info["device"] == "cpu" and info["batch"] == 4 and info["half"] is False
    assert info["class_map"] == {"0": PERSON, "32": BALL}


def test_precision_kwargs_and_weight_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.services.detectors import precision_kwargs, resolve_weights

    assert precision_kwargs(False) == {}
    fp16 = precision_kwargs(True)
    assert fp16 in ({"quantize": 16}, {"half": True})
    existing = tmp_path / "custom.pt"
    existing.write_bytes(b"x")
    assert resolve_weights(existing) == str(existing)
    # Bare names go through perf_profiles (VH_MODEL_DIR, then repo root).
    resolved = resolve_weights("yolov8n.pt")
    assert Path(resolved).name == "yolov8n.pt"
    if YOLO_WEIGHTS.is_file() and Path(resolved).is_absolute():
        assert Path(resolved).is_file()
