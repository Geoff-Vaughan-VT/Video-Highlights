"""Object detectors behind one small protocol.

Every detector maps a batch of BGR frames to :class:`Detections` whose
classes are normalised to :data:`PERSON` / :data:`BALL` regardless of the
underlying model's label map, and whose boxes are in **the pixel space of
the frame that was passed in** (the tracking engine rescales to source
pixels).

Implementations
---------------
* :class:`UltralyticsDetector`: YOLO via ``model.predict`` on a list of
  frames (batched, fp16 on CUDA, optional cached TensorRT engine).
* :class:`GroundTruthDetector`: noisy detections derived from a known
  :class:`~backend.services.tracking_types.TrackingResult` (synthetic match)
  so the tracker can be tested without a neural network.
* :class:`TiledBallDetector`: wraps another detector and adds a tiled,
  ball-only pass for tiny balls on high-resolution frames.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .tracking_types import TrackingResult

logger = logging.getLogger("videohighlights.detectors")

PERSON = 0
BALL = 1
CLASS_NAMES = {PERSON: "person", BALL: "ball"}

#: COCO ids -> normalised classes (0 person, 32 sports ball).
COCO_CLASS_MAP: Dict[int, int] = {0: PERSON, 32: BALL}

_PERSON_NAMES = {"person", "player", "players", "goalkeeper", "referee", "people"}
_BALL_NAMES = {"sports ball", "ball", "football", "soccer ball", "soccer-ball", "sports-ball"}


@dataclass
class Detections:
    """Detections for one frame (numpy, frame pixel space)."""

    xyxy: np.ndarray  # float32 [n, 4]
    conf: np.ndarray  # float32 [n]
    cls: np.ndarray  # int64 [n] in {PERSON, BALL}
    frame_index: int = -1

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(self.conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(self.cls, dtype=np.int64).reshape(-1)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError("xyxy, conf and cls must have the same length")

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    @staticmethod
    def empty(frame_index: int = -1) -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), frame_index)

    def select(self, mask: np.ndarray) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask], self.frame_index)

    def of_class(self, cls_id: int) -> "Detections":
        return self.select(self.cls == int(cls_id))

    def scaled(self, sx: float, sy: Optional[float] = None) -> "Detections":
        sy = sx if sy is None else sy
        factor = np.array([sx, sy, sx, sy], dtype=np.float32)
        return Detections(self.xyxy * factor, self.conf.copy(), self.cls.copy(), self.frame_index)

    def translated(self, dx: float, dy: float) -> "Detections":
        offset = np.array([dx, dy, dx, dy], dtype=np.float32)
        return Detections(self.xyxy + offset, self.conf.copy(), self.cls.copy(), self.frame_index)

    @staticmethod
    def concat(items: Sequence["Detections"], frame_index: int = -1) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return Detections.empty(frame_index)
        return Detections(
            np.concatenate([d.xyxy for d in items]),
            np.concatenate([d.conf for d in items]),
            np.concatenate([d.cls for d in items]),
            frame_index,
        )


@runtime_checkable
class Detector(Protocol):
    """Anything that turns frames into normalised :class:`Detections`."""

    #: preferred batch size for :func:`detect_batch` calls
    batch: int

    def detect_batch(self, frames: List[np.ndarray], *, frame_indices: Sequence[int]) -> List[Detections]:
        ...

    def info(self) -> Dict[str, object]:
        ...


# ----------------------------------------------------------------------
# Box helpers
# ----------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``[n,4]`` and ``[m,4]`` xyxy boxes."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(xyxy: np.ndarray, conf: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending confidence."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    order = np.argsort(-conf, kind="stable")
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(xyxy[i:i + 1], xyxy[order[1:]])[0]
        order = order[1:][ious < iou_thresh]
    return np.asarray(keep, dtype=np.int64)


# ----------------------------------------------------------------------
# Ultralytics YOLO
# ----------------------------------------------------------------------


def class_map_from_names(names: Mapping[int, str]) -> Dict[int, int]:
    """Derive ``{model_class_id: PERSON|BALL}`` from a model's label names."""
    mapping: Dict[int, int] = {}
    for idx, name in (names or {}).items():
        key = str(name).strip().lower()
        if key in _PERSON_NAMES:
            mapping[int(idx)] = PERSON
        elif key in _BALL_NAMES:
            mapping[int(idx)] = BALL
    return mapping


@lru_cache(maxsize=1)
def _ultralytics_uses_quantize() -> bool:
    """True when this ultralytics has the unified ``quantize`` arg (8.4+; ``half`` is deprecated)."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        return "quantize" in DEFAULT_CFG_DICT
    except Exception:  # pragma: no cover - very old ultralytics
        return False


def precision_kwargs(half: bool) -> Dict[str, object]:
    """``predict``/``export`` kwargs for fp16 (``quantize=16`` or legacy ``half=True``); empty for fp32."""
    if not half:
        return {}
    return {"quantize": 16} if _ultralytics_uses_quantize() else {"half": True}


def resolve_weights(model_path: str | Path) -> str:
    """Existing path as given, else ``perf_profiles.resolve_model_path`` (VH_MODEL_DIR), else the bare name."""
    text = str(model_path)
    if Path(text).exists():
        return text
    try:
        from .perf_profiles import resolve_model_path
    except Exception:  # pragma: no cover - module absent in older checkouts
        return text
    try:
        return str(resolve_model_path(text))
    except Exception as exc:  # pragma: no cover
        logger.warning("resolve_model_path(%s) failed: %s", text, exc)
        return text


def tensorrt_engine_path(weights: str | Path, imgsz: int, half: bool, batch: int) -> Path:
    """Cache path for an exported engine: ``<stem>_<imgsz>_<fp16|fp32>_b<batch>.engine``."""
    weights = Path(weights)
    precision = "fp16" if half else "fp32"
    return weights.with_name(f"{weights.stem}_{int(imgsz)}_{precision}_b{int(batch)}.engine")


class UltralyticsDetector:
    """YOLO detector using batched ``model.predict`` (never ``model.track``).

    Args:
        model_path: ``.pt`` weights (or an ``.engine``/``.onnx`` export).
        imgsz: inference size (letterboxed long side).
        conf: person confidence threshold.
        device: ``"cuda:0"``, ``"mps"``, ``"cpu"`` or a
            :class:`~backend.services.device.DeviceInfo`; ``None``/``"auto"``
            selects via :func:`~backend.services.device.select_device`.
        half: fp16 inference (``None`` -> on when the device supports it).
        batch: frames per ``predict`` call (``None`` -> device heuristic).
        class_map: ``{model_class_id: PERSON|BALL}``; default derives from
            the model's names (COCO: 0 -> PERSON, 32 -> BALL).
        ball_conf: separate (lower) threshold for the small ball class.
        use_tensorrt: on CUDA, export once to a cached ``.engine`` next to
            the weights and load it; any failure falls back to the weights.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        *,
        imgsz: int = 1280,
        conf: float = 0.25,
        device: object = None,
        half: Optional[bool] = None,
        batch: Optional[int] = None,
        class_map: Optional[Mapping[int, int]] = None,
        ball_conf: Optional[float] = None,
        iou: float = 0.6,
        max_det: int = 300,
        use_tensorrt: bool = False,
    ) -> None:
        from .device import DeviceInfo, select_device

        if isinstance(device, DeviceInfo):
            dev = device
        else:
            dev = select_device(str(device) if device not in (None, "") else "auto")
        self.device_info = dev
        self.device = dev.torch_device
        self.model_path = resolve_weights(model_path)
        self.imgsz = max(160, int(imgsz))
        self.conf = float(conf)
        self.ball_conf = float(ball_conf) if ball_conf is not None else min(self.conf, 0.15)
        self.half = bool(dev.supports_half) if half is None else bool(half and dev.kind == "cuda")
        self.batch = max(1, int(batch)) if batch else dev.recommended_batch(self.imgsz)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.use_tensorrt = bool(use_tensorrt)
        self.engine_path: Optional[str] = None
        self._model = None
        self._explicit_class_map = dict(class_map) if class_map else None
        self.class_map: Dict[int, int] = dict(class_map) if class_map else dict(COCO_CLASS_MAP)
        self.predict_s = 0.0
        self.frames = 0

    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        model = None
        if self.use_tensorrt and self.device_info.kind == "cuda" and not self.model_path.endswith(".engine"):
            model = self._load_tensorrt(YOLO)
        if model is None:
            model = YOLO(self.model_path)
        names = getattr(model, "names", None) or {}
        if self._explicit_class_map is None and names:
            derived = class_map_from_names(names)
            if derived:
                self.class_map = derived
        if self.device_info.kind == "cuda":
            try:
                import torch

                torch.backends.cudnn.benchmark = True
            except Exception:  # pragma: no cover
                pass
        self._model = model
        logger.info("Loaded detector %s on %s (imgsz=%d, half=%s, batch=%d, classes=%s)",
                    self.engine_path or self.model_path, self.device, self.imgsz, self.half, self.batch,
                    self.class_map)
        return model

    def _load_tensorrt(self, yolo_cls):
        engine = tensorrt_engine_path(self.model_path, self.imgsz, self.half, self.batch)
        try:
            if not engine.is_file():
                logger.info("Exporting TensorRT engine %s (one-time)", engine)
                base = yolo_cls(self.model_path)
                exported = base.export(format="engine", imgsz=self.imgsz, batch=self.batch, dynamic=True,
                                       device=self.device, verbose=False, **precision_kwargs(self.half))
                exported_path = Path(str(exported))
                if exported_path.resolve() != engine.resolve():
                    shutil.move(str(exported_path), str(engine))
            model = yolo_cls(str(engine), task="detect")
            self.engine_path = str(engine)
            return model
        except Exception as exc:  # pragma: no cover - needs TensorRT
            logger.warning("TensorRT unavailable (%s); using %s", exc, self.model_path)
            self.engine_path = None
            return None

    def info(self) -> Dict[str, object]:
        return {
            "name": "ultralytics",
            "model": Path(self.model_path).name,
            "engine": Path(self.engine_path).name if self.engine_path else None,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "ball_conf": self.ball_conf,
            "half": self.half,
            "batch": self.batch,
            "device": self.device,
            "device_name": self.device_info.name,
            "class_map": {str(k): v for k, v in self.class_map.items()},
        }

    def detect_batch(self, frames: List[np.ndarray], *, frame_indices: Sequence[int]) -> List[Detections]:
        if not frames:
            return []
        model = self._load()
        indices = list(frame_indices) if frame_indices is not None else list(range(len(frames)))
        out: List[Detections] = []
        t0 = time.perf_counter()
        for start in range(0, len(frames), self.batch):
            chunk = frames[start:start + self.batch]
            results = model.predict(
                chunk,
                imgsz=self.imgsz,
                conf=min(self.conf, self.ball_conf),
                iou=self.iou,
                classes=sorted(self.class_map.keys()),
                device=self.device,
                max_det=self.max_det,
                batch=len(chunk),
                verbose=False,
                **precision_kwargs(self.half),
            )
            for k, res in enumerate(results):
                out.append(self._convert(res, indices[start + k]))
        self.predict_s += time.perf_counter() - t0
        self.frames += len(frames)
        return out

    def _convert(self, result, frame_index: int) -> Detections:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return Detections.empty(frame_index)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        raw_cls = boxes.cls.cpu().numpy().astype(np.int64)
        mapped = np.array([self.class_map.get(int(c), -1) for c in raw_cls], dtype=np.int64)
        threshold = np.where(mapped == BALL, self.ball_conf, self.conf)
        keep = (mapped >= 0) & (conf >= threshold)
        return Detections(xyxy[keep], conf[keep], mapped[keep], frame_index)


# ----------------------------------------------------------------------
# Ground-truth detector (tests / benchmarks)
# ----------------------------------------------------------------------


class GroundTruthDetector:
    """Noisy detections sampled from a known :class:`TrackingResult`.

    Per frame (deterministic in ``(seed, frame_index)``, independent of
    batching): each true player box is dropped with probability
    ``dropout``, otherwise jittered by ``jitter_px`` (Gaussian sigma, frame
    pixels) and given a confidence in ``[0.5, 0.95]``; with probability
    ``false_positive_rate`` *per true player* a spurious person-sized box with
    confidence in ``[0.1, 0.45]`` is added. The ball (when visible in truth)
    is detected with probability ``1 - ball_dropout``. Like a real detector,
    a player whose box overlaps a nearer player's box (larger ``y2``) with
    IoU >= ``occlusion_iou`` is not detected (``None`` disables this).

    Truth coordinates are in the truth's ``frame_size`` (source pixels) and
    ``jitter_px`` is in that same space; both are scaled to each input
    frame's actual size so proxies of any height see the same relative noise.
    """

    def __init__(
        self,
        tracking_result: TrackingResult,
        *,
        dropout: float = 0.1,
        jitter_px: float = 2.0,
        false_positive_rate: float = 0.02,
        ball_dropout: float = 0.1,
        occlusion_iou: Optional[float] = 0.6,
        seed: int = 0,
        batch: int = 8,
    ) -> None:
        self.truth = tracking_result
        self.occlusion_iou = occlusion_iou
        self.dropout = float(dropout)
        self.jitter_px = float(jitter_px)
        self.false_positive_rate = float(false_positive_rate)
        self.ball_dropout = float(ball_dropout)
        self.seed = int(seed)
        self.batch = max(1, int(batch))
        self._tracks = list(tracking_result.players.values())
        self._half_frame = 0.5 / max(1e-6, float(tracking_result.fps))
        ball = tracking_result.ball
        self._ball_t = np.asarray(ball.t, dtype=np.float64)

    def info(self) -> Dict[str, object]:
        return {
            "name": "ground_truth",
            "dropout": self.dropout,
            "jitter_px": self.jitter_px,
            "false_positive_rate": self.false_positive_rate,
            "seed": self.seed,
            "batch": self.batch,
        }

    def _ball_box(self, t: float) -> Optional[Tuple[float, float, float, float]]:
        if not self._ball_t.size:
            return None
        idx = int(np.searchsorted(self._ball_t, t))
        best = None
        for i in (idx - 1, idx):
            if 0 <= i < self._ball_t.size and abs(self._ball_t[i] - t) <= self._half_frame + 1e-6:
                best = i
        if best is None:
            return None
        b = self.truth.ball
        return float(b.x[best]), float(b.y[best]), float(b.w[best]), float(b.h[best])

    def detect_one(self, frame_shape: Tuple[int, ...], frame_index: int) -> Detections:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        sx = w / float(self.truth.frame_width)
        sy = h / float(self.truth.frame_height)
        rng = np.random.default_rng((self.seed, int(frame_index)))
        t = int(frame_index) / float(self.truth.fps)
        boxes: List[Tuple[float, float, float, float]] = []
        confs: List[float] = []
        classes: List[int] = []
        sizes: List[Tuple[float, float]] = []
        truth_boxes: List[Tuple[float, float, float, float]] = []
        for track in self._tracks:
            box = track.box_at(t, tolerance_s=self._half_frame + 1e-6)
            if box is not None:
                truth_boxes.append((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy))
        occluded = np.zeros(len(truth_boxes), dtype=bool)
        if self.occlusion_iou is not None and len(truth_boxes) > 1:
            tb = np.asarray(truth_boxes, dtype=np.float32)
            ious = iou_matrix(tb, tb)
            np.fill_diagonal(ious, 0.0)
            nearer = tb[None, :, 3] > tb[:, None, 3]  # [i, j]: j is in front of i
            occluded = ((ious >= float(self.occlusion_iou)) & nearer).any(axis=1)
        for k, (x1, y1, x2, y2) in enumerate(truth_boxes):
            sizes.append((x2 - x1, y2 - y1))
            drop = rng.random() < self.dropout
            if drop or occluded[k]:
                continue
            jit = rng.normal(0.0, self.jitter_px, size=4) * (sx, sy, sx, sy) if self.jitter_px > 0 else np.zeros(4)
            boxes.append((x1 + jit[0], y1 + jit[1], x2 + jit[2], y2 + jit[3]))
            confs.append(float(rng.uniform(0.5, 0.95)))
            classes.append(PERSON)
        n_fp = int(rng.binomial(len(sizes), min(1.0, max(0.0, self.false_positive_rate)))) if sizes else 0
        for _ in range(n_fp):
            bw, bh = sizes[int(rng.integers(0, len(sizes)))]
            cx = float(rng.uniform(bw, max(bw + 1.0, w - bw)))
            cy = float(rng.uniform(bh, max(bh + 1.0, h - 1.0)))
            boxes.append((cx - bw / 2, cy - bh, cx + bw / 2, cy))
            confs.append(float(rng.uniform(0.1, 0.45)))
            classes.append(PERSON)
        ball = self._ball_box(t)
        if ball is not None and rng.random() >= self.ball_dropout:
            bx, by, bw, bh = ball[0] * sx, ball[1] * sy, max(1.0, ball[2] * sx), max(1.0, ball[3] * sy)
            jit = rng.normal(0.0, self.jitter_px * 0.5, size=2) * (sx, sy) if self.jitter_px > 0 else np.zeros(2)
            bx, by = bx + jit[0], by + jit[1]
            boxes.append((bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2))
            confs.append(float(rng.uniform(0.3, 0.9)))
            classes.append(BALL)
        if not boxes:
            return Detections.empty(frame_index)
        arr = np.asarray(boxes, dtype=np.float32)
        arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0, w - 1)
        arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, h - 1)
        return Detections(arr, np.asarray(confs, np.float32), np.asarray(classes, np.int64), int(frame_index))

    def detect_batch(self, frames: List[np.ndarray], *, frame_indices: Sequence[int]) -> List[Detections]:
        return [self.detect_one(f.shape, int(i)) for f, i in zip(frames, frame_indices)]


# ----------------------------------------------------------------------
# Tiled ball pass
# ----------------------------------------------------------------------


def tile_grid(width: int, height: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Cover a ``width x height`` frame with ``tile``-sized windows.

    Returns ``(x0, y0, x1, y1)`` windows; neighbours overlap by at least
    ``overlap * tile`` pixels and the last window in each axis is flush with
    the frame edge.
    """
    tile = max(32, int(tile))

    def starts(size: int) -> List[int]:
        if size <= tile:
            return [0]
        step = max(1, int(tile * (1.0 - max(0.0, min(0.9, overlap)))))
        out = list(range(0, size - tile, step))
        out.append(size - tile)
        return sorted(set(out))

    tw, th = min(tile, width), min(tile, height)
    return [(x, y, x + tw, y + th) for y in starts(height) for x in starts(width)]


class TiledBallDetector:
    """Full-frame detection plus a tiled, ball-only high-resolution pass.

    The base detector runs once on each full frame (all classes) and once
    on every tile (ball detections only are kept from tiles). Tile boxes are
    shifted to frame coordinates and merged with the full-frame balls by
    NMS. Frames no larger than ``tile`` are not tiled.
    """

    def __init__(self, base_detector: Detector, *, tile: int = 1280, overlap: float = 0.15,
                 nms_iou: float = 0.4) -> None:
        self.base = base_detector
        self.tile = int(tile)
        self.overlap = float(overlap)
        self.nms_iou = float(nms_iou)
        self.batch = int(getattr(base_detector, "batch", 1) or 1)

    def info(self) -> Dict[str, object]:
        base_info = self.base.info() if hasattr(self.base, "info") else {}
        return {**base_info, "ball_tiles": {"tile": self.tile, "overlap": self.overlap}}

    def detect_batch(self, frames: List[np.ndarray], *, frame_indices: Sequence[int]) -> List[Detections]:
        indices = list(frame_indices)
        full = self.base.detect_batch(frames, frame_indices=indices)
        tile_frames: List[np.ndarray] = []
        tile_owner: List[Tuple[int, int, int]] = []  # (frame k, x0, y0)
        for k, frame in enumerate(frames):
            h, w = frame.shape[:2]
            if w <= self.tile and h <= self.tile:
                continue
            for x0, y0, x1, y1 in tile_grid(w, h, self.tile, self.overlap):
                tile_frames.append(np.ascontiguousarray(frame[y0:y1, x0:x1]))
                tile_owner.append((k, x0, y0))
        if not tile_frames:
            return full
        tile_dets = self.base.detect_batch(tile_frames, frame_indices=[indices[k] for k, _, _ in tile_owner])
        extra: Dict[int, List[Detections]] = {}
        for (k, x0, y0), det in zip(tile_owner, tile_dets):
            balls = det.of_class(BALL)
            if len(balls):
                extra.setdefault(k, []).append(balls.translated(x0, y0))
        merged: List[Detections] = []
        for k, det in enumerate(full):
            if k not in extra:
                merged.append(det)
                continue
            persons = det.select(det.cls != BALL)
            balls = Detections.concat([det.of_class(BALL)] + extra[k], frame_index=indices[k])
            keep = nms(balls.xyxy, balls.conf, self.nms_iou)
            merged.append(Detections.concat([persons, balls.select(keep)], frame_index=indices[k]))
        return merged


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_detector(
    model_path: str | Path,
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    device: object = None,
    half: Optional[bool] = None,
    batch: Optional[int] = None,
    use_tensorrt: Optional[bool] = None,
    ball_tiles: bool = False,
    tile: int = 1280,
    class_map: Optional[Mapping[int, int]] = None,
) -> Detector:
    """Convenience factory used by the pipeline (profile keys map 1:1).

    ``use_tensorrt=None`` enables TensorRT only when ``VH_TENSORRT=1``.
    """
    if use_tensorrt is None:
        use_tensorrt = os.environ.get("VH_TENSORRT", "").strip().lower() in ("1", "true", "yes", "on")
    det: Detector = UltralyticsDetector(
        model_path, imgsz=imgsz, conf=conf, device=device, half=half, batch=batch,
        use_tensorrt=bool(use_tensorrt), class_map=class_map,
    )
    if ball_tiles:
        det = TiledBallDetector(det, tile=tile)
    return det
