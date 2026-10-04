"""Processing profiles, model resolution, hardware classes and runtime estimates.

Three things live here so the API, the engine, the bench and the docs agree:

1. ``PROFILES`` (fast / balanced / quality): the job-config defaults from
   ``docs/ARTIFACTS.md``. ``resolve_job_config`` lays them *under* the keys a
   caller set explicitly, so ``{"profile": "fast", "inference_imgsz": 1280}``
   keeps 1280.
2. Model resolution: ``resolve_model_path`` finds detector weights in
   ``VH_MODEL_DIR`` (``/models`` in the containers) before falling back to an
   Ultralytics download by name. ``VH_MODEL_FAMILY=yolo26`` swaps the stock
   YOLOv8 profile weights for YOLO26 of the same size.
3. ``estimate_runtime``: a transparent per-stage throughput model (proxy pass,
   detection+tracking, analysis, final render, clips+reel) per hardware class.
   Every number it uses is in ``HARDWARE_CLASSES`` / ``STAGE_MODEL`` below, and
   the bench (``bench/bench_match.py``) can override any stage with measured
   fps via ``measured=...``.

Nothing here imports torch, ultralytics or subprocess at module import time.
"""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = "balanced"

PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "proxy_height": 720,
        "inference_imgsz": 960,
        "vid_stride": 2,
        "yolo_model": "yolov8n.pt",
        "batch_size": "auto",
        "output_height": 1080,
        "debug_video": False,
        "ball_tiles": False,
        "tracker_config": "bytetrack.yaml",
    },
    "balanced": {
        "proxy_height": 1080,
        "inference_imgsz": 1280,
        "vid_stride": 1,
        "yolo_model": "yolov8s.pt",
        "batch_size": "auto",
        "output_height": 1080,
        "debug_video": False,
        "ball_tiles": False,
        "tracker_config": "botsort.yaml",
    },
    "quality": {
        "proxy_height": 1080,
        "inference_imgsz": 1536,
        "vid_stride": 1,
        "yolo_model": "yolov8m.pt",
        "batch_size": "auto",
        "output_height": 1440,
        "debug_video": False,
        "ball_tiles": True,
        "tracker_config": "botsort.yaml",
    },
}

PROFILE_KEYS = tuple(PROFILES[DEFAULT_PROFILE].keys())

# Same-size weights per detector family. Ultralytics downloads every name in
# this table on first use (or scripts/download_models.py bakes them).
MODEL_FAMILIES: Dict[str, Dict[str, str]] = {
    "yolov8": {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt", "x": "yolov8x.pt"},
    "yolo11": {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt", "x": "yolo11x.pt"},
    "yolo26": {"n": "yolo26n.pt", "s": "yolo26s.pt", "m": "yolo26m.pt", "l": "yolo26l.pt", "x": "yolo26x.pt"},
}

_STOCK_MODEL_RE = re.compile(r"^(yolov8|yolo11|yolo26)([nsmlx])\.(pt|engine|onnx)$", re.IGNORECASE)


def profile_names() -> list[str]:
    return list(PROFILES.keys())


def get_profile(name: Optional[str]) -> Dict[str, Any]:
    """Return a copy of the profile defaults (``DEFAULT_PROFILE`` for None/'')."""
    key = str(name or DEFAULT_PROFILE).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; expected one of {', '.join(PROFILES)}")
    return copy.deepcopy(PROFILES[key])


def model_size_letter(model_name: str) -> str:
    """'yolov8s.pt' -> 's'. Unknown/custom weights count as 's'."""
    match = _STOCK_MODEL_RE.match(Path(str(model_name or "")).name)
    return match.group(2).lower() if match else "s"


def apply_model_family(model_name: str, family: Optional[str] = None) -> str:
    """Swap a stock model name to the same size in another family.

    ``family`` defaults to ``VH_MODEL_FAMILY`` (yolov8 | yolo11 | yolo26).
    Custom weights (paths, fine-tuned ``best.pt``) are returned unchanged.
    """
    family = str(family if family is not None else os.getenv("VH_MODEL_FAMILY", "")).strip().lower()
    if not family or family not in MODEL_FAMILIES:
        return model_name
    match = _STOCK_MODEL_RE.match(str(model_name or ""))
    if not match:
        return model_name
    return MODEL_FAMILIES[family][match.group(2).lower()]


def resolve_job_config(config: Optional[Mapping[str, Any]], *, strict: bool = False) -> Dict[str, Any]:
    """Merge profile defaults under the explicit keys of ``config``.

    * ``config["profile"]`` picks the profile (default ``balanced``).
    * Any key present in ``config`` with a non-None value wins over the
      profile. ``None`` counts as "not set" so UI forms can send blanks.
    * ``yolo_model`` from the profile (not an explicit one) is passed through
      ``apply_model_family`` so ``VH_MODEL_FAMILY=yolo26`` upgrades defaults.
    * Unknown profile: ``ValueError`` when ``strict``, else falls back to the
      default and records ``profile_warning``.
    * Adds ``profile`` (resolved name) and ``profile_overrides`` (the profile
      keys the caller overrode) for logging/UI.
    """
    source = dict(config or {})
    requested = str(source.get("profile") or DEFAULT_PROFILE).strip().lower()
    warning = None
    if requested not in PROFILES:
        if strict:
            raise ValueError(f"Unknown profile {source.get('profile')!r}; expected one of {', '.join(PROFILES)}")
        warning = f"unknown profile {source.get('profile')!r}; using {DEFAULT_PROFILE}"
        requested = DEFAULT_PROFILE

    resolved = get_profile(requested)
    resolved["yolo_model"] = apply_model_family(resolved["yolo_model"])
    overrides = []
    for key, value in source.items():
        if key == "profile" or value is None:
            continue
        if key in resolved and resolved[key] != value:
            overrides.append(key)
        resolved[key] = value
    resolved["profile"] = requested
    resolved["profile_overrides"] = sorted(overrides)
    if warning:
        resolved["profile_warning"] = warning
    return resolved


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    """Directory holding detector weights: ``VH_MODEL_DIR`` or ``<repo>/models``."""
    explicit = os.getenv("VH_MODEL_DIR", "").strip()
    return Path(explicit) if explicit else REPO_ROOT / "models"


def resolve_model_path(model_name: str) -> str:
    """Return a loadable path (or bare name) for ``YOLO(...)``.

    Order: existing path as given -> ``VH_MODEL_DIR/<name>`` -> repo root
    (the bundled ``yolov8n.pt``) -> ``VH_MODEL_DIR/<name>`` as the download
    target when that directory is writable (Ultralytics downloads stock
    names straight to the path it is given) -> the bare name.
    """
    name = str(model_name or "").strip() or PROFILES[DEFAULT_PROFILE]["yolo_model"]
    given = Path(name)
    if given.exists():
        return str(given)
    base = given.name
    candidates = [model_dir() / base, REPO_ROOT / base]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    target_dir = model_dir()
    if not given.is_absolute() and len(given.parts) == 1 and target_dir.is_dir() and os.access(target_dir, os.W_OK):
        return str(target_dir / base)
    return name


# ---------------------------------------------------------------------------
# Hardware classes and the runtime model
# ---------------------------------------------------------------------------
#
# Reference throughputs (frames/s). Sources: PLAN.md budget for an RTX 4080
# (proxy ~6 min, detect 12-20 min at 1280 batch 16, render ~8 min for 90 min
# 4K30), NVDEC/NVENC session throughput for the GPU generation, and relative
# MPS/CPU YOLO throughput. They are planning numbers, not promises; measure
# with bench/bench_match.py and pass the result via ``measured=``.
#
#   proxy_fps_4k   ffmpeg: decode 4K source (hwaccel) + scale + encode proxy
#   detect_fps_640 YOLOv8s fp16 at imgsz 640 with a full batch (PyTorch eager)
#   render_fps_4k  ffmpeg: decode 4K source + crop + scale 1080p + encode
#   auto_batch     batch at imgsz 1280 that fits memory
#   cpu_factor     analysis/tracker CPU speed relative to a fast desktop core

HARDWARE_CLASSES: Dict[str, Dict[str, Any]] = {
    "rtx_4090": {
        "label": "NVIDIA RTX 4090 / 5090 class (24-32 GB)",
        "device": "cuda", "encoder": "h264_nvenc", "hwaccel": "cuda",
        "proxy_fps_4k": 480.0, "detect_fps_640": 1250.0, "render_fps_4k": 380.0,
        "auto_batch": 32, "cpu_factor": 1.0,
    },
    "rtx_4080": {
        "label": "NVIDIA RTX 4080 / 5080 class (16 GB)",
        "device": "cuda", "encoder": "h264_nvenc", "hwaccel": "cuda",
        "proxy_fps_4k": 450.0, "detect_fps_640": 900.0, "render_fps_4k": 350.0,
        "auto_batch": 16, "cpu_factor": 1.0,
    },
    "rtx_3080": {
        "label": "NVIDIA RTX 3080 / 3090 / 4070 class (10-12 GB)",
        "device": "cuda", "encoder": "h264_nvenc", "hwaccel": "cuda",
        "proxy_fps_4k": 360.0, "detect_fps_640": 620.0, "render_fps_4k": 300.0,
        "auto_batch": 8, "cpu_factor": 1.0,
    },
    "dgx_spark": {
        "label": "NVIDIA DGX Spark (GB10, 128 GB unified, arm64)",
        "device": "cuda", "encoder": "h264_nvenc", "hwaccel": "cuda",
        "proxy_fps_4k": 330.0, "detect_fps_640": 700.0, "render_fps_4k": 270.0,
        "auto_batch": 32, "cpu_factor": 1.1,
    },
    "apple_m2_ultra": {
        "label": "Apple M2/M3 Ultra, M4 Max class (MPS + VideoToolbox)",
        "device": "mps", "encoder": "h264_videotoolbox", "hwaccel": "videotoolbox",
        "proxy_fps_4k": 280.0, "detect_fps_640": 360.0, "render_fps_4k": 230.0,
        "auto_batch": 16, "cpu_factor": 0.9,
    },
    "apple_m1_max": {
        "label": "Apple M1/M2/M3 Max and smaller (MPS + VideoToolbox)",
        "device": "mps", "encoder": "h264_videotoolbox", "hwaccel": "videotoolbox",
        "proxy_fps_4k": 190.0, "detect_fps_640": 180.0, "render_fps_4k": 160.0,
        "auto_batch": 8, "cpu_factor": 1.0,
    },
    "cpu_8core": {
        "label": "CPU only, 8 cores (fallback, not a target)",
        "device": "cpu", "encoder": "libx264", "hwaccel": None,
        "proxy_fps_4k": 55.0, "detect_fps_640": 22.0, "render_fps_4k": 30.0,
        "auto_batch": 4, "cpu_factor": 1.5,
    },
}

DEFAULT_HARDWARE_CLASS = "cpu_8core"

STAGE_MODEL: Dict[str, Any] = {
    # Relative detector cost vs YOLOv8s (same imgsz).
    "model_speed": {"n": 1.7, "s": 1.0, "m": 0.5, "l": 0.33, "x": 0.2},
    # Batch efficiency (fraction of full-batch throughput).
    "batch_efficiency": [(1, 0.5), (2, 0.65), (4, 0.8), (8, 0.92), (16, 1.0)],
    # Per analysed frame CPU work on top of the detector (tracker update,
    # team-colour sampling, bookkeeping), milliseconds on a fast core.
    "tracker_ms_per_frame": 1.5,
    # Extra detector cost of the high-res ball tile pass (quality profile).
    "ball_tiles_cost": 0.35,
    # Analysis (ball track, stats, events, camera plan): seconds per match minute.
    "analysis_s_per_min": 1.2,
    # Decode/encode cost scales with pixels^exponent relative to 4K.
    "pixel_exponent": 0.85,
    # Encoding a 1440p output costs this much more than 1080p.
    "output_1440_cost": 1.25,
    # Clips are stream-copy cuts; the reel is re-encoded at output height.
    "clips_fixed_s": 20.0,
    "reel_duration_s": 300.0,
    "reel_speedup_vs_render": 2.0,
    # Debug wide video is rendered from the proxy.
    "debug_speedup_vs_render": 1.5,
}

_PIXELS_4K = 3840 * 2160


def hardware_classes() -> list[str]:
    return list(HARDWARE_CLASSES.keys())


def _batch_efficiency(batch: int) -> float:
    eff = STAGE_MODEL["batch_efficiency"][0][1]
    for size, value in STAGE_MODEL["batch_efficiency"]:
        if batch >= size:
            eff = value
    return eff


def suggest_batch_size(hardware_class: str, imgsz: int) -> int:
    """Advisory 'auto' batch: the class batch at 1280 scaled by pixel count, power of two, 1..64."""
    hw = HARDWARE_CLASSES.get(hardware_class, HARDWARE_CLASSES[DEFAULT_HARDWARE_CLASS])
    raw = hw["auto_batch"] * (1280.0 / max(320, int(imgsz))) ** 2
    batch = 2 ** int(math.floor(math.log2(max(1.0, raw))))
    return int(max(1, min(64, batch)))


def _source_scale(source_height: int, exponent: float) -> float:
    """Throughput multiplier for a source of ``source_height`` (16:9) vs 4K."""
    height = max(144, int(source_height))
    pixels = (height * 16 / 9) * height
    return (_PIXELS_4K / pixels) ** exponent


def estimate_runtime(
    duration_s: float,
    profile: str = DEFAULT_PROFILE,
    hardware_class: str = DEFAULT_HARDWARE_CLASS,
    *,
    source_height: int = 2160,
    source_fps: float = 30.0,
    config: Optional[Mapping[str, Any]] = None,
    measured: Optional[Mapping[str, float]] = None,
    tensorrt: bool = False,
) -> Dict[str, Any]:
    """Estimate wall time per stage for one match.

    ``config`` holds explicit job keys layered over ``profile`` (same rules
    as ``resolve_job_config``). ``measured`` may set any of
    ``proxy_fps`` (source frames/s), ``detect_fps`` (detector images/s at the
    resolved imgsz/batch, tracker excluded), ``render_fps`` (source
    frames/s) to replace the class reference numbers; the bench uses this.

    Returns ``{"stages": [...], "stages_s": {...}, "total_s", "total_min", ...}``.
    """
    if hardware_class not in HARDWARE_CLASSES:
        raise ValueError(f"Unknown hardware class {hardware_class!r}; expected one of {', '.join(HARDWARE_CLASSES)}")
    cfg = resolve_job_config({**dict(config or {}), "profile": profile}, strict=True)
    hw = HARDWARE_CLASSES[hardware_class]
    measured = dict(measured or {})
    duration_s = max(0.0, float(duration_s))
    source_fps = max(1.0, float(source_fps))
    source_frames = duration_s * source_fps
    imgsz = int(cfg["inference_imgsz"])
    stride = max(1, int(cfg["vid_stride"]))
    size = model_size_letter(str(cfg["yolo_model"]))
    batch = cfg.get("batch_size", "auto")
    batch = suggest_batch_size(hardware_class, imgsz) if str(batch).lower() == "auto" else max(1, int(batch))
    exponent = STAGE_MODEL["pixel_exponent"]
    src_scale = _source_scale(source_height, exponent)

    stages = []

    # 1. Proxy pass: one decode of the source, proxy encode + analysis audio.
    proxy_fps = float(measured.get("proxy_fps") or hw["proxy_fps_4k"] * src_scale)
    proxy_note = "measured" if measured.get("proxy_fps") else f"{hw['proxy_fps_4k']:.0f} fps at 4K x source scale {src_scale:.2f}"
    stages.append({"name": "proxy", "frames": source_frames, "fps": proxy_fps,
                   "seconds": source_frames / proxy_fps, "basis": proxy_note})

    # 2. Detection + tracking on the proxy.
    analysed = source_frames / stride
    if measured.get("detect_fps"):
        det_fps = float(measured["detect_fps"])
        det_note = f"measured at imgsz {imgsz}"
    else:
        det_fps = (hw["detect_fps_640"] * STAGE_MODEL["model_speed"].get(size, 1.0)
                   * (640.0 / imgsz) ** 2 * _batch_efficiency(batch) * (1.8 if tensorrt else 1.0))
        det_note = (f"{hw['detect_fps_640']:.0f} fps@640 (v8s) x model {size} x (640/{imgsz})^2 x batch {batch}"
                    + (" x TensorRT 1.8" if tensorrt else ""))
    det_cost = 1.0 + (STAGE_MODEL["ball_tiles_cost"] if cfg.get("ball_tiles") else 0.0)
    per_frame = det_cost / det_fps + STAGE_MODEL["tracker_ms_per_frame"] * hw["cpu_factor"] / 1000.0
    stages.append({"name": "detect_track", "frames": analysed, "fps": 1.0 / per_frame,
                   "seconds": analysed * per_frame,
                   "basis": det_note + (" + ball tiles" if cfg.get("ball_tiles") else "") + f", stride {stride}, batch {batch}"})

    # 3. Analysis: ball track, stats, events, camera plan (CPU, no video decode).
    analysis_s = duration_s / 60.0 * STAGE_MODEL["analysis_s_per_min"] * hw["cpu_factor"]
    stages.append({"name": "analysis", "frames": 0, "fps": None, "seconds": analysis_s,
                   "basis": f"{STAGE_MODEL['analysis_s_per_min']} s per match minute x cpu {hw['cpu_factor']}"})

    # 4. Final render: decode source once, crop+scale, encode at output height.
    out_cost = STAGE_MODEL["output_1440_cost"] if int(cfg["output_height"]) > 1080 else 1.0
    render_fps = float(measured.get("render_fps") or hw["render_fps_4k"] * src_scale / out_cost)
    render_note = "measured" if measured.get("render_fps") else f"{hw['render_fps_4k']:.0f} fps at 4K x source scale {src_scale:.2f} / output {out_cost}"
    stages.append({"name": "render", "frames": source_frames, "fps": render_fps,
                   "seconds": source_frames / render_fps, "basis": render_note})

    # 5. Clips (stream copy) + reel (re-encode of ~5 min at output height).
    reel_frames = min(duration_s, STAGE_MODEL["reel_duration_s"]) * source_fps
    reel_fps = render_fps * STAGE_MODEL["reel_speedup_vs_render"]
    clips_s = STAGE_MODEL["clips_fixed_s"] + reel_frames / reel_fps
    stages.append({"name": "clips_reel", "frames": reel_frames, "fps": reel_fps, "seconds": clips_s,
                   "basis": f"{STAGE_MODEL['clips_fixed_s']:.0f} s cuts + reel at {STAGE_MODEL['reel_speedup_vs_render']}x render fps"})

    if cfg.get("debug_video"):
        dbg_fps = render_fps * STAGE_MODEL["debug_speedup_vs_render"]
        stages.append({"name": "debug_video", "frames": source_frames, "fps": dbg_fps,
                       "seconds": source_frames / dbg_fps, "basis": "wide debug render from proxy"})

    for stage in stages:
        stage["seconds"] = round(float(stage["seconds"]), 1)
        if stage["fps"] is not None:
            stage["fps"] = round(float(stage["fps"]), 1)
        stage["frames"] = int(round(stage["frames"]))
    total = round(sum(stage["seconds"] for stage in stages), 1)
    return {
        "profile": cfg["profile"],
        "hardware_class": hardware_class,
        "hardware_label": hw["label"],
        "device": hw["device"],
        "encoder": hw["encoder"],
        "duration_s": duration_s,
        "source_height": int(source_height),
        "source_fps": source_fps,
        "inference_imgsz": imgsz,
        "vid_stride": stride,
        "yolo_model": cfg["yolo_model"],
        "batch_size": batch,
        "stages": stages,
        "stages_s": {stage["name"]: stage["seconds"] for stage in stages},
        "total_s": total,
        "total_min": round(total / 60.0, 1),
        "realtime_factor": round(total / duration_s, 3) if duration_s else None,
    }


# ---------------------------------------------------------------------------
# Hardware classification
# ---------------------------------------------------------------------------

_NVIDIA_RULES = [
    ("dgx_spark", ("gb10", "dgx spark", "grace blackwell")),
    ("rtx_4090", ("5090", "4090", "rtx 6000", "rtx pro 6000", "a100", "h100", "h200", "b200", "l40")),
    ("rtx_4080", ("5080", "4080", "5070 ti", "4070 ti", "rtx 5000", "a6000", "rtx 4500")),
    ("rtx_3080", ("3090", "3080", "4070", "5070", "3070", "4060", "5060", "a5000", "a4000", "titan")),
]


def _gpu_names(status: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    smi = status.get("nvidia_smi") or {}
    for gpu in smi.get("gpus") or []:
        if gpu.get("name"):
            names.append(str(gpu["name"]))
    torch_info = status.get("torch") or {}
    for name in torch_info.get("devices") or []:
        if name:
            names.append(str(name))
    return names


def _gpu_memory_mb(status: Mapping[str, Any]) -> int:
    smi = status.get("nvidia_smi") or {}
    values = [int(gpu.get("memory_total_mb") or 0) for gpu in smi.get("gpus") or []]
    return max(values) if values else 0


def classify_hardware(status: Optional[Mapping[str, Any]]) -> str:
    """Map a ``get_gpu_status()`` payload to a ``HARDWARE_CLASSES`` key.

    NVIDIA: by GPU name (``nvidia_smi.gpus[].name`` or ``torch.devices``),
    unknown CUDA GPUs by memory (>=20 GB 4090 class, >=14 GB 4080 class,
    else 3080 class; a 0/absent memory with an aarch64 host is a Spark).
    Apple: MPS available -> Ultra / M4+ Max is ``apple_m2_ultra``, anything
    else ``apple_m1_max``. Otherwise ``cpu_8core``.
    """
    status = status or {}
    torch_info = status.get("torch") or {}
    platform_info = status.get("platform") or {}
    names = [name.lower() for name in _gpu_names(status)]
    cuda = bool(torch_info.get("cuda_available")) or bool((status.get("nvidia_smi") or {}).get("available"))
    if cuda or names:
        for hw_class, needles in _NVIDIA_RULES:
            if any(needle in name for name in names for needle in needles):
                return hw_class
        if cuda:
            memory = _gpu_memory_mb(status)
            machine = str(platform_info.get("machine") or "").lower()
            if memory == 0 and machine in {"aarch64", "arm64"}:
                return "dgx_spark"  # GB10 reports unified memory as [N/A]
            if memory >= 20000:
                return "rtx_4090"
            if memory >= 14000:
                return "rtx_4080"
            return "rtx_3080"
    mps = bool(status.get("mps_available")) or bool(torch_info.get("mps_available"))
    if mps:
        brand = str(platform_info.get("cpu_brand") or "").lower()
        if "ultra" in brand or re.search(r"\bm[4-9]\s+max\b", brand):
            return "apple_m2_ultra"
        return "apple_m1_max"
    return DEFAULT_HARDWARE_CLASS


def runtime_table(duration_s: float = 5400.0, source_height: int = 2160, source_fps: float = 30.0) -> Dict[str, Dict[str, float]]:
    """{hardware_class: {profile: total_minutes}} for docs/UI."""
    return {
        hw: {
            profile: estimate_runtime(duration_s, profile, hw, source_height=source_height, source_fps=source_fps)["total_min"]
            for profile in PROFILES
        }
        for hw in HARDWARE_CLASSES
    }
