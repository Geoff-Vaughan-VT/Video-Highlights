from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from ..config import settings
from .perf_profiles import resolve_model_path

# Stock Ultralytics weights fine-tuned when the config names no base model
# (the "balanced" profile's detector). Resolved through VH_MODEL_DIR first.
DEFAULT_BASE_MODEL = "yolov8s.pt"


_SAFE_RUN_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _coerce_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(maximum, max(minimum, parsed))


def _safe_name(value: Any, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    cleaned = _SAFE_RUN_NAME.sub("-", raw).strip(".-")
    return cleaned or fallback


def build_yolo_train_args(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_yaml = Path(str(config.get("dataset_yaml") or config.get("data") or "").strip()).expanduser()
    if not dataset_yaml:
        raise ValueError("dataset_yaml is required for YOLO detector training.")
    if not dataset_yaml.exists() or not dataset_yaml.is_file():
        raise ValueError(f"YOLO dataset YAML was not found: {dataset_yaml}")

    base_model = str(config.get("base_model") or config.get("model") or DEFAULT_BASE_MODEL).strip()
    if not base_model:
        raise ValueError("base_model is required for YOLO detector training.")

    project_root = Path(str(config.get("project_dir") or Path(settings.output_root) / "training" / "yolo")).expanduser()
    run_name = _safe_name(config.get("run_name"), f"{Path(base_model).stem}-custom")

    args: Dict[str, Any] = {
        "data": str(dataset_yaml.resolve()),
        "epochs": _coerce_int(config.get("epochs"), 50, 1, 1000),
        "imgsz": _coerce_int(config.get("imgsz"), 960, 320, 2048),
        "batch": _coerce_int(config.get("batch"), 8, -1, 512),
        "workers": _coerce_int(config.get("workers"), 4, 0, 64),
        "patience": _coerce_int(config.get("patience"), 25, 0, 300),
        "project": str(project_root.resolve()),
        "name": run_name,
        "exist_ok": False,
        "plots": bool(config.get("plots", True)),
        "verbose": bool(config.get("verbose", False)),
    }

    device = str(config.get("device") or "").strip()
    if device:
        args["device"] = device
    if "cache" in config:
        args["cache"] = bool(config.get("cache"))
    if "freeze" in config:
        args["freeze"] = _coerce_int(config.get("freeze"), 0, 0, 1000)

    return {"base_model": base_model, "base_model_path": resolve_model_path(base_model), "args": args}


def train_ultralytics_yolo(config: Dict[str, Any]) -> Dict[str, Any]:
    prepared = build_yolo_train_args(config)
    base_model = str(prepared.get("base_model_path") or prepared["base_model"])
    train_args = dict(prepared["args"])

    from ultralytics import YOLO
    import ultralytics

    model = YOLO(base_model)
    results = model.train(**train_args)
    save_dir = Path(getattr(results, "save_dir", train_args["project"]))
    weights_dir = save_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    metrics: Dict[str, Any] = {
        "training_type": "ultralytics_yolo",
        "base_model": base_model,
        "ultralytics_version": getattr(ultralytics, "__version__", None),
        "dataset_yaml": train_args["data"],
        "save_dir": str(save_dir.resolve()),
        "best_weights_path": str(best_path.resolve()) if best_path.exists() else "",
        "last_weights_path": str(last_path.resolve()) if last_path.exists() else "",
        "train_args": train_args,
    }
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics["results"] = results_dict

    if not best_path.exists():
        raise RuntimeError(f"YOLO training finished but best.pt was not found under {weights_dir}")

    return {
        "candidate_model_version": str(best_path.resolve()),
        "metrics": metrics,
        "gates_passed": True,
    }
