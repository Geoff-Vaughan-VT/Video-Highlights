#!/usr/bin/env python3
"""Fetch Ultralytics detector weights into VH_MODEL_DIR (idempotent, offline-safe).

    python scripts/download_models.py                       # profile defaults
    python scripts/download_models.py yolov8n.pt yolo26s.pt # explicit names
    VH_MODEL_DIR=/models python scripts/download_models.py --configure-ultralytics

* Names are Ultralytics release assets (yolov8n.pt, yolov8s.pt, yolov8m.pt,
  yolo11s.pt, yolo26s.pt, ...). A file already present in the model dir is
  never downloaded again, so the script works offline once seeded.
* ``--seed-dir`` (default: repo root) is checked first and copied from, so
  the bundled ``yolov8n.pt`` never needs the network.
* ``--configure-ultralytics`` points Ultralytics' ``weights_dir`` setting at
  the model dir, so ``YOLO("yolov8s.pt")`` with a bare name resolves there.
* Exit code is non-zero only when a requested model is missing afterwards
  and ``--allow-missing`` was not given.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt"]


def _model_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    env = os.getenv("VH_MODEL_DIR", "").strip()
    return Path(env) if env else REPO_ROOT / "models"


def _download(name: str, target: Path) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    # Ultralytics downloads release assets straight to the path it is given.
    result = Path(attempt_download_asset(str(target)))
    if result.resolve() != target.resolve() and result.exists():
        shutil.copy2(result, target)


def _valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 100_000


def ensure_models(names: list[str], model_dir: Path, seed_dirs: list[Path]) -> dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for raw in names:
        name = Path(raw).name
        target = model_dir / name
        if _valid(target):
            status[name] = "present"
            continue
        seeded = next((seed / name for seed in seed_dirs if _valid(seed / name)), None)
        if seeded is not None:
            shutil.copy2(seeded, target)
            status[name] = f"copied from {seeded.parent}"
            continue
        try:
            _download(name, target)
            status[name] = "downloaded" if _valid(target) else "missing (download produced no file)"
        except Exception as exc:  # offline, unknown name, ultralytics missing
            status[name] = f"missing ({type(exc).__name__}: {exc})"
    return status


def configure_ultralytics(model_dir: Path) -> str:
    try:
        from ultralytics import settings

        settings.update({"weights_dir": str(model_dir)})
        return f"ultralytics weights_dir -> {model_dir}"
    except Exception as exc:
        return f"could not update ultralytics settings: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model names (default: $VH_MODELS or yolov8n.pt yolov8s.pt)")
    parser.add_argument("--model-dir", help="target dir (default: $VH_MODEL_DIR or <repo>/models)")
    parser.add_argument("--seed-dir", action="append", default=None, help="dir checked for existing weights (repeatable; default repo root)")
    parser.add_argument("--configure-ultralytics", action="store_true", help="set ultralytics weights_dir to the model dir")
    parser.add_argument("--allow-missing", action="store_true", help="exit 0 even if a model could not be fetched")
    args = parser.parse_args(argv)

    names = args.models or os.getenv("VH_MODELS", "").split() or DEFAULT_MODELS
    model_dir = _model_dir(args.model_dir)
    seeds = [Path(p) for p in (args.seed_dir or [str(REPO_ROOT)])]
    status = ensure_models(names, model_dir, seeds)
    for name, state in status.items():
        print(f"[models] {name}: {state}")
    if args.configure_ultralytics:
        print(f"[models] {configure_ultralytics(model_dir)}")
    missing = [name for name, state in status.items() if state.startswith("missing")]
    if missing and not args.allow_missing:
        print(f"[models] missing: {', '.join(missing)} (dir: {model_dir})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
