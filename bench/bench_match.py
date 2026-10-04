#!/usr/bin/env python3
"""Stage benchmark + 90-minute projection on this machine.

    python bench/bench_match.py --minutes 2 --height 2160              # 4K, balanced profile
    python bench/bench_match.py --minutes 0.2 --height 360             # CI smoke
    python bench/bench_match.py --source D:/Videos/match.mp4 --minutes 3

Steps (each guarded: a missing/broken module is reported as skipped, the
rest still runs):

1. synth     render a synthetic match (backend.services.synthetic_match) at
             --height (16:9) and --fps, unless --source is given.
2. proxy     backend.services.frame_source.build_proxy when importable, else
             an equivalent single ffmpeg pass (scale + encode + 16 kHz WAV).
3. detect    detectors.build_detector (or raw ultralytics) on N proxy frames
             at the profile imgsz/batch on the selected device.
4. render    ffmpeg crop+scale+encode of the source to output_height with a
             sendcmd-driven pan (the camera_crops.txt mechanism).
5. project   perf_profiles.estimate_runtime for a 90-minute match at the
             source resolution, using the measured fps for proxy/detect/
             render, next to the class reference model.

Results: bench/results/<timestamp>.json and .md (or --out DIR).
CPU numbers are only good for checking the harness works.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services import perf_profiles  # noqa: E402  (stdlib-only module)

MATCH_S = 90 * 60


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    try:
        from backend.services.ffmpeg_tools import ffmpeg_exe

        return ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def _probe_frames(path: Path) -> Dict[str, Any]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        info = {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        cap.release()
        return info
    except Exception:
        return {"frames": 0, "fps": 0.0, "width": 0, "height": 0}


def _run_stage(name: str, results: Dict[str, Any], fn: Callable[[], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    print(f"[bench] {name} ...", flush=True)
    started = time.perf_counter()
    try:
        data = fn() or {}
        data.setdefault("status", "ok")
    except Exception as exc:
        data = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
    data.setdefault("wall_s", round(time.perf_counter() - started, 3))
    results["stages"][name] = data
    detail = data.get("fps")
    print(f"[bench] {name}: {data['status']}" + (f" {detail:.1f} fps" if isinstance(detail, (int, float)) else "")
          + (f" ({data.get('impl')})" if data.get("impl") else "") + (f" - {data.get('error') or data.get('reason')}" if data["status"] != "ok" else ""),
          flush=True)
    return data if data["status"] == "ok" else None


def _pick_encoder(gpu: Dict[str, Any]) -> str:
    try:
        from backend.services import device as device_mod

        return device_mod.pick_encoder()
    except Exception:
        return str(gpu.get("recommended_encoder") or "libx264")


def _encoder_args(encoder: str) -> List[str]:
    if encoder.endswith("_nvenc"):
        return ["-c:v", encoder, "-preset", "p4", "-cq", "23"]
    if encoder.endswith("_videotoolbox"):
        return ["-c:v", encoder, "-b:v", "12M", "-allow_sw", "1"]
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
    return ["-c:v", encoder]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_synth(args: argparse.Namespace, work: Path) -> Dict[str, Any]:
    from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match

    height = int(args.height) - int(args.height) % 2
    width = int(round(height * 16 / 9))
    width -= width % 2
    scale = height / 720.0
    duration = max(2.0, float(args.minutes) * 60.0)
    goals = [(round(duration * 0.4, 2), "right"), (round(duration * 0.75, 2), "left")]
    spec = SyntheticMatchSpec(
        width=width, height=height, fps=float(args.fps), duration_s=duration,
        player_w_px=max(8, int(18 * scale)), player_h_px=max(20, int(44 * scale)),
        ball_radius_px=max(3, int(6 * scale)), goals=goals,
    )
    out = work / f"synthetic_{height}p.mp4"
    truth = generate_synthetic_match(out, spec, prefer_ffmpeg=shutil.which("ffmpeg") is not None)
    info = _probe_frames(out)
    return {"impl": "synthetic_match", "path": str(out), "width": width, "height": height, "video_fps": float(args.fps),
            "duration_s": duration, "frames": info["frames"] or int(duration * args.fps),
            "goals": truth.goal_times_s}


def stage_proxy(source: Path, src_frames: int, proxy_height: int, work: Path, gpu: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = work / "proxy"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from backend.services import frame_source  # workstream A
    except Exception as exc:
        frame_source = None
        import_error = f"{type(exc).__name__}: {exc}"
    if frame_source is not None and hasattr(frame_source, "build_proxy"):
        params = inspect.signature(frame_source.build_proxy).parameters
        kwargs: Dict[str, Any] = {}
        for key, value in (("height", proxy_height), ("thumbs", False), ("audio_wav", True)):
            if key in params:
                kwargs[key] = value
        started = time.perf_counter()
        result = frame_source.build_proxy(str(source), str(out_dir), **kwargs)
        elapsed = time.perf_counter() - started
        frames = int(getattr(result, "frame_count", 0) or src_frames)
        return {"impl": "frame_source.build_proxy", "path": str(getattr(result, "path", "")),
                "hwaccel": getattr(result, "hwaccel_used", None), "encoder": getattr(result, "encoder_used", None),
                "frames": frames, "seconds": round(elapsed, 3), "fps": src_frames / max(elapsed, 1e-6)}

    # Stand-in: same work in one ffmpeg pass.
    encoder = _pick_encoder(gpu)
    hwaccel = gpu.get("recommended_hwaccel")
    proxy = out_dir / f"proxy_{proxy_height}p.mp4"
    wav = out_dir / "audio_analysis.wav"
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", str(hwaccel)]
    cmd += ["-i", str(source), "-map", "0:v:0", "-vf", f"scale=-2:{proxy_height}", *_encoder_args(encoder),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(proxy),
            "-map", "0:a:0?", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    started = time.perf_counter()
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    elapsed = time.perf_counter() - started
    return {"impl": "ffmpeg_standin", "import_error": import_error if frame_source is None else "build_proxy missing",
            "path": str(proxy), "hwaccel": hwaccel, "encoder": encoder, "frames": src_frames,
            "seconds": round(elapsed, 3), "fps": src_frames / max(elapsed, 1e-6)}


def _read_frames(path: Path, count: int) -> List[Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {path}")
    while len(frames) < count:  # short clips: reuse frames to fill the sample
        frames.append(frames[len(frames) % max(1, len(frames))])
    return frames


def stage_detect(proxy: Path, cfg: Dict[str, Any], args: argparse.Namespace, hw_class: str) -> Dict[str, Any]:
    imgsz = int(args.imgsz or cfg["inference_imgsz"])
    model_name = str(args.model or cfg["yolo_model"])
    model_path = perf_profiles.resolve_model_path(model_name)

    device_kind, torch_device, half = "cpu", "cpu", False
    batch: Optional[int] = None
    try:
        from backend.services import device as device_mod  # workstream A

        info = device_mod.select_device(args.device)
        device_kind, torch_device, half = info.kind, info.torch_device, bool(info.supports_half)
        batch = int(info.recommended_batch(imgsz))
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                device_kind, torch_device, half = "cuda", "cuda:0", True
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device_kind, torch_device = "mps", "mps"
        except Exception:
            pass
    if args.batch:
        batch = int(args.batch)
    if not batch:
        batch = perf_profiles.suggest_batch_size(hw_class, imgsz)
    n_frames = int(args.detect_frames or (batch * 8 if device_kind != "cpu" else max(4, min(8, batch * 2))))
    frames = _read_frames(proxy, n_frames + batch)  # +1 batch for warm-up

    impl = "detectors.build_detector"
    try:
        from backend.services import detectors  # workstream A

        det = detectors.build_detector(model_path, imgsz=imgsz, device=torch_device, half=half, batch=batch,
                                       use_tensorrt=bool(args.tensorrt), ball_tiles=False)

        def run(chunk: List[Any], start: int) -> None:
            det.detect_batch(chunk, frame_indices=list(range(start, start + len(chunk))))
    except Exception as exc:
        impl = f"ultralytics.predict (detectors unavailable: {type(exc).__name__})"
        from ultralytics import YOLO

        try:
            model = YOLO(model_path)
        except Exception:
            fallback = perf_profiles.resolve_model_path("yolov8n.pt")
            impl += f"; {model_name} not loadable, used yolov8n.pt"
            model_name, model = "yolov8n.pt", YOLO(fallback)

        def run(chunk: List[Any], start: int) -> None:
            model.predict(chunk, imgsz=imgsz, device=torch_device, half=half, verbose=False, batch=len(chunk))

    run(frames[:batch], 0)  # warm-up (cudnn autotune, MPS graph build, TRT load)
    _sync(device_kind)
    timed = frames[batch:batch + n_frames]
    started = time.perf_counter()
    for start in range(0, len(timed), batch):
        run(timed[start:start + batch], start)
    _sync(device_kind)
    elapsed = time.perf_counter() - started
    return {"impl": impl, "model": model_name, "model_path": model_path, "imgsz": imgsz, "batch": batch,
            "device": torch_device, "half": half, "frames": len(timed), "seconds": round(elapsed, 3),
            "fps": len(timed) / max(elapsed, 1e-6)}


def _sync(device_kind: str) -> None:
    try:
        import torch

        if device_kind == "cuda":
            torch.cuda.synchronize()
        elif device_kind == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def stage_render(source: Path, src: Dict[str, Any], output_height: int, work: Path, gpu: Dict[str, Any]) -> Dict[str, Any]:
    """ffmpeg-native follow-cam render stand-in: fixed 1.8x crop panned by sendcmd, scaled to output_height."""
    width, height = int(src["width"]), int(src["height"])
    duration = float(src["duration_s"])
    zoom = 1.8
    crop_w = int(width / zoom) // 2 * 2
    crop_h = int(height / zoom) // 2 * 2
    out_h = min(int(output_height), height) // 2 * 2
    out_w = int(round(out_h * crop_w / crop_h)) // 2 * 2
    # One pan command every 0.2 s (the real renderer writes one per frame).
    lines = []
    steps = max(1, int(duration / 0.2))
    import math

    for i in range(steps + 1):
        t = i * 0.2
        x = int((width - crop_w) / 2 * (1 + math.sin(t / 3.0))) // 2 * 2
        y = int((height - crop_h) / 2 * (1 + 0.3 * math.sin(t / 5.0))) // 2 * 2
        lines.append(f"{t:.3f} crop x {x}, crop y {y};")
    crops = work / "camera_crops_bench.txt"
    crops.write_text("\n".join(lines) + "\n", encoding="utf-8")

    encoder = _pick_encoder(gpu)
    hwaccel = gpu.get("recommended_hwaccel")
    out = work / "render_bench.mp4"
    vf = f"sendcmd=f={crops.name},crop={crop_w}:{crop_h}:0:0,scale={out_w}:{out_h}:flags=bicubic,setsar=1"

    def make_cmd(enc: str, accel: Optional[str]) -> List[str]:
        cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
        if accel:
            cmd += ["-hwaccel", str(accel)]
        return cmd + ["-i", str(source), "-vf", vf, *_encoder_args(enc), "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]

    impl = "ffmpeg crop(sendcmd)+scale+encode stand-in"
    try:
        from backend.services import camera_render  # workstream B

        names = [n for n in dir(camera_render) if "ffmpeg" in n.lower() or "sendcmd" in n.lower() or "crops" in n.lower()]
        if names:
            impl += f" (camera_render exposes {', '.join(names)}; wire it in once its signature is final)"
    except Exception:
        pass
    started = time.perf_counter()
    proc = subprocess.run(make_cmd(encoder, hwaccel), cwd=str(work), stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if proc.returncode != 0 and (encoder != "libx264" or hwaccel):
        impl += f"; {encoder}/{hwaccel} failed, retried libx264 software"
        encoder, hwaccel = "libx264", None
        started = time.perf_counter()
        proc = subprocess.run(make_cmd(encoder, None), cwd=str(work), stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-400:])
    elapsed = time.perf_counter() - started
    frames = int(src["frames"])
    return {"impl": impl, "encoder": encoder, "hwaccel": hwaccel, "output": f"{out_w}x{out_h}",
            "frames": frames, "seconds": round(elapsed, 3), "fps": frames / max(elapsed, 1e-6)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _markdown(results: Dict[str, Any]) -> str:
    lines = [
        f"# Bench {results['timestamp']}",
        "",
        f"Host: {results['host']['platform']} | hardware_class **{results['hardware_class']}** | "
        f"device {results['gpu'].get('recommended_device')} | encoder {results['gpu'].get('recommended_encoder')} | "
        f"profile **{results['profile']}**",
        "",
        "| stage | status | impl | frames | seconds | fps |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, data in results["stages"].items():
        fps = data.get("fps")
        lines.append(f"| {name} | {data.get('status')} | {str(data.get('impl') or data.get('error') or data.get('reason') or '')[:70]} | "
                     f"{data.get('frames', '')} | {data.get('seconds', data.get('wall_s', ''))} | "
                     f"{f'{fps:.1f}' if isinstance(fps, (int, float)) else ''} |")
    proj = results.get("projection") or {}
    if proj:
        lines += ["", f"## Projected 90-minute match at {proj['source_height']}p{proj['source_fps']:.0f}", "",
                  "| stage | measured-based (min) | class reference (min) | basis |", "|---|---:|---:|---|"]
        ref = {s["name"]: s for s in proj["reference"]["stages"]}
        for stage in proj["measured"]["stages"]:
            r = ref.get(stage["name"], {})
            lines.append(f"| {stage['name']} | {stage['seconds'] / 60:.1f} | {r.get('seconds', 0) / 60:.1f} | {stage['basis']} |")
        lines.append(f"| **total** | **{proj['measured']['total_min']:.1f}** | **{proj['reference']['total_min']:.1f}** | |")
    if results.get("notes"):
        lines += ["", *[f"- {note}" for note in results["notes"]]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=1.0, help="synthetic match length (default 1)")
    parser.add_argument("--height", type=int, default=2160, help="synthetic source height, 16:9 (default 2160)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--source", help="use an existing video instead of a synthetic one")
    parser.add_argument("--profile", default="balanced", choices=perf_profiles.profile_names())
    parser.add_argument("--model", help="override detector weights")
    parser.add_argument("--imgsz", type=int, help="override inference imgsz")
    parser.add_argument("--batch", type=int, help="override batch size")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--detect-frames", type=int, help="frames to time (default: 8 batches on GPU, <=8 on CPU)")
    parser.add_argument("--tensorrt", action="store_true", help="use a cached TensorRT engine (CUDA)")
    parser.add_argument("--skip", action="append", default=[], choices=["proxy", "detect", "render"])
    parser.add_argument("--out", default=str(REPO_ROOT / "bench" / "results"))
    parser.add_argument("--keep", action="store_true", help="keep the work directory")
    args = parser.parse_args(argv)

    cfg = perf_profiles.resolve_job_config({"profile": args.profile})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        from backend.services.gpu_status import get_gpu_status

        gpu = get_gpu_status()
    except Exception as exc:
        gpu = {"error": str(exc), "recommended_device": "cpu", "recommended_encoder": "libx264"}
    hw_class = perf_profiles.classify_hardware(gpu)
    results: Dict[str, Any] = {
        "timestamp": stamp,
        "args": vars(args),
        "profile": args.profile,
        "profile_config": {k: cfg[k] for k in perf_profiles.PROFILE_KEYS},
        "hardware_class": hw_class,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
                 "cpu_count": os.cpu_count()},
        "gpu": {k: gpu.get(k) for k in ("recommended_device", "recommended_encoder", "recommended_hwaccel",
                                        "hwaccels", "encoders", "mps_available", "platform")},
        "gpu_names": [g.get("name") for g in (gpu.get("nvidia_smi") or {}).get("gpus", [])],
        "stages": {},
        "notes": [],
    }
    print(f"[bench] hardware_class={hw_class} device={gpu.get('recommended_device')} "
          f"encoder={gpu.get('recommended_encoder')} profile={args.profile}", flush=True)
    if hw_class == "cpu_8core":
        results["notes"].append("CPU-only host: numbers validate the harness, they are not a projection target.")

    work = Path(tempfile.mkdtemp(prefix="vh_bench_"))
    try:
        if args.source:
            source = Path(args.source)
            info = _probe_frames(source)
            limit_s = float(args.minutes) * 60.0
            if info["fps"] and info["frames"] / info["fps"] > limit_s + 1:
                clip = work / "source_clip.mp4"
                subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-t", str(limit_s),
                                "-c", "copy", str(clip)], check=True, stdin=subprocess.DEVNULL)
                source, info = clip, _probe_frames(clip)
            src = {"impl": "user source", "path": str(source), "status": "ok", "frames": info["frames"],
                   "width": info["width"], "height": info["height"], "video_fps": info["fps"],
                   "duration_s": info["frames"] / max(info["fps"], 1e-6)}
            results["stages"]["synth"] = src
        else:
            src = _run_stage("synth", results, lambda: stage_synth(args, work))
            if src is None:
                raise SystemExit("synthetic generation failed; nothing to bench")
            source = Path(src["path"])

        measured: Dict[str, float] = {}
        proxy_path: Optional[Path] = None
        if "proxy" not in args.skip:
            proxy = _run_stage("proxy", results, lambda: stage_proxy(source, int(src["frames"]), int(cfg["proxy_height"]), work, gpu))
            if proxy:
                measured["proxy_fps"] = proxy["fps"]
                proxy_path = Path(proxy["path"]) if proxy.get("path") else None
        if "detect" not in args.skip:
            target = proxy_path if proxy_path and proxy_path.exists() else source
            det = _run_stage("detect", results, lambda: stage_detect(target, cfg, args, hw_class))
            if det:
                measured["detect_fps"] = det["fps"]
                if det.get("model") != cfg["yolo_model"]:
                    results["notes"].append(f"detect used {det.get('model')} instead of {cfg['yolo_model']}")
        if "render" not in args.skip:
            ren = _run_stage("render", results, lambda: stage_render(source, src, int(cfg["output_height"]), work, gpu))
            if ren:
                measured["render_fps"] = ren["fps"]

        det_stage = results["stages"].get("detect", {})
        overrides = {}
        if det_stage.get("status") == "ok":
            overrides = {"inference_imgsz": det_stage["imgsz"], "batch_size": det_stage["batch"]}
        common = dict(source_height=int(src["height"]), source_fps=float(src["video_fps"] or args.fps), config=overrides)
        results["projection"] = {
            "duration_s": MATCH_S,
            "source_height": int(src["height"]),
            "source_fps": float(src["video_fps"] or args.fps),
            "measured_fps": {k: round(v, 2) for k, v in measured.items()},
            "measured": perf_profiles.estimate_runtime(MATCH_S, args.profile, hw_class, measured=measured, **common),
            "reference": perf_profiles.estimate_runtime(MATCH_S, args.profile, hw_class, **common),
        }
        if src["height"] < 2160:
            results["notes"].append(f"source is {src['height']}p; rerun with --height 2160 for a 4K projection.")
    finally:
        if args.keep:
            results["work_dir"] = str(work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md = _markdown(results)
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"[bench] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
