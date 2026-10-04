"""Hardware/accelerator probe used by ``GET /v1/health/gpu`` and the job runner.

Keys kept from v1: ``ready``, ``rendering_ready``, ``torch``, ``nvidia_smi``,
``ffmpeg_nvenc``, ``recommendation``.

Added in v2:

* ``mps_available``: Apple Metal (torch MPS) usable.
* ``hwaccels``: list parsed from ``ffmpeg -hwaccels`` (``cuda``, ``videotoolbox``...).
* ``encoders``: ``{h264_nvenc, hevc_nvenc, h264_videotoolbox, hevc_videotoolbox, libx264}``
  -> bool, as *listed by ffmpeg* (listing != usable: NVENC also needs an
  NVIDIA GPU + driver, VideoToolbox needs macOS).
* ``recommended_device``: ``cuda`` | ``mps`` | ``cpu`` (``VH_DEVICE`` wins
  when set to something other than ``auto``).
* ``recommended_encoder`` / ``recommended_hwaccel``: what ffmpeg should use
  on this host for the proxy pass and the final render.
* ``platform``: system, machine, cpu_count, cpu_brand.
* ``hardware_class``: ``perf_profiles.classify_hardware`` of this payload.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict, List

from .ffmpeg_tools import ffmpeg_exe

TRACKED_ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox", "libx264")


def _run_nvidia_smi() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi was not found on PATH", "gpus": []}
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    if result.returncode != 0:
        return {"available": False, "error": (result.stderr or result.stdout or "").strip(), "gpus": []}

    gpus: List[Dict[str, Any]] = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        name, driver, memory_total, memory_used, utilization, temperature = parts[:6]
        gpus.append(
            {
                "index": index,
                "name": name,
                "driver_version": driver,
                "memory_total_mb": _safe_int(memory_total),
                "memory_used_mb": _safe_int(memory_used),
                "utilization_gpu_percent": _safe_int(utilization),
                "temperature_c": _safe_int(temperature),
            }
        )
    return {"available": bool(gpus), "error": None, "gpus": gpus}


def _safe_int(value: object) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _run_ffmpeg(args: List[str]) -> Dict[str, Any]:
    cmd = [ffmpeg_exe(), "-hide_banner", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError:
        return {"ok": False, "output": "", "error": "ffmpeg was not found on PATH"}
    except Exception as exc:
        return {"ok": False, "output": "", "error": str(exc)}
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    if result.returncode != 0:
        return {"ok": False, "output": output, "error": (result.stderr or "").strip() or f"exit {result.returncode}"}
    return {"ok": True, "output": output, "error": None}


def parse_hwaccels(output: str) -> List[str]:
    """Parse ``ffmpeg -hwaccels`` output into a list of method names."""
    methods: List[str] = []
    seen_header = False
    for raw in (output or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("hardware acceleration methods"):
            seen_header = True
            continue
        if seen_header and " " not in line and line not in methods:
            methods.append(line)
    return methods


def parse_encoders(output: str, names: tuple[str, ...] = TRACKED_ENCODERS) -> Dict[str, bool]:
    """Return ``{encoder: listed}`` from ``ffmpeg -encoders`` output."""
    listed = set()
    for raw in (output or "").splitlines():
        parts = raw.split()
        # Encoder rows look like " V....D h264_nvenc   NVIDIA NVENC H.264 encoder".
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            listed.add(parts[1])
    return {name: name in listed for name in names}


def _ffmpeg_capabilities() -> Dict[str, Any]:
    encoders_run = _run_ffmpeg(["-encoders"])
    hwaccels_run = _run_ffmpeg(["-hwaccels"])
    encoders = parse_encoders(encoders_run["output"]) if encoders_run["ok"] else {name: False for name in TRACKED_ENCODERS}
    hwaccels = parse_hwaccels(hwaccels_run["output"]) if hwaccels_run["ok"] else []
    return {
        "encoders": encoders,
        "hwaccels": hwaccels,
        "encoders_error": encoders_run["error"],
        "hwaccels_error": hwaccels_run["error"],
    }


def _check_ffmpeg_nvenc(encoders: Dict[str, bool] | None = None, error: str | None = None) -> Dict[str, Any]:
    """v1-compatible NVENC payload (kept for the UI and job logs)."""
    if encoders is None:
        caps = _ffmpeg_capabilities()
        encoders, error = caps["encoders"], caps["encoders_error"]
    available = bool(encoders.get("h264_nvenc"))
    return {
        "available": available,
        "encoder": "h264_nvenc",
        "error": None if available else (error or "h264_nvenc encoder was not listed by ffmpeg"),
    }


def _cpu_brand(system: str) -> str:
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False, timeout=3
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        elif system == "Linux" and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith(("model name", "hardware")):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or ""


def _platform_info() -> Dict[str, Any]:
    system = platform.system()
    return {
        "system": system,
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "cpu_brand": _cpu_brand(system),
    }


def _torch_info() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "mps_built": False,
        "mps_available": False,
        "error": None,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        payload.update(
            {
                "installed": True,
                "version": str(getattr(torch, "__version__", "")),
                "cuda_version": str(getattr(torch.version, "cuda", "") or ""),
                "cuda_available": cuda_available,
                "device_count": device_count,
                "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
            }
        )
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None:
            payload["mps_built"] = bool(mps.is_built())
            payload["mps_available"] = bool(mps.is_available())
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


def _recommend_device(torch_payload: Dict[str, Any]) -> str:
    override = os.getenv("VH_DEVICE", "").strip().lower()
    if override and override != "auto":
        return override
    if torch_payload.get("cuda_available"):
        return "cuda"
    if torch_payload.get("mps_available"):
        return "mps"
    return "cpu"


def _recommend_encoder(encoders: Dict[str, bool], nvidia_ok: bool, system: str) -> str:
    if nvidia_ok and encoders.get("h264_nvenc"):
        return "h264_nvenc"
    if system == "Darwin" and encoders.get("h264_videotoolbox"):
        return "h264_videotoolbox"
    if encoders.get("libx264"):
        return "libx264"
    return "mpeg4"


def _recommend_hwaccel(hwaccels: List[str], nvidia_ok: bool, system: str) -> str | None:
    if nvidia_ok and "cuda" in hwaccels:
        return "cuda"
    if system == "Darwin" and "videotoolbox" in hwaccels:
        return "videotoolbox"
    return None


def get_gpu_status() -> Dict[str, Any]:
    torch_payload = _torch_info()
    nvidia_payload = _run_nvidia_smi()
    caps = _ffmpeg_capabilities()
    nvenc_payload = _check_ffmpeg_nvenc(caps["encoders"], caps["encoders_error"])
    platform_payload = _platform_info()

    nvidia_ok = bool(nvidia_payload.get("available"))
    mps_available = bool(torch_payload.get("mps_available"))
    ready = bool(torch_payload["cuda_available"]) and nvidia_ok
    rendering_ready = nvidia_ok and bool(nvenc_payload.get("available"))
    recommended_device = _recommend_device(torch_payload)
    recommended_encoder = _recommend_encoder(caps["encoders"], nvidia_ok, platform_payload["system"])
    recommended_hwaccel = _recommend_hwaccel(caps["hwaccels"], nvidia_ok, platform_payload["system"])

    if ready and rendering_ready:
        recommendation = "GPU analysis and NVENC clip rendering are ready."
    elif ready:
        recommendation = "GPU analysis is ready. Install an FFmpeg build with h264_nvenc for GPU clip rendering."
    elif mps_available:
        if caps["encoders"].get("h264_videotoolbox"):
            recommendation = "Apple GPU (MPS) analysis and VideoToolbox rendering are ready."
        else:
            recommendation = "Apple GPU (MPS) analysis is ready. Install ffmpeg with VideoToolbox (brew install ffmpeg)."
    else:
        recommendation = "Install CUDA-enabled PyTorch and confirm nvidia-smi works (or run natively on Apple Silicon for MPS)."

    payload: Dict[str, Any] = {
        "ready": ready,
        "rendering_ready": rendering_ready,
        "torch": torch_payload,
        "nvidia_smi": nvidia_payload,
        "ffmpeg_nvenc": nvenc_payload,
        "recommendation": recommendation,
        "mps_available": mps_available,
        "hwaccels": caps["hwaccels"],
        "encoders": caps["encoders"],
        "recommended_device": recommended_device,
        "recommended_encoder": recommended_encoder,
        "recommended_hwaccel": recommended_hwaccel,
        "platform": platform_payload,
    }
    try:
        from .perf_profiles import classify_hardware

        payload["hardware_class"] = classify_hardware(payload)
    except Exception:  # pragma: no cover - classification must never break health
        payload["hardware_class"] = "cpu_8core"
    return payload
