"""Compute-device and media-acceleration selection.

One place decides *where* inference runs (CUDA / Apple MPS / CPU) and which
ffmpeg hardware decoder/encoder the media passes use, so the proxy pass,
the detector and the renderer agree on the same hardware.

* :func:`select_device` honours ``VH_DEVICE`` (``auto``, ``cuda``,
  ``cuda:1``, ``mps``, ``cpu``) and falls back to CPU when the requested
  accelerator is not usable.
* :func:`encoder_preferences` / :func:`hwaccel_preferences` return ordered
  candidate lists; :func:`pick_encoder` / :func:`pick_hwaccel` return the
  first candidate that actually works on this machine. ffmpeg *lists*
  ``h264_nvenc`` and ``cuda`` on many builds without a GPU, so every
  candidate is verified with a tiny real encode/device init. All probes are
  cached for the life of the process.

``torch`` is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("videohighlights.device")

DEVICE_ENV = "VH_DEVICE"
FFMPEG_ENV = "VH_FFMPEG"
FFPROBE_ENV = "VH_FFPROBE"

_PROBE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class DeviceInfo:
    """Where inference runs.

    ``torch_device`` is the string to hand to torch/ultralytics
    (``"cuda:0"``, ``"mps"``, ``"cpu"``).
    """

    kind: str  # "cuda" | "mps" | "cpu"
    name: str
    index: int = 0
    vram_gb: float = 0.0
    supports_half: bool = False
    requested: str = "auto"
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def torch_device(self) -> str:
        if self.kind == "cuda":
            return f"cuda:{self.index}"
        return self.kind

    @property
    def is_accelerated(self) -> bool:
        return self.kind in ("cuda", "mps")

    def recommended_batch(self, imgsz: int = 1280) -> int:
        """Heuristic inference batch size for ``imgsz`` (square letterbox).

        CUDA scales with VRAM (reference points at imgsz 1280: <6 GB -> 4,
        8 GB -> 8, 16 GB -> 16, 24+ GB -> 24, very large/unified memory ->
        32) and inversely with the pixel count of ``imgsz``. MPS -> 4, CPU -> 1.
        """
        imgsz = max(160, int(imgsz or 1280))
        if self.kind == "cuda":
            vram = float(self.vram_gb or 0.0)
            if vram <= 0:
                base = 4
            elif vram < 6:
                base = 4
            elif vram < 12:
                base = 8
            elif vram < 20:
                base = 16
            elif vram < 48:
                base = 24
            else:
                base = 32
            scaled = base * (1280.0 / imgsz) ** 2
            return int(max(1, min(64, round(scaled))))
        if self.kind == "mps":
            return 4 if imgsz >= 960 else 8
        return 1

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "index": self.index,
            "torch_device": self.torch_device,
            "vram_gb": round(float(self.vram_gb), 2),
            "supports_half": self.supports_half,
            "requested": self.requested,
            "notes": list(self.notes),
        }


def _cpu_device(requested: str, *notes: str) -> DeviceInfo:
    name = platform.processor() or platform.machine() or "cpu"
    return DeviceInfo(kind="cpu", name=name, requested=requested, notes=tuple(n for n in notes if n))


def _parse_preference(preference: Optional[str]) -> Tuple[str, int]:
    pref = (preference or "auto").strip().lower()
    if pref in ("", "auto", "gpu"):
        return "auto", 0
    if pref.startswith("cuda"):
        idx = 0
        if ":" in pref:
            try:
                idx = int(pref.split(":", 1)[1])
            except ValueError:
                idx = 0
        return "cuda", max(0, idx)
    if pref.isdigit():  # "0", "1" (ultralytics style)
        return "cuda", int(pref)
    if pref in ("mps", "metal", "apple"):
        return "mps", 0
    if pref == "cpu":
        return "cpu", 0
    logger.warning("Unknown device preference %r; using auto", preference)
    return "auto", 0


def _cuda_info(index: int) -> Optional[DeviceInfo]:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return None
    try:
        if not torch.cuda.is_available():
            return None
        count = torch.cuda.device_count()
        if index >= count:
            logger.warning("Requested cuda:%d but only %d CUDA device(s); using cuda:0", index, count)
            index = 0
        props = torch.cuda.get_device_properties(index)
        vram_gb = float(getattr(props, "total_memory", 0)) / (1024 ** 3)
        major = int(getattr(props, "major", 0))
        return DeviceInfo(
            kind="cuda",
            name=str(props.name),
            index=index,
            vram_gb=vram_gb,
            supports_half=major >= 6,  # Pascal+ have usable fp16
        )
    except Exception as exc:  # pragma: no cover - hardware specific
        logger.warning("CUDA probe failed: %s", exc)
        return None


def _mps_info() -> Optional[DeviceInfo]:
    try:
        import torch
    except Exception:  # pragma: no cover
        return None
    try:
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            return None
    except Exception:  # pragma: no cover
        return None
    chip = platform.processor() or platform.machine() or "Apple Silicon"
    # fp16 on MPS through ultralytics is not reliable across torch releases;
    # keep fp32 (MPS is memory-bandwidth bound either way).
    return DeviceInfo(kind="mps", name=f"Apple MPS ({chip})", supports_half=False)


def select_device(preference: Optional[str] = "auto") -> DeviceInfo:
    """Pick the inference device.

    ``VH_DEVICE`` (when set and not ``auto``) overrides ``preference``.
    Order for ``auto``: CUDA -> MPS -> CPU. An unavailable explicit request
    falls back to the best available device with a warning.
    """
    env_pref = os.environ.get(DEVICE_ENV, "").strip()
    requested = env_pref if env_pref and env_pref.lower() != "auto" else (preference or "auto")
    kind, index = _parse_preference(requested)

    if kind == "cpu":
        return _cpu_device(str(requested))
    if kind in ("cuda", "auto"):
        info = _cuda_info(index)
        if info is not None:
            return replace(info, requested=str(requested))
        if kind == "cuda":
            logger.warning("CUDA requested (%s) but not available; falling back", requested)
    if kind in ("mps", "auto", "cuda"):
        info = _mps_info()
        if info is not None:
            return replace(info, requested=str(requested))
        if kind == "mps":
            logger.warning("MPS requested but not available; using CPU")
    return _cpu_device(str(requested), "no accelerator available" if kind != "cpu" else "")


# ----------------------------------------------------------------------
# ffmpeg capability probes
# ----------------------------------------------------------------------


def ffmpeg_binary() -> str:
    """Path of ffmpeg (``VH_FFMPEG`` overrides PATH lookup)."""
    return os.environ.get(FFMPEG_ENV) or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_binary() -> str:
    """Path of ffprobe (``VH_FFPROBE`` overrides PATH lookup)."""
    return os.environ.get(FFPROBE_ENV) or shutil.which("ffprobe") or "ffprobe"


def _run_quiet(cmd: List[str], timeout: float = _PROBE_TIMEOUT_S) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, str(exc)
    out = (proc.stdout or b"").decode("utf-8", "replace") + (proc.stderr or b"").decode("utf-8", "replace")
    return int(proc.returncode), out


@lru_cache(maxsize=1)
def ffmpeg_version() -> Tuple[int, int]:
    """(major, minor) of the ffmpeg binary; (0, 0) when unknown."""
    code, out = _run_quiet([ffmpeg_binary(), "-hide_banner", "-version"])
    if code != 0:
        return (0, 0)
    first = out.splitlines()[0] if out else ""
    # "ffmpeg version 6.1.1-3ubuntu5 Copyright..." or "ffmpeg version n7.0 ..."
    try:
        token = first.split("version", 1)[1].strip().split()[0].lstrip("nN")
        parts = token.replace("-", ".").split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    except Exception:
        return (0, 0)


@lru_cache(maxsize=1)
def available_encoders() -> frozenset:
    """Names of encoders compiled into ffmpeg (not necessarily usable)."""
    code, out = _run_quiet([ffmpeg_binary(), "-hide_banner", "-encoders"])
    names = set()
    if code == 0:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
                names.add(parts[1])
    return frozenset(names)


@lru_cache(maxsize=1)
def available_hwaccels() -> frozenset:
    """hwaccel methods compiled into ffmpeg (not necessarily usable)."""
    code, out = _run_quiet([ffmpeg_binary(), "-hide_banner", "-hwaccels"])
    if code != 0:
        return frozenset()
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return frozenset(ln for ln in lines if ":" not in ln and " " not in ln)


@lru_cache(maxsize=None)
def encoder_works(encoder: str) -> bool:
    """True when ``encoder`` can encode a tiny clip on this machine."""
    if encoder not in available_encoders():
        return False
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=25:d=0.2",
        "-c:v", encoder, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    code, out = _run_quiet(cmd)
    if code != 0:
        logger.debug("encoder %s unusable: %s", encoder, out.strip()[-300:])
    return code == 0


@lru_cache(maxsize=None)
def hwaccel_works(hwaccel: str) -> bool:
    """True when the ffmpeg hardware device for ``hwaccel`` initialises."""
    if hwaccel not in available_hwaccels():
        return False
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-init_hw_device", f"{hwaccel}=hw",
        "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04", "-f", "null", "-",
    ]
    code, out = _run_quiet(cmd)
    if code != 0:
        logger.debug("hwaccel %s unusable: %s", hwaccel, out.strip()[-300:])
    return code == 0


def encoder_preferences(device: Optional[DeviceInfo] = None) -> List[str]:
    """Ordered H.264 encoder candidates: h264_nvenc > h264_videotoolbox > libx264.

    ``device`` narrows the list (a CPU-only run on a machine with NVENC may
    still use NVENC: encoding is independent of where inference runs, so
    ``device`` is only used to skip candidates that cannot exist).
    """
    prefs: List[str] = []
    is_mac = sys.platform == "darwin"
    if not is_mac:
        prefs.append("h264_nvenc")
    if is_mac:
        prefs.append("h264_videotoolbox")
    prefs.append("libx264")
    if device is not None and device.kind == "mps" and "h264_nvenc" in prefs:
        prefs.remove("h264_nvenc")
    return prefs


def hwaccel_preferences() -> List[str]:
    """Ordered ffmpeg decode hwaccel candidates for this platform."""
    if sys.platform == "darwin":
        return ["videotoolbox"]
    return ["cuda"]


@lru_cache(maxsize=None)
def pick_encoder(preferred: Optional[str] = None) -> str:
    """First working encoder from :func:`encoder_preferences` (``libx264`` floor)."""
    candidates = ([preferred] if preferred else []) + encoder_preferences()
    for enc in candidates:
        if enc == "libx264":
            return enc
        if encoder_works(enc):
            return enc
    return "libx264"


@lru_cache(maxsize=1)
def pick_hwaccel() -> Optional[str]:
    """First working decode hwaccel, or None for software decode."""
    for accel in hwaccel_preferences():
        if hwaccel_works(accel):
            return accel
    return None


def clear_probe_caches() -> None:
    """Forget cached ffmpeg probes (tests / after driver changes)."""
    for fn in (ffmpeg_version, available_encoders, available_hwaccels, encoder_works, hwaccel_works,
               pick_encoder, pick_hwaccel):
        fn.cache_clear()
