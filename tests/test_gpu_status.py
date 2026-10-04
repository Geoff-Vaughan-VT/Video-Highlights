from __future__ import annotations

import subprocess
import sys
import types
from typing import Dict, List

import pytest

from backend.services import gpu_status as gs

ENCODERS_NVIDIA = """Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 A....D aac                  AAC (Advanced Audio Coding)
"""
ENCODERS_MAC = """Encoders:
 V....D libx264              libx264 H.264
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)
 V....D hevc_videotoolbox    VideoToolbox H.265 Encoder (codec hevc)
"""
HWACCELS_NVIDIA = "Hardware acceleration methods:\nvdpau\ncuda\nvaapi\n\n"
HWACCELS_MAC = "Hardware acceleration methods:\nvideotoolbox\n"
SMI_4090 = "NVIDIA GeForce RTX 4090, 560.94, 24564, 1200, 3, 41\n"


def _fake_run(responses: Dict[str, object]):
    """Return a subprocess.run stand-in keyed by the interesting argv token."""

    def run(cmd: List[str], *args, **kwargs):
        key = "nvidia-smi" if cmd[0] == "nvidia-smi" else cmd[-1] if "ffmpeg" in str(cmd[0]) else cmd[0]
        value = responses.get(key)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout=str(value), stderr="")

    return run


def _fake_torch(cuda: bool, mps: bool, names: List[str] | None = None) -> types.ModuleType:
    torch = types.ModuleType("torch")
    names = names or []
    torch.__version__ = "2.9.0"
    torch.version = types.SimpleNamespace(cuda="12.8" if cuda else None)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        device_count=lambda: len(names),
        get_device_name=lambda index: names[index],
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_built=lambda: mps, is_available=lambda: mps))
    return torch


@pytest.fixture(autouse=True)
def _no_device_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VH_DEVICE", raising=False)
    monkeypatch.setattr(gs, "ffmpeg_exe", lambda: "ffmpeg")


def test_parse_helpers() -> None:
    assert gs.parse_hwaccels(HWACCELS_NVIDIA) == ["vdpau", "cuda", "vaapi"]
    assert gs.parse_hwaccels("") == []
    enc = gs.parse_encoders(ENCODERS_NVIDIA)
    assert enc == {"h264_nvenc": True, "hevc_nvenc": True, "h264_videotoolbox": False, "hevc_videotoolbox": False, "libx264": True}
    assert gs.parse_encoders(ENCODERS_MAC)["h264_videotoolbox"] is True


def test_nvidia_rig(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, False, ["NVIDIA GeForce RTX 4090"]))
    monkeypatch.setattr(gs.subprocess, "run", _fake_run(
        {"nvidia-smi": SMI_4090, "-encoders": ENCODERS_NVIDIA, "-hwaccels": HWACCELS_NVIDIA}))
    monkeypatch.setattr(gs.platform, "system", lambda: "Windows")
    status = gs.get_gpu_status()
    # v1 keys preserved
    for key in ("ready", "rendering_ready", "torch", "nvidia_smi", "ffmpeg_nvenc", "recommendation"):
        assert key in status
    assert status["ready"] is True and status["rendering_ready"] is True
    assert status["ffmpeg_nvenc"]["available"] is True
    assert status["hwaccels"] == ["vdpau", "cuda", "vaapi"]
    assert status["encoders"]["h264_nvenc"] is True
    assert status["recommended_device"] == "cuda"
    assert status["recommended_encoder"] == "h264_nvenc"
    assert status["recommended_hwaccel"] == "cuda"
    assert status["mps_available"] is False
    assert status["hardware_class"] == "rtx_4090"


def test_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, True))
    monkeypatch.setattr(gs.subprocess, "run", _fake_run(
        {"-encoders": ENCODERS_MAC, "-hwaccels": HWACCELS_MAC, "sysctl": "Apple M2 Ultra\n"}))
    monkeypatch.setattr(gs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gs.platform, "machine", lambda: "arm64")
    status = gs.get_gpu_status()
    assert status["ready"] is False and status["rendering_ready"] is False
    assert status["nvidia_smi"]["available"] is False
    assert status["mps_available"] is True and status["torch"]["mps_available"] is True
    assert status["recommended_device"] == "mps"
    assert status["recommended_encoder"] == "h264_videotoolbox"
    assert status["recommended_hwaccel"] == "videotoolbox"
    assert status["platform"]["cpu_brand"] == "Apple M2 Ultra"
    assert status["hardware_class"] == "apple_m2_ultra"
    assert "MPS" in status["recommendation"]


def test_nvenc_listed_but_no_gpu_falls_back_to_x264(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, False))
    monkeypatch.setattr(gs.subprocess, "run", _fake_run({"-encoders": ENCODERS_NVIDIA, "-hwaccels": HWACCELS_NVIDIA}))
    monkeypatch.setattr(gs.platform, "system", lambda: "Linux")
    status = gs.get_gpu_status()
    assert status["encoders"]["h264_nvenc"] is True  # listed by ffmpeg...
    assert status["rendering_ready"] is False  # ...but no NVIDIA GPU
    assert status["recommended_encoder"] == "libx264"
    assert status["recommended_hwaccel"] is None
    assert status["recommended_device"] == "cpu"
    assert status["hardware_class"] == "cpu_8core"


def test_no_ffmpeg_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = types.ModuleType("torch")  # import works, attribute access fails
    monkeypatch.setitem(sys.modules, "torch", broken)
    monkeypatch.setattr(gs.subprocess, "run", _fake_run({}))
    status = gs.get_gpu_status()
    assert status["torch"]["error"]
    assert status["hwaccels"] == []
    assert not any(status["encoders"].values())
    assert status["ffmpeg_nvenc"]["available"] is False
    assert status["recommended_encoder"] == "mpeg4"
    assert status["hardware_class"] == "cpu_8core"


def test_vh_device_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, False, ["NVIDIA GeForce RTX 4080"]))
    monkeypatch.setattr(gs.subprocess, "run", _fake_run({"-encoders": "", "-hwaccels": ""}))
    monkeypatch.setenv("VH_DEVICE", "cpu")
    assert gs.get_gpu_status()["recommended_device"] == "cpu"
    monkeypatch.setenv("VH_DEVICE", "auto")
    assert gs.get_gpu_status()["recommended_device"] == "cuda"


def test_real_probe_never_raises() -> None:
    status = gs.get_gpu_status()
    assert isinstance(status["encoders"], dict)
    assert status["hardware_class"]
