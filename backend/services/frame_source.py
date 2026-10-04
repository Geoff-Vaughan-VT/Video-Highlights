"""Single-pass media ingest: probe, proxy + analysis audio + thumbnails, frame reader.

The 4K source is decoded **once** for analysis: :func:`build_proxy` runs one
ffmpeg command with up to three outputs

1. ``proxy_<H>p.mp4``: H.264 yuv420p, constant frame rate (frame index ==
   ``t * fps`` exactly), ``+faststart``, AAC audio. Every analysis stage and
   the browser player read this file.
2. ``audio_analysis.wav``: mono 16 kHz ``pcm_s16le`` for audio analysis.
3. ``thumbs/NNNN.jpg`` (optional): one thumbnail every 10 s (``NNNN`` * 10 =
   seconds) for the UI scrub bar.

Trimming uses input seeking (``-ss``/``-t`` before ``-i``), so a trimmed
window costs no extra pass. Hardware decode (NVDEC / VideoToolbox) and
encode (NVENC / VideoToolbox) are used when they really work, with an
automatic software retry when a hardware path fails mid-run.

:class:`FrameReader` decodes the proxy on a background thread into a
bounded queue and yields ``(frame_index, t, frame)`` items or batches.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import queue
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import device as device_mod

logger = logging.getLogger("videohighlights.frame_source")

ProgressCallback = Callable[[str, float, str, Optional[Dict[str, object]]], None]

AUDIO_ANALYSIS_FILENAME = "audio_analysis.wav"
AUDIO_ANALYSIS_RATE = 16000
THUMBS_DIRNAME = "thumbs"
THUMB_INTERVAL_S = 10.0
THUMB_HEIGHT = 180

_COMMON_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0, 100.0, 120.0)


class ProxyCancelled(RuntimeError):
    """Raised by :func:`build_proxy` when ``cancel_event`` was set."""


class ProxyError(RuntimeError):
    """ffmpeg failed to build the proxy (message carries the stderr tail)."""


# ----------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------


@dataclass
class VideoInfo:
    """Container/stream facts from ffprobe.

    ``width``/``height`` are the *coded* dimensions; ``display_width`` /
    ``display_height`` apply ``rotation`` (ffmpeg and OpenCV autorotate, so
    the display size is the pixel space every consumer sees).
    """

    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int
    codec: str
    has_audio: bool
    rotation: int = 0
    is_vfr: bool = False
    r_frame_rate: float = 0.0
    avg_frame_rate: float = 0.0
    audio_codec: Optional[str] = None
    pix_fmt: Optional[str] = None
    bit_rate: Optional[int] = None

    @property
    def display_width(self) -> int:
        return self.height if abs(self.rotation) % 180 == 90 else self.width

    @property
    def display_height(self) -> int:
        return self.width if abs(self.rotation) % 180 == 90 else self.height

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "fps": round(self.fps, 4),
            "duration_s": round(self.duration_s, 3),
            "frame_count": self.frame_count,
            "codec": self.codec,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "rotation": self.rotation,
            "is_vfr": self.is_vfr,
            "r_frame_rate": round(self.r_frame_rate, 4),
            "avg_frame_rate": round(self.avg_frame_rate, 4),
        }


def _parse_rate(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in ("0/0", "N/A"):
        return 0.0
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        try:
            return float(text)
        except ValueError:
            return 0.0


def _stream_rotation(stream: Dict[str, object]) -> int:
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    for side in stream.get("side_data_list") or []:
        if isinstance(side, dict) and "rotation" in side:
            try:
                # Display matrix rotation is counter-clockwise; normalise to 0..359.
                return int(-float(side["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    return 0


def probe_video(path: Union[str, Path]) -> VideoInfo:
    """Inspect ``path`` with ``ffprobe -print_format json``.

    ``fps`` is ``r_frame_rate`` for constant-rate streams; when
    ``avg_frame_rate`` disagrees by more than 1 % the stream is flagged
    ``is_vfr`` and ``fps`` is the average rate snapped to a common rate.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cmd = [
        device_mod.ffprobe_binary(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProxyError(f"ffprobe failed for {path}: {exc}") from exc
    if proc.returncode != 0:
        raise ProxyError(f"ffprobe failed for {path}: {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    if video is None:
        raise ProxyError(f"No video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    r_rate = _parse_rate(video.get("r_frame_rate"))
    avg_rate = _parse_rate(video.get("avg_frame_rate"))
    is_vfr = bool(r_rate and avg_rate and abs(r_rate - avg_rate) / max(r_rate, avg_rate) > 0.01)
    if is_vfr:
        fps = snap_fps(avg_rate)
    else:
        fps = r_rate or avg_rate
    if not fps or fps > 1000:
        fps = avg_rate if 0 < avg_rate <= 1000 else 30.0

    duration = _parse_rate(video.get("duration")) or _parse_rate(fmt.get("duration"))
    nb_frames = video.get("nb_frames")
    try:
        frame_count = int(nb_frames) if nb_frames not in (None, "N/A") else 0
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0 and duration > 0:
        frame_count = int(round(duration * fps))
    bit_rate = fmt.get("bit_rate")
    return VideoInfo(
        path=str(path),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=float(fps),
        duration_s=float(duration),
        frame_count=int(frame_count),
        codec=str(video.get("codec_name") or "unknown"),
        has_audio=audio is not None,
        rotation=_stream_rotation(video),
        is_vfr=is_vfr,
        r_frame_rate=float(r_rate),
        avg_frame_rate=float(avg_rate),
        audio_codec=str(audio.get("codec_name")) if audio is not None else None,
        pix_fmt=video.get("pix_fmt"),
        bit_rate=int(bit_rate) if str(bit_rate or "").isdigit() else None,
    )


def snap_fps(fps: float) -> float:
    """Snap ``fps`` to the nearest common broadcast rate when within 2 %."""
    if fps <= 0:
        return 30.0
    best = min(_COMMON_RATES, key=lambda r: abs(r - fps))
    if abs(best - fps) / best <= 0.02:
        return best
    return round(fps, 3)


# ----------------------------------------------------------------------
# Proxy
# ----------------------------------------------------------------------


@dataclass
class ProxyResult:
    """Outputs of :func:`build_proxy`.

    ``scale`` is ``proxy_width / source_width`` (display orientation); use
    ``source_width``/``source_height`` to map proxy pixels back to source.
    ``trim_start_s`` is the offset of proxy t=0 in the source timebase.
    """

    path: str
    width: int
    height: int
    fps: float
    scale: float
    audio_path: Optional[str]
    thumbs_dir: Optional[str]
    elapsed_s: float
    hwaccel_used: Optional[str]
    encoder_used: str
    source_width: int = 0
    source_height: int = 0
    duration_s: float = 0.0
    frame_count: int = 0
    trim_start_s: float = 0.0
    has_audio: bool = False
    source_info: Optional[VideoInfo] = None
    command: List[str] = field(default_factory=list)

    @property
    def scale_xy(self) -> Tuple[float, float]:
        """(proxy_w / source_w, proxy_h / source_h); differs from ``scale`` only by rounding."""
        sw = self.source_width or self.width
        sh = self.source_height or self.height
        return self.width / float(sw), self.height / float(sh)

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "scale": round(self.scale, 6),
            "audio_path": self.audio_path,
            "thumbs_dir": self.thumbs_dir,
            "elapsed_s": round(self.elapsed_s, 3),
            "hwaccel_used": self.hwaccel_used,
            "encoder_used": self.encoder_used,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "duration_s": round(self.duration_s, 3),
            "frame_count": self.frame_count,
            "trim_start_s": round(self.trim_start_s, 3),
            "has_audio": self.has_audio,
        }


def _video_encoder_args(encoder: str, fps: float, height: int) -> List[str]:
    gop = max(1, int(round(fps * 2)))
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "21",
                "-b:v", "0", "-g", str(gop), "-bf", "2"]
    if encoder == "h264_videotoolbox":
        mbps = 14 if height >= 1080 else 8 if height >= 720 else 4
        return ["-c:v", "h264_videotoolbox", "-b:v", f"{mbps}M", "-g", str(gop), "-allow_sw", "1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", str(gop)]


def _cfr_args(fps: float) -> List[str]:
    rate = f"{fps:.6f}".rstrip("0").rstrip(".")
    if device_mod.ffmpeg_version() >= (5, 1):
        return ["-fps_mode", "cfr", "-r", rate]
    return ["-vsync", "cfr", "-r", rate]


def _passthrough_args() -> List[str]:
    if device_mod.ffmpeg_version() >= (5, 1):
        return ["-fps_mode", "passthrough"]
    return ["-vsync", "passthrough"]


def _build_command(
    info: VideoInfo,
    out_dir: Path,
    *,
    height: int,
    fps: float,
    trim_start: Optional[float],
    duration: Optional[float],
    hwaccel: Optional[str],
    encoder: str,
    thumbs: bool,
    audio_wav: bool,
    proxy_name: str,
) -> List[str]:
    cmd: List[str] = [device_mod.ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
                      "-progress", "pipe:1", "-stats_period", "0.5"]
    if hwaccel:
        # Default hwaccel_output_format: frames are downloaded to system
        # memory, so the CPU scale/fps filters below keep working.
        cmd += ["-hwaccel", hwaccel]
    if trim_start and trim_start > 0:
        cmd += ["-ss", f"{trim_start:.3f}"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(info.path)]

    target_h = min(int(height), int(info.display_height)) if info.display_height else int(height)
    target_h -= target_h % 2
    scale = f"scale=-2:{target_h}:flags=bicubic" if target_h != info.display_height else "null"
    graph = f"[0:v:0]{scale},format=yuv420p"
    if thumbs:
        graph += f",split=2[pv][tv];[tv]fps=1/{THUMB_INTERVAL_S:g}:eof_action=pass,scale=-2:{THUMB_HEIGHT}[th]"
    else:
        graph += "[pv]"
    cmd += ["-filter_complex", graph]

    # Output 1: proxy
    cmd += ["-map", "[pv]"]
    if info.has_audio:
        cmd += ["-map", "0:a:0"]
    cmd += _video_encoder_args(encoder, fps, target_h)
    cmd += ["-pix_fmt", "yuv420p"] + _cfr_args(fps)
    if info.has_audio:
        if (info.audio_codec or "") == "aac" and not trim_start:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += ["-movflags", "+faststart", str(out_dir / proxy_name)]

    # Output 2: analysis audio
    if audio_wav and info.has_audio:
        cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(AUDIO_ANALYSIS_RATE), "-c:a", "pcm_s16le",
                str(out_dir / AUDIO_ANALYSIS_FILENAME)]

    # Output 3: thumbnails
    if thumbs:
        cmd += ["-map", "[th]"] + _passthrough_args() + ["-q:v", "5", "-start_number", "0",
                                                       str(out_dir / THUMBS_DIRNAME / "%04d.jpg")]
    return cmd


def _write_silent_wav(path: Path, duration_s: float, rate: int = AUDIO_ANALYSIS_RATE) -> None:
    n = max(0, int(round(duration_s * rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        chunk = b"\x00\x00" * min(n, rate * 10)
        remaining = n
        while remaining > 0:
            step = min(remaining, rate * 10)
            handle.writeframes(chunk[: step * 2])
            remaining -= step


def _drain(stream, sink: Deque[str]) -> None:
    try:
        for raw in iter(stream.readline, b""):
            sink.append(raw.decode("utf-8", "replace").rstrip())
    except Exception:  # pragma: no cover - pipe closed
        pass


def _stop_process(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """Ask ffmpeg to finish ('q' on stdin), then terminate, then kill."""
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=grace_s)


def _run_ffmpeg_with_progress(
    cmd: List[str],
    *,
    expected_duration_s: float,
    fps: float,
    progress_cb: Optional[ProgressCallback],
    cancel_event: Optional[threading.Event],
    stage: str = "proxy",
) -> Tuple[int, str]:
    """Run ``cmd`` (which must include ``-progress pipe:1``), reporting progress."""
    logger.debug("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_tail: Deque[str] = collections.deque(maxlen=60)
    drain = threading.Thread(target=_drain, args=(proc.stderr, stderr_tail), daemon=True)
    drain.start()
    started = time.monotonic()
    block: Dict[str, str] = {}
    last_emit = 0.0
    cancelled = False
    assert proc.stdout is not None
    try:
        for raw in iter(proc.stdout.readline, b""):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            line = raw.decode("utf-8", "replace").strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            block[key.strip()] = value.strip()
            if key != "progress":
                continue
            now = time.monotonic()
            done = value.strip() == "end"
            if progress_cb is None or (not done and now - last_emit < 0.5):
                block = {}
                continue
            out_us = _parse_rate(block.get("out_time_us") or block.get("out_time_ms"))
            out_s = max(0.0, out_us / 1e6)
            frame = int(_parse_rate(block.get("frame")))
            if out_s <= 0 and frame > 0 and fps > 0:
                out_s = frame / fps
            frac = 1.0 if done else (min(0.999, out_s / expected_duration_s) if expected_duration_s > 0 else 0.0)
            elapsed = now - started
            eta = (elapsed * (1.0 - frac) / frac) if 0 < frac < 1 else (0.0 if done else None)
            data: Dict[str, object] = {
                "frame": frame,
                "fps_processing": round(_parse_rate(block.get("fps")), 2),
                "speed": block.get("speed"),
                "out_time_s": round(out_s, 3),
                "elapsed_s": round(elapsed, 2),
                "eta_s": round(eta, 1) if eta is not None else None,
            }
            try:
                progress_cb(stage, frac, "Building analysis proxy", data)
            except Exception:  # pragma: no cover - callback bugs must not kill ffmpeg
                logger.exception("progress callback failed")
            last_emit = now
            block = {}
        if cancelled:
            _stop_process(proc)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            _stop_process(proc)
        drain.join(timeout=2.0)
    if cancelled:
        raise ProxyCancelled("proxy build cancelled")
    return code, "\n".join(stderr_tail)


def build_proxy(
    source: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    height: int = 1080,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    hwaccel: Optional[str] = "auto",
    encoder: Optional[str] = "auto",
    thumbs: bool = True,
    audio_wav: bool = True,
    target_fps: Optional[float] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> ProxyResult:
    """Decode ``source`` once and write the proxy, analysis WAV and thumbnails.

    Args:
        source: original (e.g. 4K) video.
        out_dir: run directory; files are written as ``proxy_<H>p.mp4``,
            ``audio_analysis.wav`` and ``thumbs/NNNN.jpg``.
        height: proxy height (never upscales; width keeps aspect, even).
        trim_start / trim_end: processing window in source seconds.
        hwaccel: ``"auto"`` (probe cuda / videotoolbox once), an explicit
            ffmpeg hwaccel name, or ``None``/``"none"`` for software decode.
        encoder: ``"auto"`` (h264_nvenc > h264_videotoolbox > libx264) or an
            explicit ffmpeg encoder name.
        target_fps: force the proxy frame rate (default: source rate,
            snapped for VFR sources).
        progress_cb: ``(stage, fraction, message, data)`` with ``data`` keys
            ``frame, fps_processing, out_time_s, elapsed_s, eta_s``.
        cancel_event: when set, ffmpeg is stopped and :class:`ProxyCancelled`
            is raised.
    """
    started = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(source)
    fps = float(target_fps) if target_fps else float(info.fps)

    start = max(0.0, float(trim_start or 0.0))
    end = float(trim_end) if trim_end is not None and float(trim_end) > 0 else None
    if info.duration_s > 0:
        start = min(start, max(0.0, info.duration_s - 1.0 / max(fps, 1.0)))
        if end is not None:
            end = min(end, info.duration_s)
    if end is not None and end <= start:
        raise ValueError(f"trim_end ({trim_end}) must be greater than trim_start ({trim_start})")
    window = (end - start) if end is not None else max(0.0, info.duration_s - start)
    duration_arg = (end - start) if end is not None else None

    hw_req = (hwaccel or "none").strip().lower()
    if hw_req == "auto":
        hw_used: Optional[str] = device_mod.pick_hwaccel()
    elif hw_req in ("none", "", "off", "cpu"):
        hw_used = None
    else:
        hw_used = hw_req
    enc_req = (encoder or "auto").strip().lower()
    enc_used = device_mod.pick_encoder() if enc_req == "auto" else enc_req

    target_h = min(int(height), int(info.display_height) or int(height))
    target_h -= target_h % 2
    proxy_name = f"proxy_{int(height)}p.mp4"
    thumbs_dir = out_dir / THUMBS_DIRNAME
    if thumbs:
        if thumbs_dir.exists():
            shutil.rmtree(thumbs_dir, ignore_errors=True)
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    attempts: List[Tuple[Optional[str], str]] = [(hw_used, enc_used)]
    if hw_used is not None or enc_used != "libx264":
        attempts.append((None, "libx264"))

    last_err = ""
    cmd: List[str] = []
    for attempt_hw, attempt_enc in attempts:
        cmd = _build_command(
            info, out_dir, height=height, fps=fps, trim_start=start, duration=duration_arg,
            hwaccel=attempt_hw, encoder=attempt_enc, thumbs=thumbs, audio_wav=audio_wav,
            proxy_name=proxy_name,
        )
        logger.info("Building %dp proxy of %s (hwaccel=%s, encoder=%s, window=%.1fs)",
                    target_h, info.path, attempt_hw or "none", attempt_enc, window)
        code, err = _run_ffmpeg_with_progress(
            cmd, expected_duration_s=window, fps=fps, progress_cb=progress_cb, cancel_event=cancel_event,
        )
        if code == 0 and (out_dir / proxy_name).is_file():
            hw_used, enc_used = attempt_hw, attempt_enc
            break
        last_err = err
        logger.warning("Proxy attempt (hwaccel=%s, encoder=%s) failed with code %s: %s",
                       attempt_hw, attempt_enc, code, err[-400:])
    else:
        raise ProxyError(f"ffmpeg proxy build failed: {last_err[-1200:]}")

    proxy_path = out_dir / proxy_name
    proxy_info = probe_video(proxy_path)
    audio_path: Optional[Path] = out_dir / AUDIO_ANALYSIS_FILENAME
    if audio_wav:
        if not info.has_audio or not audio_path.is_file():
            _write_silent_wav(audio_path, proxy_info.duration_s or window)
    else:
        audio_path = None

    src_w = int(info.display_width or proxy_info.width)
    src_h = int(info.display_height or proxy_info.height)
    result = ProxyResult(
        path=str(proxy_path),
        width=int(proxy_info.width),
        height=int(proxy_info.height),
        fps=float(proxy_info.fps or fps),
        scale=float(proxy_info.width) / float(src_w) if src_w else 1.0,
        audio_path=str(audio_path) if audio_path is not None else None,
        thumbs_dir=str(thumbs_dir) if thumbs else None,
        elapsed_s=time.monotonic() - started,
        hwaccel_used=hw_used,
        encoder_used=enc_used,
        source_width=src_w,
        source_height=src_h,
        duration_s=float(proxy_info.duration_s or window),
        frame_count=int(proxy_info.frame_count),
        trim_start_s=start,
        has_audio=bool(info.has_audio),
        source_info=info,
        command=cmd,
    )
    if progress_cb is not None:
        try:
            progress_cb("proxy", 1.0, "Proxy ready", {**result.as_dict(), "eta_s": 0.0})
        except Exception:  # pragma: no cover
            logger.exception("progress callback failed")
    logger.info("Proxy ready: %s (%dx%d @ %.3f fps, %d frames) in %.1fs",
                result.path, result.width, result.height, result.fps, result.frame_count, result.elapsed_s)
    return result


def proxy_result_from_file(path: Union[str, Path], *, source_size: Optional[Tuple[int, int]] = None,
                           trim_start_s: float = 0.0) -> ProxyResult:
    """Describe an existing proxy (e.g. when reusing a previous run's proxy).

    ``audio_path`` / ``thumbs_dir`` point at the ``audio_analysis.wav`` and
    ``thumbs/`` that :func:`build_proxy` writes next to the proxy, when they
    exist; otherwise None. As in :func:`build_proxy`, the wav may be the
    silent placeholder written for a source without audio: ``has_audio``
    (from probing the proxy) says whether there is real audio.
    """
    info = probe_video(path)
    src_w, src_h = source_size or (info.display_width, info.display_height)
    parent = Path(path).parent
    wav = parent / AUDIO_ANALYSIS_FILENAME
    audio_path = str(wav) if wav.is_file() and wav.stat().st_size > 0 else None
    thumbs = parent / THUMBS_DIRNAME
    thumbs_dir = str(thumbs) if thumbs.is_dir() else None
    return ProxyResult(
        path=str(path), width=info.display_width, height=info.display_height, fps=info.fps,
        scale=info.display_width / float(src_w) if src_w else 1.0, audio_path=audio_path, thumbs_dir=thumbs_dir,
        elapsed_s=0.0, hwaccel_used=None, encoder_used=info.codec, source_width=int(src_w),
        source_height=int(src_h), duration_s=info.duration_s, frame_count=info.frame_count,
        trim_start_s=float(trim_start_s), has_audio=info.has_audio, source_info=info,
    )


# ----------------------------------------------------------------------
# Frame reader
# ----------------------------------------------------------------------


@dataclass
class FrameItem:
    """One decoded frame: absolute frame index in the video, time, BGR pixels."""

    index: int
    t: float
    frame: np.ndarray


_SENTINEL = object()


class FrameReader:
    """Threaded OpenCV decoder with a bounded prefetch queue.

    Iterating yields :class:`FrameItem` in strictly increasing ``index``
    order; :meth:`batches` groups them. Skipped frames (``stride > 1``) are
    ``grab()``-ed, not converted to BGR. ``t = index / fps`` (exact for the
    CFR proxy).

    Use as a context manager (or call :meth:`close`) so the decode thread is
    always joined.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        prefetch: int = 32,
        cancel_event: Optional[threading.Event] = None,
        fps: Optional[float] = None,
    ) -> None:
        import cv2

        self.path = str(path)
        self.stride = max(1, int(stride))
        self.start_frame = max(0, int(start_frame))
        self.end_frame = None if end_frame is None else int(end_frame)
        self.cancel_event = cancel_event
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self.fps = float(fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max(1, int(prefetch)))
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self.decode_s = 0.0
        self.frames_decoded = 0

    # ------------------------------------------------------------------
    @property
    def expected_frames(self) -> int:
        """Number of frames this reader will yield (estimate from container)."""
        end = self.frame_count if self.end_frame is None else min(self.end_frame, self.frame_count or self.end_frame)
        span = max(0, end - self.start_frame)
        return int(math.ceil(span / self.stride)) if span else 0

    def _cancelled(self) -> bool:
        return self._stop.is_set() or (self.cancel_event is not None and self.cancel_event.is_set())

    def _seek(self) -> int:
        """Position the capture at ``start_frame``; returns the actual next index."""
        import cv2

        if self.start_frame <= 0:
            return 0
        if self.start_frame > 120:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.start_frame))
            pos = int(round(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if pos == self.start_frame:
                return pos
            logger.debug("Seek landed on %d instead of %d; grabbing forward", pos, self.start_frame)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        idx = 0
        while idx < self.start_frame and not self._cancelled():
            if not self._cap.grab():
                break
            idx += 1
        return idx

    def _put(self, item: object) -> bool:
        while not self._cancelled():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            idx = self._seek()
            while not self._cancelled():
                if self.end_frame is not None and idx >= self.end_frame:
                    break
                t0 = time.perf_counter()
                if (idx - self.start_frame) % self.stride == 0:
                    ok, frame = self._cap.read()
                    self.decode_s += time.perf_counter() - t0
                    if not ok or frame is None:
                        break
                    self.frames_decoded += 1
                    if not self._put(FrameItem(index=idx, t=idx / self.fps, frame=frame)):
                        break
                else:
                    ok = self._cap.grab()
                    self.decode_s += time.perf_counter() - t0
                    if not ok:
                        break
                idx += 1
        except BaseException as exc:  # pragma: no cover - surfaced to the consumer
            self._error = exc
        finally:
            try:
                self._cap.release()
            except Exception:
                pass
            # Always deliver the sentinel unless the consumer stopped us.
            while True:
                try:
                    self._queue.put(_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    if self._cancelled():
                        break

    def start(self) -> "FrameReader":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="FrameReader", daemon=True)
            self._thread.start()
        return self

    def __iter__(self) -> Iterator[FrameItem]:
        self.start()
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    break
                if self._thread is not None and not self._thread.is_alive() and self._queue.empty():
                    break
                continue
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]
        if self._error is not None:
            raise RuntimeError(f"Frame decode failed: {self._error}") from self._error

    def batches(self, batch_size: int) -> Iterator[List[FrameItem]]:
        """Yield lists of up to ``batch_size`` consecutive items."""
        batch: List[FrameItem] = []
        size = max(1, int(batch_size))
        for item in self:
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    def close(self) -> None:
        self._stop.set()
        # Unblock a producer waiting on a full queue.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        else:
            try:
                self._cap.release()
            except Exception:
                pass

    def __enter__(self) -> "FrameReader":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_frames_at(path: Union[str, Path], indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Decode a handful of specific frames (sequential grab; for small sets)."""
    import cv2

    wanted = sorted(set(int(i) for i in indices if int(i) >= 0))
    out: Dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(path))
    try:
        idx = 0
        for target in wanted:
            while idx < target:
                if not cap.grab():
                    return out
                idx += 1
            ok, frame = cap.read()
            if not ok:
                return out
            out[target] = frame
            idx += 1
    finally:
        cap.release()
    return out
