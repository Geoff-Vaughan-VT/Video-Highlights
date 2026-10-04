"""ffmpeg clip cutting, concatenation and encoder selection.

Every cut seeks on the *input* (``-ss`` before ``-i``) and limits the output
with ``-t``, so ffmpeg only decodes the requested window instead of decoding
the whole file from 0:00 up to the cut point. Cuts are re-encoded with a fast
preset by default because stream-copy cuts only land exactly on keyframes;
``cut_clip_from_rendered(..., reencode=False)`` offers the stream-copy path
(``-avoid_negative_ts make_zero``) when frame accuracy does not matter.

Encoder selection (``h264_nvenc`` -> ``h264_videotoolbox`` -> ``libx264``) is
probed once per process with a real 0.2 s test encode (an encoder listed by
``ffmpeg -encoders`` may still fail without the matching GPU/driver) and
cached.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .ffmpeg_tools import ffmpeg_exe, ffprobe_exe
from ..utils import ensure_dir

LOGGER = logging.getLogger("videohighlights.event_clip_renderer")

# Encoder name -> quality/speed arguments for a fast, good-looking H.264.
ENCODER_ARGS: Dict[str, List[str]] = {
    "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
    "h264_videotoolbox": ["-b:v", "8M", "-allow_sw", "1"],
    "libx264": ["-preset", "veryfast", "-crf", "21"],
    "mpeg4": ["-q:v", "3"],
}
GPU_ENCODERS = ("h264_nvenc", "h264_videotoolbox")


def _ffmpeg_encoder_available(encoder_name: str) -> bool:
    """True when ``ffmpeg -encoders`` lists the encoder (not proof it works)."""
    return encoder_name.strip().lower() in _listed_encoders()


@lru_cache(maxsize=1)
def _listed_encoders() -> frozenset:
    try:
        result = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return frozenset()
    names = set()
    for line in f"{result.stdout}\n{result.stderr}".splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1].lower())
    return frozenset(names)


@lru_cache(maxsize=8)
def _encoder_works(encoder_name: str) -> bool:
    """Real 0.2 s test encode (GPU encoders need a GPU + driver)."""
    if encoder_name not in _listed_encoders():
        return False
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc2=size=256x144:rate=10:duration=0.2",
        "-c:v", encoder_name, *ENCODER_ARGS.get(encoder_name, []), "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except Exception:
        return False
    ok = result.returncode == 0
    LOGGER.debug("encoder probe %s: %s", encoder_name, "ok" if ok else (result.stderr or "").strip()[:200])
    return ok


def select_h264_encoder(prefer_gpu: bool = True, requested: str = "auto") -> Tuple[str, List[str]]:
    """Pick a working H.264 encoder: nvenc -> videotoolbox -> libx264 (-> mpeg4).

    ``requested`` may name an encoder explicitly; it is used when it works,
    otherwise the automatic chain applies. Results are cached per process.
    """
    chain: List[str] = []
    if requested and requested != "auto":
        chain.append(requested)
    if prefer_gpu:
        chain.extend(GPU_ENCODERS)
    chain.extend(["libx264", "mpeg4"])
    for name in chain:
        if _encoder_works(name):
            return name, list(ENCODER_ARGS.get(name, []))
    return "libx264", list(ENCODER_ARGS["libx264"])


def _codec_attempts(prefer_gpu: bool) -> List[Tuple[str, List[str]]]:
    """Ordered encoder attempts: the probed best first, then safe fallbacks."""
    best = select_h264_encoder(prefer_gpu=prefer_gpu)
    attempts = [best]
    for name in ("libx264", "mpeg4"):
        if name != best[0]:
            attempts.append((name, list(ENCODER_ARGS[name])))
    return attempts


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def probe_media(path: str) -> Dict[str, object]:
    """ffprobe summary: duration, width, height, fps, has_audio, has_video."""
    info: Dict[str, object] = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
                               "has_audio": False, "has_video": False}
    try:
        result = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate,duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        data = json.loads(result.stdout or "{}")
    except Exception as exc:
        LOGGER.debug("ffprobe failed for %s: %s", path, exc)
        return info
    try:
        info["duration"] = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        pass
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "audio":
            info["has_audio"] = True
        elif kind == "video" and not info["has_video"]:
            info["has_video"] = True
            info["width"] = int(stream.get("width") or 0)
            info["height"] = int(stream.get("height") or 0)
            for key in ("avg_frame_rate", "r_frame_rate"):
                raw = str(stream.get(key) or "0/0")
                try:
                    num, den = raw.split("/") if "/" in raw else (raw, "1")
                    fps = float(num) / float(den) if float(den) else 0.0
                except (TypeError, ValueError):
                    fps = 0.0
                if fps > 0:
                    info["fps"] = fps
                    break
            try:
                vdur = float(stream.get("duration") or 0.0)
                if vdur > 0:
                    info["video_duration"] = vdur
            except (TypeError, ValueError):
                pass
    return info


def probe_duration(path: str) -> float:
    return float(probe_media(path).get("duration") or 0.0)


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------


def build_command(
    video_path: str,
    output_path: str,
    start_s: float,
    end_s: float,
    *,
    include_audio: bool = True,
    codec: str = "libx264",
    codec_args: Optional[Sequence[str]] = None,
    copy: bool = False,
) -> List[str]:
    """ffmpeg command that decodes ONLY ``[start_s, end_s]``.

    ``-ss`` precedes ``-i`` (input seeking) and ``-t`` bounds the duration.
    """
    start_s = max(0.0, float(start_s))
    duration = float(end_s) - start_s
    if duration <= 0:
        raise ValueError("end_s must be greater than start_s")
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}",
        "-map", "0:v:0",
    ]
    if include_audio:
        cmd += ["-map", "0:a:0?"]
    if copy:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", codec, *(codec_args if codec_args is not None else ENCODER_ARGS.get(codec, [])),
                "-pix_fmt", "yuv420p"]
        if include_audio:
            cmd += ["-c:a", "aac", "-b:a", "160k"]
    if not include_audio:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(output_path)]
    return cmd


def _run_attempts(attempt_cmds: List[Tuple[str, List[str]]], out_file: Path) -> str:
    last_error = ""
    for label, cmd in attempt_cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return str(out_file.resolve())
        detail = (result.stderr or "").strip() or (result.stdout or "").strip() or f"rc={result.returncode}"
        last_error = f"{label}: {detail[-400:]}"
        LOGGER.debug("ffmpeg attempt failed (%s)", last_error)
    raise RuntimeError(f"ffmpeg failed ({last_error})")


def render_clip_ffmpeg(
    video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
    include_audio: bool = True,
    prefer_gpu: bool = True,
) -> str:
    """Re-encode ``[start_seconds, end_seconds]`` of ``video_path`` (input seek)."""
    start_s = max(0.0, float(start_seconds))
    end_s = float(end_seconds)
    if end_s <= start_s:
        raise ValueError("end_seconds must be greater than start_seconds")
    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))
    attempts = [
        (codec, build_command(video_path, str(out_file), start_s, end_s, include_audio=include_audio,
                              codec=codec, codec_args=args))
        for codec, args in _codec_attempts(prefer_gpu)
    ]
    try:
        return _run_attempts(attempts, out_file)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to render clip with ffmpeg ({exc})") from exc


def cut_clip_from_rendered(
    movie_path: str,
    start_s: float,
    end_s: float,
    out_path: str,
    *,
    reencode: bool = True,
    include_audio: bool = True,
    prefer_gpu: bool = True,
) -> str:
    """Cut a highlight from the finished follow-cam movie.

    ``reencode=True`` (default): frame-accurate fast re-encode of the window
    only. ``reencode=False``: stream copy with ``-avoid_negative_ts
    make_zero`` (instant; the start snaps to the previous keyframe), falling
    back to a re-encode if the copy fails.
    """
    out_file = Path(out_path)
    ensure_dir(str(out_file.parent))
    attempts: List[Tuple[str, List[str]]] = []
    if not reencode:
        attempts.append(("copy", build_command(movie_path, str(out_file), start_s, end_s,
                                               include_audio=include_audio, copy=True)))
    for codec, args in _codec_attempts(prefer_gpu):
        attempts.append((codec, build_command(movie_path, str(out_file), start_s, end_s,
                                              include_audio=include_audio, codec=codec, codec_args=args)))
    return _run_attempts(attempts, out_file)


def concat_clips_ffmpeg(
    clip_paths: List[str],
    output_path: str,
    include_audio: bool = True,
    *,
    copy: bool = False,
    prefer_gpu: bool = True,
) -> str:
    """Concatenate clips with the concat demuxer.

    ``copy=True`` tries a pure stream copy first (clips cut from the same
    movie share codec parameters); otherwise / on failure it re-encodes.
    """
    if not clip_paths:
        raise ValueError("clip_paths must not be empty")
    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = Path(handle.name)
        for path in clip_paths:
            escaped = str(Path(path).resolve()).replace("'", r"'\''")
            handle.write(f"file '{escaped}'\n")
    try:
        base = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_path)]
        tail_audio = ["-c:a", "aac"] if include_audio else ["-an"]
        attempts: List[Tuple[str, List[str]]] = []
        if copy:
            attempts.append(("copy", base + ["-c", "copy"] + ([] if include_audio else ["-an"])
                             + ["-movflags", "+faststart", str(out_file)]))
        for codec, args in _codec_attempts(prefer_gpu):
            attempts.append((codec, base + ["-c:v", codec, *args, "-pix_fmt", "yuv420p"] + tail_audio
                             + ["-movflags", "+faststart", str(out_file)]))
        try:
            return _run_attempts(attempts, out_file)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to concat clips ({exc})") from exc
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
