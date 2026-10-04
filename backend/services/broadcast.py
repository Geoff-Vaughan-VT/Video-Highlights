"""Broadcast polish: story-aware clip boundaries and the highlight reel.

Two editor's-craft capabilities:

1. **Boundary refinement** - a highlight starts where the *move* began (the
   dead ball or the change of attacking direction that launched it), and it
   ends when the *emotion* resolves (crowd noise decays back to baseline),
   not at fixed offsets.
2. **Reel building** - a broadcast-style montage built with ffmpeg only (no
   moviepy): chronological clips joined with ``xfade``/``acrossfade``,
   loudness-normalized audio, slow-motion replays after goals with a "GOAL"
   banner, optional title cards and cold open. Clips without an audio stream
   get generated silence. Large reels (> 15 clips) use per-clip fades and
   the concat demuxer so the filter graph stays small.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .event_clip_renderer import concat_clips_ffmpeg, probe_media, select_h264_encoder
from .ffmpeg_tools import ffmpeg_exe

LOGGER = logging.getLogger("videohighlights.broadcast")

AUDIO_ANALYSIS_WAV = "audio_analysis.wav"


@dataclass
class BroadcastConfig:
    # Story-aware starts.
    max_lookback_s: float = 15.0
    preroll_s: float = 1.5
    direction_scan_step_s: float = 0.5
    # Emotion-aware endings.
    min_post_s: float = 4.0
    max_post_s: float = 8.0
    max_post_goal_s: float = 12.0
    decay_fraction: float = 0.25  # end when RMS falls below baseline + frac*(peak-baseline)
    decay_sustain_s: float = 1.0
    # Reel construction.
    cold_open_s: float = 3.0
    crossfade_s: float = 0.5
    replay_speed: float = 0.5  # 0.5 -> setpts=2.0*PTS
    replay_pre_s: float = 4.0
    replay_post_s: float = 1.0
    replay_audio_volume: float = 0.35
    fade_out_s: float = 1.0
    title_card_s: float = 1.5
    max_xfade_clips: int = 15
    # Audio normalization: per-clip two-pass gain (audio-only volumedetect
    # scan, then volume=) to this mean level, peaks kept under -1 dBFS (no
    # limiter in the xfade graph: its lookahead stalls the shared graph). Cheap and robust to silent clips (single-pass
    # loudnorm returns NaN on generated silence; dynaudnorm cost ~25% of the
    # reel encode time).
    normalize_audio: bool = True
    audio_target_mean_db: float = -20.0
    audio_max_gain_db: float = 18.0
    # Optional extra filter applied to the final mix (e.g. "dynaudnorm").
    audio_normalize_filter: str = ""
    # Concat-path segments encoded in parallel.
    parallel_segments: int = 2


# ---------------------------------------------------------------------------
# Audio envelope
# ---------------------------------------------------------------------------


def _block_mean_squares_from_stream(read_block, hop: int) -> np.ndarray:
    """Mean square per ``hop``-sample block from a generator of float32 chunks."""
    out: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float64)
    for chunk in read_block():
        data = np.concatenate([carry, np.asarray(chunk, dtype=np.float64)])
        full = (len(data) // hop) * hop
        if full:
            out.append(np.mean(data[:full].reshape(-1, hop) ** 2, axis=1))
        carry = data[full:]
    if len(carry):
        out.append(np.array([float(np.mean(carry ** 2))]))
    return np.concatenate(out) if out else np.zeros(0)


def _wav_blocks(path: Path, hop_s: float) -> Optional[Tuple[np.ndarray, float]]:
    try:
        from scipy.io import wavfile

        rate, data = wavfile.read(str(path), mmap=True)
    except Exception as exc:
        LOGGER.debug("wav read failed for %s (%s)", path, exc)
        return None
    hop = max(1, int(round(hop_s * rate)))
    scale = 1.0
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)

    def _gen():
        step = hop * 4096
        for i in range(0, data.shape[0], step):
            block = np.asarray(data[i:i + step], dtype=np.float64)
            if block.ndim > 1:
                block = block.mean(axis=1)
            yield block / scale

    return _block_mean_squares_from_stream(_gen, hop), float(hop) / float(rate)


def _ffmpeg_blocks(path: Path, hop_s: float, sample_rate: int = 16000) -> Optional[Tuple[np.ndarray, float]]:
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "f32le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as exc:
        LOGGER.debug("ffmpeg audio decode unavailable (%s)", exc)
        return None
    hop = max(1, int(round(hop_s * sample_rate)))

    def _gen():
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(hop * 4 * 4096)
            if not raw:
                break
            usable = len(raw) - (len(raw) % 4)
            yield np.frombuffer(raw[:usable], dtype=np.float32)

    try:
        blocks = _block_mean_squares_from_stream(_gen, hop)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()
    if len(blocks) == 0:
        return None
    return blocks, float(hop) / float(sample_rate)


def compute_audio_envelope(
    video_path: Optional[str],
    audio_path: Optional[str] = None,
    *,
    hop_s: float = 0.1,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(times, rms) crowd-noise envelope (0.2 s windows every 0.1 s), or None.

    Source preference: explicit ``audio_path``; ``video_path`` itself when it
    is a ``.wav``; ``audio_analysis.wav`` next to the video (written by the
    proxy pass, so the 4K source is never decoded for audio); otherwise the
    video's audio decoded by ffmpeg (streamed, constant memory). No librosa.
    """
    candidates: List[Path] = []
    if audio_path:
        candidates.append(Path(audio_path))
    if video_path:
        vp = Path(video_path)
        if vp.suffix.lower() == ".wav":
            candidates.append(vp)
        else:
            sibling = vp.parent / AUDIO_ANALYSIS_WAV
            if sibling.is_file():
                candidates.append(sibling)
            candidates.append(vp)
    for path in candidates:
        if not path.is_file():
            continue
        result = _wav_blocks(path, hop_s) if path.suffix.lower() == ".wav" else None
        if result is None:
            result = _ffmpeg_blocks(path, hop_s)
        if result is None:
            continue
        ms, hop_actual = result
        if len(ms) < 2:
            continue
        # 2-block (0.2 s) window centred on each block boundary.
        prev = np.concatenate([[ms[0]], ms[:-1]])
        rms = np.sqrt((prev + ms) / 2.0)
        times = np.arange(len(ms), dtype=np.float64) * hop_actual
        LOGGER.info("audio envelope from %s: %d frames (%.1fs)", path.name, len(rms), times[-1])
        return times, rms.astype(np.float64)
    LOGGER.info("audio envelope unavailable for %s; using fixed clip endings", video_path or audio_path)
    return None


# ---------------------------------------------------------------------------
# Boundary refinement
# ---------------------------------------------------------------------------


def story_start(
    event_t: float,
    ball_track,
    segments: Sequence[object],
    goal_side: Optional[str],
    config: Optional[BroadcastConfig] = None,
) -> float:
    """Walk back from an event to where the move began.

    The start is the latest of: the end of the previous dead-ball period
    (restart/set piece/goal hold), the moment the attack toward the scored
    goal began (ball x-velocity turned toward it), and a hard lookback cap.
    """
    cfg = config or BroadcastConfig()
    floor_t = max(0.0, event_t - cfg.max_lookback_s)

    dead_ball_end = floor_t
    for seg in segments:
        state = getattr(seg, "state", None) or (seg.get("state") if isinstance(seg, dict) else "")
        end_s = float(getattr(seg, "end_s", None) or (seg.get("end_s") if isinstance(seg, dict) else 0.0))
        if end_s <= event_t and state != "in_play" and end_s > dead_ball_end:
            dead_ball_end = end_s

    direction_origin = floor_t
    if goal_side in {"left", "right"} and ball_track is not None and len(ball_track) >= 2:
        toward = -1.0 if goal_side == "left" else 1.0
        t = event_t
        while t > floor_t:
            vx, _vy = ball_track.velocity_at(t, window_s=0.6)
            if vx * toward <= 0.0 and abs(vx) > 15.0:
                # Ball was clearly moving the other way here: the attack
                # started after this point.
                direction_origin = t
                break
            t -= cfg.direction_scan_step_s

    start = max(floor_t, dead_ball_end, direction_origin) - cfg.preroll_s
    return max(0.0, min(start, event_t - 2.0))


def emotion_end(
    event_t: float,
    envelope: Optional[Tuple[np.ndarray, np.ndarray]],
    is_goal: bool,
    config: Optional[BroadcastConfig] = None,
) -> float:
    """End the clip when crowd noise has decayed back toward baseline."""
    cfg = config or BroadcastConfig()
    max_post = cfg.max_post_goal_s if is_goal else cfg.max_post_s
    if envelope is None:
        return event_t + max_post
    times, rms = envelope
    if len(times) < 8:
        return event_t + max_post

    baseline = float(np.median(rms))
    peak_lo = int(np.searchsorted(times, event_t))
    peak_hi = int(np.searchsorted(times, event_t + 3.0))
    peak = float(rms[peak_lo:peak_hi].max()) if peak_hi > peak_lo else baseline
    if peak <= baseline:
        return event_t + max_post
    threshold = baseline + cfg.decay_fraction * (peak - baseline)

    lo = int(np.searchsorted(times, event_t + cfg.min_post_s))
    hi = int(np.searchsorted(times, event_t + max_post))
    below_since: Optional[float] = None
    for i in range(lo, min(hi, len(times))):
        if rms[i] < threshold:
            if below_since is None:
                below_since = float(times[i])
            elif float(times[i]) - below_since >= cfg.decay_sustain_s:
                return below_since
        else:
            below_since = None
    return event_t + max_post


def refine_intervals(
    intervals: Sequence[Tuple[float, float]],
    event_rows: Sequence[Dict[str, object]],
    ball_track,
    segments: Sequence[object],
    envelope: Optional[Tuple[np.ndarray, np.ndarray]],
    duration_s: float,
    config: Optional[BroadcastConfig] = None,
) -> List[Tuple[float, float]]:
    """Refine clip boundaries around known events.

    ``event_rows``: dicts with ``t``, ``event_type`` and optional ``side``
    (goals/cards/set-piece kicks) in the same timebase as ``intervals``.
    Intervals containing an event get a story-aware start (goals) and an
    emotion-aware end; other intervals keep their start and get the audio
    ending when available.
    """
    cfg = config or BroadcastConfig()
    refined: List[Tuple[float, float]] = []
    for start_s, end_s in intervals:
        row = next(
            (r for r in event_rows if start_s <= float(r.get("t", -1.0)) <= end_s), None
        )
        new_start, new_end = start_s, end_s
        if row is not None:
            event_t = float(row["t"])
            is_goal = str(row.get("event_type")) == "goal"
            if is_goal:
                new_start = min(
                    start_s,
                    story_start(event_t, ball_track, segments, row.get("side"), cfg),
                )
            new_end = max(end_s, emotion_end(event_t, envelope, is_goal, cfg))
        elif envelope is not None:
            mid = (start_s + end_s) / 2.0
            new_end = max(end_s, emotion_end(mid, envelope, False, cfg))
        refined.append((max(0.0, new_start), min(duration_s, new_end)))
    # Boundary growth can create overlaps; merge them.
    refined.sort()
    merged: List[Tuple[float, float]] = []
    for s, e in refined:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged




# ---------------------------------------------------------------------------
# Reel building (ffmpeg only)
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@dataclass
class _Segment:
    kind: str  # body | replay | title | cold_open
    path: Optional[str]
    start: float  # seconds into the source clip
    src_dur: float  # seconds read from the source clip
    out_dur: float  # seconds in the reel
    has_audio: bool
    speed: float = 1.0
    banner: Optional[str] = None
    label: Optional[str] = None
    gain_db: float = 0.0


def _measure_volume(path: str, start: float, duration: float) -> Optional[Tuple[float, float]]:
    """(mean_volume, max_volume) in dB from an audio-only scan of the window."""
    import re

    cmd = [ffmpeg_exe(), "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
           "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except Exception:
        return None
    mean = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", result.stderr or "")
    peak = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", result.stderr or "")
    if not mean or not peak or "inf" in mean.group(1):
        return None
    return float(mean.group(1)), float(peak.group(1))


def _assign_gains(segments: Sequence[_Segment], cfg: BroadcastConfig) -> None:
    from concurrent.futures import ThreadPoolExecutor

    todo = [seg for seg in segments if seg.has_audio and seg.path and seg.kind != "title"]
    if not todo or not cfg.normalize_audio:
        return
    with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
        results = list(pool.map(lambda sg: _measure_volume(str(sg.path), sg.start, sg.src_dur), todo))
    for seg, res in zip(todo, results):
        if res is None:
            continue
        mean_db, peak_db = res
        gain = cfg.audio_target_mean_db - mean_db
        gain = min(gain, -1.0 - peak_db)  # never push peaks past -1 dBFS
        seg.gain_db = float(max(-cfg.audio_max_gain_db, min(cfg.audio_max_gain_db, gain)))


def _find_font() -> Optional[str]:
    explicit = os.getenv("VH_FONT_FILE", "").strip()
    for candidate in ([explicit] if explicit else []) + list(_FONT_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _safe_text(text: str) -> str:
    """drawtext-safe label (letters, digits, spaces, dashes)."""
    cleaned = "".join(ch if (ch.isalnum() or ch in " -") else " " for ch in str(text))
    return " ".join(cleaned.split()).upper()[:40] or "HIGHLIGHT"


def _drawtext(text: str, font: Optional[str], size: str, x: str, y: str, box_alpha: float = 0.55,
              border: int = 14) -> str:
    font_opt = ""
    if font:
        font_opt = "fontfile='" + font.replace("\\", "/").replace("'", "") + "':"
    return (
        f"drawtext={font_opt}text='{_safe_text(text)}':fontcolor=white:fontsize={size}"
        f":box=1:boxcolor=black@{box_alpha}:boxborderw={border}:x={x}:y={y}"
    )


def _atempo_chain(speed: float) -> List[str]:
    parts: List[str] = []
    remaining = float(speed)
    while remaining < 0.5 - 1e-9:
        parts.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0 + 1e-9:
        parts.append("atempo=2.0")
        remaining /= 2.0
    if abs(remaining - 1.0) > 1e-6:
        parts.append(f"atempo={remaining:.4f}")
    return parts


def _segment_filters(seg: _Segment, input_idx: Optional[int], k: int, size: Tuple[int, int], fps: float,
                     cfg: BroadcastConfig, font: Optional[str], text_enabled: bool,
                     fades: Optional[float] = None, normalize: bool = False) -> List[str]:
    """Filter-graph lines producing ``[v{k}]`` and ``[a{k}]`` for one segment."""
    w, h = size
    dur = f"{seg.out_dur:.4f}"
    if seg.kind == "title":
        vparts = [f"color=c=0x101820:s={w}x{h}:r={fps:.4f}:d={dur}", "format=yuv420p", "setsar=1"]
        if text_enabled:
            vparts.append(_drawtext(seg.label or "HIGHLIGHT", font, "h/12", "(w-text_w)/2", "(h-text_h)/2",
                                    box_alpha=0.0, border=0))
        vsrc = ",".join(vparts)
    else:
        vparts = ["setpts=PTS-STARTPTS"]
        if abs(seg.speed - 1.0) > 1e-6:
            vparts.append(f"setpts={1.0 / seg.speed:.4f}*PTS")
        vparts += [
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1", f"fps={fps:.4f}", "format=yuv420p",
        ]
        if seg.banner and text_enabled:
            vparts.append(_drawtext(seg.banner, font, "h/9", "(w-text_w)/2", "h/12"))
            vparts.append(_drawtext("REPLAY", font, "h/24", "w-text_w-24", "24", box_alpha=0.4, border=8))
        vsrc = f"[{input_idx}:v]" + ",".join(vparts)
    # Segment durations are probed, so a plain trim (no tpad, ~15% slower)
    # pins every segment to its exact length for the xfade offsets.
    vtail = [f"trim=duration={dur}", "setpts=PTS-STARTPTS"]
    if fades:
        vtail += [f"fade=t=in:st=0:d={fades:.3f}",
                  f"fade=t=out:st={max(0.0, seg.out_dur - fades):.4f}:d={fades:.3f}"]
    lines = [vsrc + "," + ",".join(vtail) + f"[v{k}]"]

    if seg.has_audio and seg.path and input_idx is not None:
        aparts = ["asetpts=PTS-STARTPTS", "aresample=48000",
                  "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"]
        if abs(seg.speed - 1.0) > 1e-6:
            aparts += _atempo_chain(seg.speed) + [f"volume={cfg.replay_audio_volume:.3f}"]
        if normalize and abs(seg.gain_db) > 0.05:
            aparts.append(f"volume={seg.gain_db:.2f}dB")
        asrc = f"[{input_idx}:a]" + ",".join(aparts)
    else:
        asrc = "anullsrc=r=48000:cl=stereo"
    atail = ["apad", f"atrim=duration={dur}", "asetpts=PTS-STARTPTS"]
    if fades:
        atail += [f"afade=t=in:st=0:d={fades:.3f}",
                  f"afade=t=out:st={max(0.0, seg.out_dur - fades):.4f}:d={fades:.3f}"]
    if normalize and fades and seg.has_audio and cfg.normalize_audio:
        atail.append("alimiter=limit=0.95:level=disabled")
    lines.append(asrc + "," + ",".join(atail) + f"[a{k}]")
    return lines


def _run_ffmpeg(cmd: List[str], timeout_s: Optional[float]) -> Tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # pragma: no cover - missing binary etc.
        return False, str(exc)
    return result.returncode == 0, (result.stderr or "").strip()[-800:]


def _encode_args(encoder: Tuple[str, List[str]], fps: float) -> List[str]:
    return ["-c:v", encoder[0], *encoder[1], "-pix_fmt", "yuv420p", "-r", f"{fps:.4f}",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]


def _inputs_for(segments: Sequence[_Segment]) -> Tuple[List[str], List[Optional[int]]]:
    args: List[str] = []
    indices: List[Optional[int]] = []
    count = 0
    for seg in segments:
        if seg.kind == "title" or not seg.path:
            indices.append(None)
            continue
        args += ["-ss", f"{seg.start:.3f}", "-t", f"{seg.src_dur:.3f}", "-i", str(seg.path)]
        indices.append(count)
        count += 1
    return args, indices


def _reel_xfade(segments: List[_Segment], output_path: str, size: Tuple[int, int], fps: float,
                crossfade: float, encoder: Tuple[str, List[str]], cfg: BroadcastConfig,
                font: Optional[str], text_enabled: bool, timeout_s: Optional[float]) -> Tuple[bool, str]:
    input_args, indices = _inputs_for(segments)
    lines: List[str] = []
    for k, (seg, idx) in enumerate(zip(segments, indices)):
        lines += _segment_filters(seg, idx, k, size, fps, cfg, font, text_enabled, normalize=True)
    total = segments[0].out_dur
    vcur, acur = "v0", "a0"
    for k in range(1, len(segments)):
        if crossfade > 0:
            offset = total - crossfade
            lines.append(f"[{vcur}][v{k}]xfade=transition=fade:duration={crossfade:.3f}:offset={offset:.4f}[vx{k}]")
            lines.append(f"[{acur}][a{k}]acrossfade=d={crossfade:.3f}:c1=tri:c2=tri[ax{k}]")
            total += segments[k].out_dur - crossfade
        else:
            lines.append(f"[{vcur}][{acur}][v{k}][a{k}]concat=n=2:v=1:a=1[vx{k}][ax{k}]")
            total += segments[k].out_dur
        vcur, acur = f"vx{k}", f"ax{k}"
    fade_out = min(cfg.fade_out_s, total / 4.0)
    lines.append(f"[{vcur}]fade=t=out:st={max(0.0, total - fade_out):.4f}:d={fade_out:.3f}[vout]")
    lines.append(
        f"[{acur}]{cfg.audio_normalize_filter + ',' if cfg.audio_normalize_filter else ''}"
        f"afade=t=out:st={max(0.0, total - fade_out):.4f}:d={fade_out:.3f}[aout]"
    )
    graph = ";\n".join(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".ffgraph", delete=False, encoding="utf-8") as handle:
        handle.write(graph)
        graph_path = handle.name
    try:
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *input_args,
               "-filter_complex_script", graph_path, "-map", "[vout]", "-map", "[aout]",
               *_encode_args(encoder, fps), "-movflags", "+faststart", str(output_path)]
        ok, err = _run_ffmpeg(cmd, timeout_s)
        if not ok:
            LOGGER.debug("xfade reel graph failed:\n%s\n%s", graph, err)
        return ok, err
    finally:
        try:
            os.unlink(graph_path)
        except OSError:
            pass


def _reel_concat_fades(segments: List[_Segment], output_path: str, size: Tuple[int, int], fps: float,
                       crossfade: float, encoder: Tuple[str, List[str]], cfg: BroadcastConfig,
                       font: Optional[str], text_enabled: bool, timeout_s: Optional[float]) -> Tuple[bool, str]:
    """Per-segment normalized intermediates with fade in/out + concat demuxer copy."""
    work = Path(tempfile.mkdtemp(prefix="reel_", dir=str(Path(output_path).parent)))
    try:
        from concurrent.futures import ThreadPoolExecutor

        def _encode(k: int) -> Tuple[bool, str, str]:
            seg = segments[k]
            input_args, indices = _inputs_for([seg])
            lines = _segment_filters(seg, indices[0], 0, size, fps, cfg, font, text_enabled,
                                     fades=crossfade if crossfade > 0 else None, normalize=True)
            part = work / f"seg_{k:04d}.mp4"
            cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *input_args,
                   "-filter_complex", ";".join(lines), "-map", "[v0]", "-map", "[a0]",
                   *_encode_args(encoder, fps), str(part)]
            ok, err = _run_ffmpeg(cmd, timeout_s)
            return ok, err, str(part)

        with ThreadPoolExecutor(max_workers=max(1, int(cfg.parallel_segments))) as pool:
            results = list(pool.map(_encode, range(len(segments))))
        parts: List[str] = []
        for k, (ok, err, part) in enumerate(results):
            if not ok:
                return False, f"segment {k}: {err}"
            parts.append(part)
        list_path = work / "list.txt"
        list_path.write_text(
            "".join("file '" + p.replace("'", r"'\''") + "'\n" for p in parts), encoding="utf-8"
        )
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(list_path), "-c", "copy", "-movflags", "+faststart", str(output_path)]
        return _run_ffmpeg(cmd, timeout_s)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_broadcast_reel(
    clip_specs: Sequence[Dict[str, object]],
    output_path: str,
    config: Optional[BroadcastConfig] = None,
    *,
    crossfade_s: Optional[float] = None,
    slowmo_replays: bool = True,
    title_cards: bool = False,
    encoder: str = "auto",
    cold_open: bool = False,
    timeout_s: Optional[float] = None,
) -> Optional[str]:
    """Assemble the broadcast-style reel with ffmpeg (no moviepy).

    ``clip_specs``: chronological dicts with ``path`` (rendered clip file),
    ``start_s``/``end_s`` (timebase of ``occurred_at_s``), and optional
    ``event_type``, ``occurred_at_s``, ``confidence``, ``title``.

    * Clips are joined with ``xfade`` + ``acrossfade`` (``crossfade_s``,
      default ``config.crossfade_s`` = 0.5 s); the reel duration is
      ``sum(segments) - crossfade * (n - 1)``.
    * Goals get a slow-motion replay (``setpts=2.0*PTS`` for the default
      ``replay_speed`` 0.5, quiet ``atempo`` audio) with a "GOAL" banner.
    * Audio is resampled to 48 kHz stereo and normalized per clip (two-pass:
      audio-only ``volumedetect`` scan, then ``volume`` gain to
      ``audio_target_mean_db`` with peaks kept under -1 dBFS);
      clips without audio get generated silence.
    * Encoder: ``h264_nvenc`` -> ``h264_videotoolbox`` -> ``libx264``
      (probed once, cached) unless ``encoder`` names one.
    * More than ``config.max_xfade_clips`` segments: per-clip fades + concat
      demuxer. If the xfade graph fails: concat demuxer stream copy, then the
      per-clip-fade path.

    Returns ``output_path`` or None when nothing could be built.
    """
    import time

    cfg = config or BroadcastConfig()
    crossfade = float(cfg.crossfade_s if crossfade_s is None else crossfade_s)
    t_begin = time.perf_counter()

    specs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    for spec in clip_specs:
        path = spec.get("path")
        if not path or not Path(str(path)).is_file():
            continue
        info = probe_media(str(path))
        duration = float(info.get("video_duration") or info.get("duration") or 0.0)
        if not info.get("has_video") or duration < 0.3:
            LOGGER.warning("reel: skipping unreadable/empty clip %s", path)
            continue
        info["duration"] = duration
        specs.append((dict(spec), info))
    if not specs:
        return None

    first = specs[0][1]
    width = int(first.get("width") or 1280) // 2 * 2 or 1280
    height = int(first.get("height") or 720) // 2 * 2 or 720
    fps = float(first.get("fps") or 30.0)
    if not (1.0 <= fps <= 120.0):
        fps = 30.0

    segments: List[_Segment] = []
    if cold_open:
        def _score(item: Tuple[Dict[str, object], Dict[str, object]]) -> float:
            bonus = 2.0 if item[0].get("event_type") == "goal" else 0.0
            return bonus + float(item[0].get("confidence") or 0.0)

        best, best_info = max(specs, key=_score)
        dur = float(best_info["duration"])
        if best.get("occurred_at_s") is not None:
            local_t = float(best["occurred_at_s"]) - float(best.get("start_s") or 0.0)  # type: ignore[arg-type]
        else:
            local_t = dur / 2.0
        t0 = max(0.0, min(local_t - cfg.cold_open_s / 2.0, dur - cfg.cold_open_s))
        length = min(cfg.cold_open_s, dur - t0)
        if length >= 1.0:
            segments.append(_Segment("cold_open", str(best["path"]), t0, length, length,
                                     bool(best_info.get("has_audio"))))

    for spec, info in specs:
        dur = float(info["duration"])
        event_type = str(spec.get("event_type") or "")
        if title_cards:
            label = str(spec.get("title") or event_type.replace("_", " ") or "highlight")
            segments.append(_Segment("title", None, 0.0, cfg.title_card_s, cfg.title_card_s, False, label=label))
        segments.append(_Segment("body", str(spec["path"]), 0.0, dur, dur, bool(info.get("has_audio"))))
        if slowmo_replays and event_type == "goal" and spec.get("occurred_at_s") is not None:
            local_t = float(spec["occurred_at_s"]) - float(spec.get("start_s") or 0.0)  # type: ignore[arg-type]
            r0 = max(0.0, local_t - cfg.replay_pre_s)
            r1 = min(dur, local_t + cfg.replay_post_s)
            if r1 - r0 >= 1.0:
                speed = max(0.1, min(1.0, cfg.replay_speed))
                segments.append(_Segment("replay", str(spec["path"]), r0, r1 - r0, (r1 - r0) / speed,
                                         bool(info.get("has_audio")), speed=speed, banner="GOAL"))

    _assign_gains(segments, cfg)
    min_dur = min(seg.out_dur for seg in segments)
    xf = 0.0 if len(segments) < 2 else max(0.0, min(crossfade, 0.45 * min_dur))
    if encoder in ("auto", "h264_nvenc", "h264_videotoolbox"):
        enc = select_h264_encoder(prefer_gpu=True, requested=encoder)
    else:
        enc = select_h264_encoder(prefer_gpu=False, requested=encoder)
    font = _find_font()
    timeout = timeout_s if timeout_s is not None else max(300.0, 20.0 * sum(s.out_dur for s in segments))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _verify() -> bool:
        return out.is_file() and out.stat().st_size > 0 and probe_media(str(out)).get("duration", 0.0) > 0

    method = ""
    if len(segments) > cfg.max_xfade_clips:
        ok, err = _reel_concat_fades(segments, str(out), (width, height), fps, xf, enc, cfg, font, True, timeout)
        method = "concat_fades"
    else:
        ok, err = _reel_xfade(segments, str(out), (width, height), fps, xf, enc, cfg, font, True, timeout)
        method = "xfade"
        if not ok and any(s.banner or s.kind == "title" for s in segments):
            LOGGER.info("reel: retrying without text overlays (%s)", err[-200:])
            ok, err = _reel_xfade(segments, str(out), (width, height), fps, xf, enc, cfg, font, False, timeout)
        if not ok and enc[0] != "libx264":
            ok, err = _reel_xfade(segments, str(out), (width, height), fps, xf,
                                  ("libx264", ["-preset", "veryfast", "-crf", "21"]), cfg, font, False, timeout)
        if not ok:
            LOGGER.warning("reel: xfade graph failed (%s); falling back to concat demuxer", err[-300:])
            try:
                concat_clips_ffmpeg([str(s["path"]) for s, _ in specs], str(out), copy=True)
                ok, method = _verify(), "concat_copy"
            except Exception as exc:
                LOGGER.warning("reel: concat copy failed (%s)", exc)
                ok = False
            if not ok:
                ok, err = _reel_concat_fades(segments, str(out), (width, height), fps, xf,
                                             ("libx264", ["-preset", "veryfast", "-crf", "21"]),
                                             cfg, font, False, timeout)
                method = "concat_fades"
    if not ok or not _verify():
        LOGGER.warning("broadcast reel failed: %s", err[-300:] if not ok else "empty output")
        return None
    LOGGER.info(
        "broadcast reel written: %s (%d segments, %s, encoder %s, %.1fs)",
        output_path, len(segments), method, enc[0], time.perf_counter() - t_begin,
    )
    return str(output_path)
