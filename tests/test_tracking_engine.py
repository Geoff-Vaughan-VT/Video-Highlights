from __future__ import annotations

import shutil
import sys
import threading
import types
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytest

from backend.services.detectors import PERSON, Detections, GroundTruthDetector
from backend.services.frame_source import build_proxy
from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match
from backend.services.tracking_engine import (
    ByteTrackLite,
    RowBuffer,
    TrackPoint,
    identity_metrics,
    legacy_track_video_tuple,
    legacy_views,
    make_tracker,
    select_focus,
    torso_histogram,
    track_video,
)
from backend.services.tracking_types import TrackingResult

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SPEC = SyntheticMatchSpec(width=1280, height=720, fps=25.0, duration_s=8.0, seed=7, goals=[(3.2, "right")])


@pytest.fixture(scope="module")
def match(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("tracking")
    truth = generate_synthetic_match(root / "match.mp4", SPEC, prefer_ffmpeg=True)
    proxy = build_proxy(root / "match.mp4", root / "run", height=360, thumbs=False)
    return {"root": root, "truth": truth, "proxy": proxy}


def _detector(match, **kw) -> GroundTruthDetector:
    params = dict(dropout=0.1, jitter_px=2.0, false_positive_rate=0.02, seed=1)
    params.update(kw)
    return GroundTruthDetector(match["truth"].tracking, **params)


@pytest.fixture(scope="module")
def tracked(match):
    events: List[tuple] = []
    result = track_video(match["proxy"], detector=_detector(match), batch_size=8,
                         progress_cb=lambda *a: events.append(a))
    return result, events


def _dominant_truth(result: TrackingResult, truth: TrackingResult) -> Dict[int, int]:
    metrics = identity_metrics(result, truth)
    return {int(v["dominant_id"]): gid for gid, v in metrics["players"].items() if v["dominant_id"] is not None}


# ----------------------------------------------------------------------
# Identity recovery on noisy synthetic detections
# ----------------------------------------------------------------------


def test_recovers_player_identities(match, tracked) -> None:
    result, _ = tracked
    metrics = identity_metrics(result, match["truth"].tracking)
    assert metrics["total"] == 11
    assert metrics["recovered_fraction"] >= 0.9, metrics
    assert all(v["switches"] <= 1 for v in metrics["players"].values()), metrics
    # No identity explosion from false positives / dropouts.
    assert len(result.players) <= 13


def test_recovers_identities_with_stride_and_another_seed(tmp_path: Path) -> None:
    spec = SyntheticMatchSpec(width=1280, height=720, fps=25.0, duration_s=6.0, seed=21, goals=[])
    truth = generate_synthetic_match(tmp_path / "m.mp4", spec)
    proxy = build_proxy(tmp_path / "m.mp4", tmp_path / "run", height=360, thumbs=False, audio_wav=False)
    det = GroundTruthDetector(truth.tracking, seed=3)
    for stride in (1, 2):
        result = track_video(proxy, detector=det, stride=stride)
        metrics = identity_metrics(result, truth.tracking)
        assert metrics["recovered_fraction"] >= 0.9, (stride, metrics)
        assert metrics["max_switches"] <= 1, (stride, metrics)
        assert result.vid_stride == stride
        assert all(np.all(np.diff(p.t) >= (stride / 25.0) - 1e-4) for p in result.players.values())


def test_coordinates_are_in_source_pixel_space(match, tracked) -> None:
    result, _ = tracked
    truth = match["truth"].tracking
    assert result.frame_size == (1280, 720)
    assert result.proxy_scale == pytest.approx(0.5)
    assert result.fps == pytest.approx(25.0)
    assert result.duration_s == pytest.approx(8.0, abs=0.05)
    assert result.processing_video_path == match["proxy"].path
    mapping = _dominant_truth(result, truth)
    errors = []
    for pid, gid in mapping.items():
        pred, gt = result.players[pid], truth.players[gid]
        idx = np.searchsorted(gt.t, pred.t)
        idx = np.clip(idx, 0, len(gt) - 1)
        errors.append(np.abs(pred.center_xy() - gt.center_xy()[idx]).mean())
    assert float(np.median(errors)) < 3.0  # source px (proxy is half size)
    for track in result.players.values():
        assert track.x2.max() <= 1280 and track.y2.max() <= 720
        assert np.all(np.diff(track.t) > 0)
        assert np.allclose(track.t * 25.0, np.round(track.t * 25.0), atol=1e-3)  # t = frame / fps


def test_ball_detections_are_kept_raw(match, tracked) -> None:
    result, _ = tracked
    truth_ball = match["truth"].tracking.ball
    ball = result.ball
    assert 0.8 * len(truth_ball) <= len(ball) <= len(truth_ball)  # 10 % ball dropout, nothing else lost
    assert np.all(ball.conf > 0) and np.all(ball.w > 0) and np.all(ball.h > 0)
    idx = np.clip(np.searchsorted(truth_ball.t, ball.t), 0, len(truth_ball) - 1)
    err = np.hypot(ball.x - truth_ball.x[idx], ball.y - truth_ball.y[idx])
    assert float(np.median(err)) < 3.0
    assert float(np.median(ball.w)) == pytest.approx(12.0, abs=1.5)  # source px, not proxy px
    # Ball never leaks into person tracks.
    heights = np.concatenate([p.box_heights() for p in result.players.values()])
    assert heights.min() > 20


def test_progress_timings_and_detector_info(tracked) -> None:
    result, events = tracked
    assert events[0][0] == "tracking" and events[0][1] == 0.0
    last = events[-1]
    assert last[1] == pytest.approx(1.0)
    for key in ("frames_done", "frames_total", "fps_processing", "eta_s", "device", "batch"):
        assert key in last[3]
    assert last[3]["frames_done"] == last[3]["frames_total"] == 200
    for key in ("decode_s", "detect_s", "track_s", "total_s"):
        assert result.timings[key] >= 0
    assert result.timings["cancelled"] is False
    assert result.detector["name"] == "ground_truth"
    assert result.detector["tracker"] == "bytetrack"
    stats = result.detector["tracking_stats"]
    assert stats["identities"] == len(result.players)
    assert "dropped_short_tracks" in stats


# ----------------------------------------------------------------------
# Focus selection
# ----------------------------------------------------------------------


def test_focus_roi_selects_player_at_requested_time(match) -> None:
    truth = match["truth"].tracking
    gid = 4
    t_roi = 2.0
    box = truth.players[gid].box_at(t_roi)
    pad = 6.0
    roi = {
        "x1_norm": (box[0] - pad) / 1280, "y1_norm": (box[1] - pad) / 720,
        "x2_norm": (box[2] + pad) / 1280, "y2_norm": (box[3] + pad) / 720, "t": t_roi,
    }
    result = track_video(match["proxy"], detector=_detector(match), focus_roi=roi)
    mapping = _dominant_truth(result, truth)
    assert result.focus_track_id is not None
    assert mapping[result.focus_track_id] == gid
    assert result.detector["focus_selection"]["method"] == "roi"

    # Pixel {x, y, w, h} form (source pixels), drawn at t=0.
    box0 = truth.players[9].box_at(0.0)
    roi_px = {"x": box0[0] - 4, "y": box0[1] - 4, "w": box0[2] - box0[0] + 8, "h": box0[3] - box0[1] + 8}
    result_px = track_video(match["proxy"], detector=_detector(match), focus_roi=roi_px)
    assert _dominant_truth(result_px, truth)[result_px.focus_track_id] == 9


def test_focus_track_id_passthrough_and_fallback(match, tracked) -> None:
    result, _ = tracked
    some_id = sorted(result.players)[2]
    again = track_video(match["proxy"], detector=_detector(match), focus_track_id=some_id)
    assert again.focus_track_id == some_id
    assert again.detector["focus_selection"]["method"] == "track_id"
    assert result.focus_track_id is None  # nothing requested
    far_roi = {"x1_norm": 0.0, "y1_norm": 0.0, "x2_norm": 0.02, "y2_norm": 0.02}
    fallback = track_video(match["proxy"], detector=_detector(match), focus_roi=far_roi)
    assert fallback.focus_track_id == fallback.longest_track_id()
    assert fallback.detector["focus_selection"]["method"] == "fallback_longest"


def test_select_focus_retargets_loaded_tracks(tmp_path: Path, match) -> None:
    """Public wrapper used by the pipeline when tracks are reused (no re-detection)."""
    import backend.services.tracking_engine as te

    truth = match["truth"].tracking
    truth.save(tmp_path)
    loaded = TrackingResult.load(tmp_path)
    assert te._select_focus is not None  # the private name stays available

    sel = select_focus(loaded, focus_track_id=3)
    assert sel == {"method": "track_id", "requested": 3, "track_id": 3}
    assert loaded.focus_track_id == 3

    box = truth.players[6].box_at(1.0)
    roi = {"x1_norm": (box[0] - 4) / 1280, "y1_norm": (box[1] - 4) / 720,
           "x2_norm": (box[2] + 4) / 1280, "y2_norm": (box[3] + 4) / 720, "t": 1.0}
    sel = select_focus(loaded, focus_roi=roi)
    assert sel["method"] == "roi" and loaded.focus_track_id == 6

    # Unknown id without an ROI: no focus; unknown id with a far ROI: longest track.
    assert select_focus(loaded, focus_track_id=999) == {"method": "none"}
    assert loaded.focus_track_id is None
    far = {"x1_norm": 0.0, "y1_norm": 0.0, "x2_norm": 0.02, "y2_norm": 0.02}
    sel = select_focus(loaded, focus_track_id=999, focus_roi=far)
    assert sel["method"] == "fallback_longest" and loaded.focus_track_id == loaded.longest_track_id()
    assert select_focus(loaded) == {"method": "none"}


# ----------------------------------------------------------------------
# Stitching
# ----------------------------------------------------------------------


class _BlackoutDetector:
    """Hides one truth player for a time window so the tracker must drop it."""

    def __init__(self, base: GroundTruthDetector, box_fn, t0: float, t1: float, fps: float) -> None:
        self.base, self.box_fn, self.t0, self.t1, self.fps = base, box_fn, t0, t1, fps
        self.batch = base.batch

    def info(self) -> Dict[str, object]:
        return {"name": "blackout"}

    def detect_batch(self, frames: List[np.ndarray], *, frame_indices: Sequence[int]) -> List[Detections]:
        out = []
        for det, idx in zip(self.base.detect_batch(frames, frame_indices=frame_indices), frame_indices):
            t = idx / self.fps
            if self.t0 <= t < self.t1:
                target = self.box_fn(t)
                if target is not None:
                    from backend.services.detectors import iou_matrix

                    ious = iou_matrix(det.xyxy, np.asarray([target], np.float32) * 0.5)[:, 0]
                    det = det.select(~((ious > 0.3) & (det.cls == PERSON)))
            out.append(det)
        return out


def test_split_track_is_stitched_back_into_one_identity(match) -> None:
    truth = match["truth"].tracking
    gid = 2
    hidden = _BlackoutDetector(_detector(match, dropout=0.0, false_positive_rate=0.0),
                               lambda t: truth.players[gid].box_at(t), 3.0, 5.0, 25.0)
    unstitched = track_video(match["proxy"], detector=hidden, stitch=False)
    stitched = track_video(match["proxy"], detector=hidden)
    # Without stitching the hidden player comes back under a new raw ID.
    assert len(unstitched.players) == len(stitched.players) + 1
    pid = {v: k for k, v in _dominant_truth(stitched, truth).items()}[gid]
    track = stitched.players[pid]
    assert len(track.source_track_ids) == 2
    assert track.start_s < 3.0 and track.end_s > 7.5
    gap = np.diff(track.t).max()
    assert 1.9 <= gap <= 2.2
    assert stitched.detector["tracking_stats"]["stitch_links"] >= 1
    # The other players are untouched.
    others = [p for k, p in stitched.players.items() if k != pid]
    assert all(len(p.source_track_ids) == 1 for p in others)


# ----------------------------------------------------------------------
# Cancel, persistence, legacy adapters
# ----------------------------------------------------------------------


class _CancellingDetector:
    def __init__(self, base, cancel: threading.Event, after_batches: int) -> None:
        self.base, self.cancel, self.after = base, cancel, after_batches
        self.batch = 4
        self.calls = 0

    def info(self) -> Dict[str, object]:
        return {"name": "cancelling"}

    def detect_batch(self, frames, *, frame_indices):
        self.calls += 1
        if self.calls >= self.after:
            self.cancel.set()
        return self.base.detect_batch(frames, frame_indices=frame_indices)


def test_cancel_returns_partial_result(match) -> None:
    cancel = threading.Event()
    det = _CancellingDetector(_detector(match), cancel, after_batches=10)
    result = track_video(match["proxy"], detector=det, batch_size=4, cancel_event=cancel)
    assert result.timings["cancelled"] is True
    assert 30 <= result.timings["frames"] < 200
    assert result.duration_s < 8.0
    assert result.players  # tracks from the processed part are kept
    assert max(p.end_s for p in result.players.values()) < result.duration_s + 0.05


def test_save_load_roundtrip(tmp_path: Path, tracked) -> None:
    result, _ = tracked
    result.focus_track_id = sorted(result.players)[0]
    result.save(tmp_path)
    loaded = TrackingResult.load(tmp_path)
    assert loaded.frame_size == result.frame_size
    assert loaded.focus_track_id == result.focus_track_id
    assert set(loaded.players) == set(result.players)
    for pid, track in result.players.items():
        other = loaded.players[pid]
        assert np.allclose(other.t, track.t) and np.allclose(other.x1, track.x1)
        assert other.source_track_ids == track.source_track_ids
    assert len(loaded.ball) == len(result.ball)
    assert np.allclose(loaded.ball.w, result.ball.w)
    assert loaded.timings["total_s"] == pytest.approx(result.timings["total_s"], abs=1e-3)
    result.focus_track_id = None


def test_legacy_views_match_old_track_video_shapes(tracked) -> None:
    result, _ = tracked
    tracks, ball, positions = legacy_views(result)
    assert list(tracks) == [result.longest_track_id()]
    points = next(iter(tracks.values()))
    assert isinstance(points[0], TrackPoint)
    p = points[0]
    assert p.xy == pytest.approx(((p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2))
    assert len(ball) == len(result.ball) and ball[0].bbox is not None
    assert positions.dtype == np.float32 and positions.shape[1] == 3
    assert np.all(np.diff(positions[:, 0]) >= 0)

    target = sorted(result.players)[1]
    tracks2, _ball2, fps, size, meta = legacy_track_video_tuple(result, target)
    assert list(tracks2) == [target]
    assert fps == pytest.approx(25.0) and size == (1280, 720)
    assert meta["target_track_id"] == target
    assert meta["stitched_track_ids"] == result.players[target].source_track_ids
    assert meta["player_positions"].shape[1] == 3

    import VideoHighlights

    legacy_fields = set(VideoHighlights.TrackPoint.__dataclass_fields__)
    assert legacy_fields == set(TrackPoint.__dataclass_fields__)


# ----------------------------------------------------------------------
# Tracker unit tests
# ----------------------------------------------------------------------


def test_bytetrack_lite_keeps_ids_through_short_dropout() -> None:
    trk = ByteTrackLite(frame_rate=25)
    ids_a, ids_b = set(), set()
    for f in range(40):
        a = [100 + 3 * f, 100, 120 + 3 * f, 150]
        b = [400 - 3 * f, 300, 420 - 3 * f, 350]
        boxes, confs = [], []
        if not (15 <= f < 20):  # 'a' undetected for 5 frames
            boxes.append(a)
            confs.append(0.9)
        boxes.append(b)
        confs.append(0.8)
        out = trk.update(np.asarray(boxes, np.float32), np.asarray(confs, np.float32))
        for tid, box, _score, _d in out:
            (ids_a if box[1] < 200 else ids_b).add(tid)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_bytetrack_lite_confirms_new_tracks_and_backfills_first_detection() -> None:
    trk = ByteTrackLite(frame_rate=25)
    trk.update(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    out1 = trk.update(np.array([[10, 10, 30, 60]], np.float32), np.array([0.9], np.float32))
    assert out1 == []  # unconfirmed after one hit
    out2 = trk.update(np.array([[11, 10, 31, 60]], np.float32), np.array([0.9], np.float32))
    assert len(out2) == 1
    assert [(f, tid) for f, tid, _b, _s in trk.backfill] == [(2, out2[0][0])]
    # A one-frame false positive never becomes a track.
    trk.update(np.array([[11, 10, 31, 60], [300, 300, 320, 350]], np.float32), np.array([0.9, 0.5], np.float32))
    out4 = trk.update(np.array([[12, 10, 32, 60]], np.float32), np.array([0.9], np.float32))
    assert [o[0] for o in out4] == [out2[0][0]]


def test_appearance_prevents_swap_when_players_cross() -> None:
    """Red and blue players cross with identical speed; colour keeps IDs."""
    trk = ByteTrackLite(frame_rate=25)
    owner: Dict[int, set] = {0: set(), 1: set()}
    for f in range(60):
        frame = np.full((200, 400, 3), (60, 140, 60), np.uint8)
        xa = 100 + 3 * f  # red moves right
        xb = 280 - 3 * f  # blue moves left
        boxes = [[xa, 80, xa + 20, 130], [xb, 82, xb + 20, 132]]
        frame[80:130, xa:xa + 20] = (40, 40, 220)
        frame[82:132, xb:xb + 20] = (220, 120, 30)
        out = trk.update(np.asarray(boxes, np.float32), np.array([0.9, 0.9], np.float32), frame)
        for tid, _box, _s, d in out:
            owner[d].add(tid)
    assert len(owner[0]) == 1 and len(owner[1]) == 1 and owner[0] != owner[1]


def test_torso_histogram_is_normalised() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :] = (40, 40, 220)
    hist = torso_histogram(frame, (10, 10, 40, 90))
    assert hist is not None and hist.sum() == pytest.approx(1.0)
    assert torso_histogram(frame, (10, 10, 11, 12)) is None


def test_botsort_adapter_runs_with_scipy_backed_lap(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ultralytics")
    from scipy.optimize import linear_sum_assignment

    def lapjv(cost, extend_cost=True, cost_limit=np.inf):
        cost = np.asarray(cost, dtype=np.float64)
        x = -np.ones(cost.shape[0], dtype=int)
        y = -np.ones(cost.shape[1], dtype=int)
        if cost.size:
            rows, cols = linear_sum_assignment(np.where(cost <= cost_limit, cost, 1e9))
            for r, c in zip(rows, cols):
                if cost[r, c] <= cost_limit:
                    x[r], y[c] = c, r
        return 0.0, x, y

    fake = types.ModuleType("lap")
    fake.__version__ = "0.5.12"
    fake.lapjv = lapjv
    if "lap" not in sys.modules:
        monkeypatch.setitem(sys.modules, "lap", fake)
    trk, name = make_tracker("botsort", None, 25.0)
    assert name == "botsort"
    ids = set()
    for f in range(30):
        out = trk.update(np.array([[100 + 2 * f, 50, 120 + 2 * f, 100]], np.float32), np.array([0.9], np.float32),
                         np.zeros((200, 400, 3), np.uint8))
        ids.update(o[0] for o in out)
    assert ids == {1}


def test_unknown_tracker_falls_back_to_bytetrack() -> None:
    trk, name = make_tracker("nonsense", None, 25.0)
    assert name == "bytetrack" and isinstance(trk, ByteTrackLite)


def test_row_buffer_growth_and_memory_budget() -> None:
    buf = RowBuffer(7, capacity=16)
    for k in range(100):
        buf.append_rows(np.full((37, 7), k, np.float32))
    arr = buf.array()
    assert arr.shape == (3700, 7)
    assert arr[0, 0] == 0 and arr[-1, 0] == 99
    assert buf.nbytes <= 2 * 3700 * 7 * 4 + 16 * 7 * 4
    # 90 min x 30 fps x 25 players: worst case (2x over-allocation) well under 2 GB.
    rows = 90 * 60 * 30 * 25
    assert 2 * rows * 7 * 4 < 0.5 * 1024 ** 3
