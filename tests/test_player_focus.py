from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services.player_focus import choose_target_track_id, resolve_player_roi_box, stitch_target_track


@dataclass
class _Point:
    t: float
    xy: tuple[float, float]
    bbox: tuple[float, float, float, float]


def test_resolve_player_roi_box_from_normalized_coords() -> None:
    roi = {
        "normalized": True,
        "x1_norm": 0.25,
        "y1_norm": 0.10,
        "x2_norm": 0.50,
        "y2_norm": 0.60,
    }

    result = resolve_player_roi_box(roi, frame_width=200, frame_height=100)
    assert result == (50.0, 10.0, 100.0, 60.0)


def test_resolve_player_roi_box_from_xywh_normalized() -> None:
    roi = {
        "normalized": True,
        "x": 0.20,
        "y": 0.30,
        "w": 0.15,
        "h": 0.25,
    }

    result = resolve_player_roi_box(roi, frame_width=400, frame_height=200)
    assert result == pytest.approx((80.0, 60.0, 140.0, 110.0))


def test_resolve_player_roi_box_clamps_and_rejects_tiny_boxes() -> None:
    assert resolve_player_roi_box({"normalized": True, "x1_norm": 0.2, "y1_norm": 0.2, "x2_norm": 0.201, "y2_norm": 0.21}, 100, 100) is None
    clamped = resolve_player_roi_box({"x": -10, "y": -5, "w": 40, "h": 50}, 100, 100)
    assert clamped == (0.0, 0.0, 30.0, 45.0)


def test_choose_target_track_id_prefers_track_with_strong_roi_overlap() -> None:
    user_box = (40.0, 10.0, 70.0, 90.0)
    tracks = {
        3: [
            _Point(t=0.0, xy=(88.0, 50.0), bbox=(76.0, 18.0, 100.0, 88.0)),
            _Point(t=0.5, xy=(90.0, 50.0), bbox=(78.0, 18.0, 102.0, 88.0)),
        ],
        7: [
            _Point(t=0.0, xy=(55.0, 50.0), bbox=(43.0, 16.0, 68.0, 88.0)),
            _Point(t=0.5, xy=(57.0, 50.0), bbox=(45.0, 16.0, 70.0, 88.0)),
        ],
    }

    assert choose_target_track_id(tracks, user_box, window_t=1.0) == 7


def test_stitch_target_track_merges_adjacent_track_fragments() -> None:
    tracks = {
        11: [
            _Point(t=0.0, xy=(100.0, 80.0), bbox=(88.0, 42.0, 112.0, 118.0)),
            _Point(t=0.5, xy=(102.0, 80.0), bbox=(90.0, 42.0, 114.0, 118.0)),
            _Point(t=1.0, xy=(104.0, 80.0), bbox=(92.0, 42.0, 116.0, 118.0)),
        ],
        21: [
            _Point(t=1.2, xy=(106.0, 80.0), bbox=(94.0, 42.0, 118.0, 118.0)),
            _Point(t=1.7, xy=(109.0, 80.0), bbox=(97.0, 42.0, 121.0, 118.0)),
        ],
        99: [
            _Point(t=1.1, xy=(240.0, 90.0), bbox=(225.0, 45.0, 255.0, 125.0)),
            _Point(t=1.6, xy=(244.0, 90.0), bbox=(229.0, 45.0, 259.0, 125.0)),
        ],
    }

    stitched_ids, stitched = stitch_target_track(tracks, 11, max_gap_seconds=0.5)

    assert stitched_ids == [11, 21]
    assert [round(point.t, 2) for point in stitched] == [0.0, 0.5, 1.0, 1.2, 1.7]


# ----------------------------------------------------------------------
# Whole-match stitching (stitch_tracks) and ROI-at-time selection
# ----------------------------------------------------------------------

import time  # noqa: E402

import numpy as np  # noqa: E402

from backend.services.player_focus import (  # noqa: E402
    TrackFragment,
    appearance_similarity,
    select_track_at,
    stitch_tracks,
)
from backend.services.tracking_types import player_track_from_rows  # noqa: E402

_RED = np.array([0.85, 0.15, 0.0, 0.0])
_BLUE = np.array([0.0, 0.0, 0.15, 0.85])


def _frag(tid, t0, t1, x0, x1, *, y=100.0, h=40.0, app=None, team=-1, vx=None):
    w = h * 0.4
    vel = (vx if vx is not None else (x1 - x0) / max(1e-6, t1 - t0), 0.0)
    return TrackFragment(
        track_id=tid, t_start=t0, t_end=t1,
        start_box=(x0 - w / 2, y - h, x0 + w / 2, y), end_box=(x1 - w / 2, y - h, x1 + w / 2, y),
        start_vel=vel, end_vel=vel, appearance=app, samples=max(3, int((t1 - t0) * 25)), team=team,
    )


def test_appearance_similarity_bhattacharyya() -> None:
    assert appearance_similarity(_RED, _RED) == pytest.approx(1.0)
    assert appearance_similarity(_RED, _BLUE) < 0.5
    assert appearance_similarity(None, _RED) is None


def test_stitch_tracks_links_motion_continuous_fragments() -> None:
    frags = [
        _frag(1, 0.0, 2.0, 100, 200),  # moving right 50 px/s
        _frag(2, 2.6, 5.0, 232, 350),  # continues after a 0.6 s gap
        _frag(3, 0.0, 5.0, 600, 650),  # unrelated player far away
    ]
    chains = stitch_tracks(frags)
    assert sorted(chains) == [[1, 2], [3]]


def test_stitch_tracks_appearance_vetoes_wrong_kit() -> None:
    frags = [
        _frag(1, 0.0, 2.0, 100, 200, app=_RED),
        _frag(2, 2.3, 4.0, 215, 300, app=_BLUE),  # perfect motion, wrong colour
        _frag(3, 2.5, 4.0, 225, 300, app=_RED),  # slightly worse motion, right colour
    ]
    details = stitch_tracks(frags, return_details=True)
    assert [1, 3] in details.chains
    assert [2] in details.chains
    assert details.links[0].appearance_sim == pytest.approx(1.0)


def test_stitch_tracks_respects_team_time_overlap_and_gap() -> None:
    frags = [
        _frag(1, 0.0, 2.0, 100, 200, team=0),
        _frag(2, 2.2, 4.0, 205, 300, team=1),  # other team
        _frag(3, 1.0, 4.0, 200, 300),  # overlaps track 1 by 1 s: same player impossible
        _frag(4, 8.0, 9.0, 300, 320),  # >= 4 s after every other fragment ends
    ]
    chains = stitch_tracks(frags, max_gap_s=3.0)
    assert all(len(c) == 1 for c in chains)
    assert sorted(c[0] for c in chains) == [1, 2, 3, 4]


def test_stitch_tracks_one_successor_per_fragment_and_chains_in_time_order() -> None:
    frags = [
        _frag(10, 0.0, 1.0, 100, 150),
        _frag(11, 1.3, 2.0, 165, 200),
        _frag(12, 2.4, 3.0, 220, 250),
        _frag(13, 1.3, 2.0, 168, 200),  # competing candidate for 10's successor
    ]
    chains = stitch_tracks(frags)
    flat = sorted(t for c in chains for t in c)
    assert flat == [10, 11, 12, 13]  # every fragment exactly once
    longest = max(chains, key=len)
    assert longest[0] == 10 and len(longest) == 3
    starts = {f.track_id: f.t_start for f in frags}
    assert [starts[t] for t in longest] == sorted(starts[t] for t in longest)


def test_stitch_tracks_scales_to_many_fragments() -> None:
    rng = np.random.default_rng(0)
    frags = []
    tid = 0
    for lane in range(25):  # 25 players, each split into 200 fragments
        t = 0.0
        x = 50.0 + lane * 70
        for _ in range(200):
            dur = float(rng.uniform(1.0, 5.0))
            frags.append(_frag(tid, t, t + dur, x, x, y=100.0 + lane * 60))
            tid += 1
            t += dur + float(rng.uniform(0.1, 1.0))
    started = time.perf_counter()
    chains = stitch_tracks(frags)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
    assert len(chains) <= 40  # 5000 fragments collapse to ~25 identities


def test_select_track_at_uses_time_of_roi() -> None:
    a = player_track_from_rows(1, [(t / 10, 100 + t, 50, 120 + t, 100) for t in range(0, 50)])
    b = player_track_from_rows(2, [(t / 10, 300 - 4 * t, 50, 320 - 4 * t, 100) for t in range(0, 50)])
    roi_late = (195.0, 45.0, 230.0, 105.0)  # b is here at t=2.5 s, a is far away
    assert select_track_at({1: a, 2: b}, roi_late, t=2.5) == 2
    assert select_track_at({1: a, 2: b}, (100.0, 45.0, 125.0, 105.0), t=0.0) == 1
    assert select_track_at({1: a, 2: b}, (900.0, 45.0, 925.0, 105.0), t=0.0) is None
    assert select_track_at({1: a, 2: b}, roi_late, t=20.0) is None  # nothing near that time
