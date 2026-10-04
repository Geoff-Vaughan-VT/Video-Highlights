from __future__ import annotations

import numpy as np
import pytest

from backend.services.game_tracking import build_ball_track, estimate_field_geometry
from backend.services.team_classification import (
    TeamConfig,
    classify_player_teams,
    compute_team_stats,
    detect_team_colors,
    hex_to_bgr,
)

cv2 = pytest.importorskip("cv2")

W, H = 640, 360
RED = TeamConfig(name="Lions", color_hex="#d32f2f")
BLUE = TeamConfig(name="Hawks", color_hex="#1976d2")


def test_hex_to_bgr() -> None:
    assert hex_to_bgr("#ff0000") == (0, 0, 255)
    assert hex_to_bgr("00ff00") == (0, 255, 0)


def _write_two_team_video(path, frames=60):
    """Red shirts on the left half, blue shirts on the right half."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert writer.isOpened()
    positions = []
    red_spots = [(120, 150), (180, 240), (240, 120)]
    blue_spots = [(420, 160), (480, 250), (540, 130)]
    for i in range(frames):
        t = i / 10.0
        frame = np.full((H, W, 3), (40, 90, 40), dtype=np.uint8)
        for x, y in red_spots:
            cv2.rectangle(frame, (x - 9, y - 12), (x + 9, y + 12), hex_to_bgr(RED.color_hex), -1)
            positions.append((t, float(x), float(y)))
        for x, y in blue_spots:
            cv2.rectangle(frame, (x - 9, y - 12), (x + 9, y + 12), hex_to_bgr(BLUE.color_hex), -1)
            positions.append((t, float(x), float(y)))
        writer.write(frame)
    writer.release()
    return np.asarray(positions, dtype=np.float64)


def test_classify_player_teams_by_jersey_color(tmp_path) -> None:
    video = tmp_path / "teams.mp4"
    positions = _write_two_team_video(video)

    labeled = classify_player_teams(str(video), positions, RED, BLUE)

    assert len(labeled) > 20
    known = labeled[labeled[:, 3] >= 0]
    assert len(known) >= 0.7 * len(labeled), "too many unknowns"
    # Left-half positions must be team 0 (red), right-half team 1 (blue).
    left = known[known[:, 1] < W / 2]
    right = known[known[:, 1] >= W / 2]
    assert len(left) and np.mean(left[:, 3] == 0) > 0.9
    assert len(right) and np.mean(right[:, 3] == 1) > 0.9


def test_compute_team_stats_possession_sides_and_goal_attribution() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    cloud = np.stack(
        [rng.uniform(0, 60, n), rng.uniform(100, 1820, n), rng.uniform(200, 880, n)],
        axis=1,
    )
    geometry = estimate_field_geometry(cloud, (1920, 1080))

    # Labeled positions across 60s: Lions (0) on the left, Hawks (1) right;
    # the ball lives on the LEFT side, next to Lions players -> Lions possession.
    rows = []
    for t in np.arange(0.0, 60.0, 0.5):
        rows.append((t, 500.0, 500.0, 0))
        rows.append((t, 520.0, 560.0, 0))
        rows.append((t, 1400.0, 500.0, 1))
    labeled = np.asarray(rows, dtype=np.float32)
    ball = build_ball_track([(t, 505.0, 510.0) for t in np.arange(0.0, 60.0, 0.5)], (1920, 1080))

    goal_events = [{"t": 30.0, "side": "left", "confidence": 0.9}]
    stats = compute_team_stats(labeled, ball, geometry, goal_events, RED, BLUE, 60.0)

    assert stats["defending_side"]["Lions"] == "left"
    assert stats["defending_side"]["Hawks"] == "right"
    assert stats["possession_pct"]["Lions"] > 80.0
    # Goal INTO the left goal (defended by Lions) is scored BY Hawks.
    assert stats["goals"]["Hawks"] == 1
    assert stats["goals"]["Lions"] == 0
    assert stats["goal_attribution"][0]["team"] == "Hawks"


def test_compute_team_stats_empty_labels_is_graceful() -> None:
    rng = np.random.default_rng(3)
    cloud = np.stack([rng.uniform(0, 60, 200), rng.uniform(100, 1820, 200), rng.uniform(200, 880, 200)], axis=1)
    geometry = estimate_field_geometry(cloud, (1920, 1080))
    stats = compute_team_stats(np.empty((0, 4)), None, geometry, [], RED, BLUE, 60.0)
    assert "note" in stats


def _bgr_dist(hex_a: str, hex_b: str) -> float:
    a, b = np.asarray(hex_to_bgr(hex_a), float), np.asarray(hex_to_bgr(hex_b), float)
    return float(np.linalg.norm(a - b))


def test_detect_team_colors_finds_both_kits(tmp_path) -> None:
    video = tmp_path / "kits.mp4"
    positions = _write_two_team_video(video)

    detected = detect_team_colors(str(video), positions)

    assert detected is not None
    hex_a, hex_b = detected
    # Detected pair must be two genuinely different colors...
    assert _bgr_dist(hex_a, hex_b) > 60.0
    # ...and each configured kit must be close to one of them (order-free).
    for kit in (RED.color_hex, BLUE.color_hex):
        assert min(_bgr_dist(kit, hex_a), _bgr_dist(kit, hex_b)) < 90.0, (
            f"kit {kit} not matched by detected {hex_a}/{hex_b}"
        )


def test_detect_team_colors_needs_enough_signal(tmp_path) -> None:
    video = tmp_path / "kits2.mp4"
    _write_two_team_video(video)
    # Too few tracked positions -> refuse to guess.
    assert detect_team_colors(str(video), None) is None
    few = np.asarray([(0.5, 120.0, 150.0)] * 10, dtype=np.float64)
    assert detect_team_colors(str(video), few) is None


# ---------------------------------------------------------------------------
# Legacy cap regression + v2 per-track labelling
# ---------------------------------------------------------------------------

import copy  # noqa: E402

from backend.services.synthetic_match import SyntheticMatchSpec, generate_synthetic_match  # noqa: E402
from backend.services.team_classification import (  # noqa: E402
    TeamClassifierConfig,
    assign_track_teams,
    assign_track_teams_with_report,
)
from backend.services.tracking_types import TEAM_REFEREE, TEAM_UNKNOWN, PlayerTrack  # noqa: E402


def test_classify_player_teams_covers_whole_window_despite_row_cap(tmp_path) -> None:
    """The old 4000-row cap stopped labelling after the first minutes."""
    video = tmp_path / "long.mp4"
    positions = _write_two_team_video(video, frames=120)  # 12 s
    cfg = TeamClassifierConfig(max_samples=10, sample_fps=1.0)
    labeled = classify_player_teams(str(video), positions, RED, BLUE, config=cfg)
    assert len(labeled) > 10
    assert float(labeled[:, 0].max()) > 10.0, "labelling stopped early"
    known = labeled[labeled[:, 3] >= 0]
    assert np.mean(known[known[:, 1] < W / 2, 3] == 0) > 0.9


@pytest.fixture(scope="module")
def synthetic_kits(tmp_path_factory):
    path = tmp_path_factory.mktemp("kits") / "match.mp4"
    gt = generate_synthetic_match(path, SyntheticMatchSpec(duration_s=12.0))
    return path, gt


def _unlabelled(tracking):
    tr = copy.deepcopy(tracking)
    for p in tr.players.values():
        p.team = TEAM_UNKNOWN
        p.team_confidence = 0.0
    return tr


def _accuracy(tr, truth) -> float:
    return float(np.mean([tr.players[k].team == truth[k] for k in truth]))


# Synthetic kits: team A red (BGR 40,40,220), team B cyan (BGR 220,200,40), referee near-black.
RED_KIT = TeamConfig(name="Reds", color_hex="#dc2828")
CYAN_KIT = TeamConfig(name="Cyans", color_hex="#28c8dc")


def test_assign_track_teams_without_colours(synthetic_kits) -> None:
    path, gt = synthetic_kits
    truth = {k: p.team for k, p in gt.tracking.players.items()}
    tr, report = assign_track_teams_with_report(str(path), _unlabelled(gt.tracking), sample_hz=1.0)
    # Clusters are named by population: both teams have 5 players, so check
    # the partition (allowing a label swap) and the referee.
    acc = max(_accuracy(tr, truth), _accuracy(_swap_teams(tr), truth))
    assert acc >= 0.9
    referee_ids = [k for k, v in truth.items() if v == TEAM_REFEREE]
    assert all(tr.players[k].team == TEAM_REFEREE for k in referee_ids)
    assert report.colors_source == "detected"
    assert report.track_coverage_pct == pytest.approx(100.0)
    assert set(report.team_colors_hex) >= {"0", "1"}
    detected = {report.team_colors_hex["0"], report.team_colors_hex["1"]}
    for kit in (RED_KIT.color_hex, CYAN_KIT.color_hex):
        assert min(_bgr_dist(kit, h) for h in detected) < 60.0
    assert tr.detector["team_assignment"]["tracks_referee"] == len(referee_ids)
    assert all(p.jersey_color_hex for p in tr.players.values())


def _swap_teams(tracking):
    tr = copy.deepcopy(tracking)
    for p in tr.players.values():
        if p.team in (0, 1):
            p.team = 1 - p.team
    return tr


def test_assign_track_teams_with_supplied_colours(synthetic_kits) -> None:
    path, gt = synthetic_kits
    truth = {k: p.team for k, p in gt.tracking.players.items()}
    # Slightly-off picker colours (as a user would choose them).
    tr = assign_track_teams(str(path), _unlabelled(gt.tracking), TeamConfig("A", "#d32f2f"),
                            TeamConfig("B", "#00bcd4"), sample_hz=1.0)
    assert _accuracy(tr, truth) >= 0.9
    assert all(0.0 < p.team_confidence <= 1.0 for p in tr.players.values())
    # Supplying the kits the other way round swaps the labels.
    swapped = assign_track_teams(str(path), _unlabelled(gt.tracking), CYAN_KIT, RED_KIT, sample_hz=1.0)
    assert _accuracy(_swap_teams(swapped), truth) >= 0.9
    assert all(swapped.players[k].team == TEAM_REFEREE for k, v in truth.items() if v == TEAM_REFEREE)


def test_assign_track_teams_labels_short_fragments_via_extra_frames(synthetic_kits) -> None:
    """Fragmented tracks (3 s pieces) with a sparse uniform sample still get labelled."""
    path, gt = synthetic_kits
    fragmented = _unlabelled(gt.tracking)
    pieces = {}
    truth = {}
    next_id = 100
    for p in fragmented.players.values():
        for start in np.arange(0.0, 12.0, 3.0):
            m = (p.t >= start) & (p.t < start + 2.8)  # no two pieces at one instant
            if not m.any():
                continue
            pieces[next_id] = PlayerTrack(next_id, p.t[m], p.x1[m], p.y1[m], p.x2[m], p.y2[m], p.conf[m])
            truth[next_id] = gt.tracking.players[p.track_id].team
            next_id += 1
    fragmented.players = pieces
    tr, report = assign_track_teams_with_report(str(path), fragmented, RED_KIT, CYAN_KIT, sample_hz=0.1)
    assert report.frames_planned > 2  # the uniform sample alone is 2 frames
    assert report.track_coverage_pct >= 90.0
    assert _accuracy(tr, truth) >= 0.9


def test_assign_track_teams_unreadable_video_is_graceful(tmp_path, synthetic_kits) -> None:
    _, gt = synthetic_kits
    tr, report = assign_track_teams_with_report(str(tmp_path / "missing.mp4"), _unlabelled(gt.tracking))
    assert report.frames_read == 0
    assert all(p.team == TEAM_UNKNOWN for p in tr.players.values())
