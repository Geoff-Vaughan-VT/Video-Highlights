"""Stats engine (backend/services/match_stats.py) on synthetic ground truth.

(tests/test_match_stats.py covers the API stat catalog, a different module.)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from backend.services.game_tracking import (
    GoalEvent,
    SetPieceEvent,
    analyze_game_states,
    build_ball_track,
    estimate_field_geometry,
)
from backend.services.match_stats import (
    PLAYER_STATS_FILENAME,
    TEAM_STATS_FILENAME,
    MatchStatsConfig,
    analyze_match,
    compute_match_stats,
    compute_player_stats,
    compute_team_stats_v2,
    load_stats,
    team_stats_for_catalog,
    write_stats,
)
from backend.services.pitch_calibration import calibrate_auto, calibrate_from_corners
from backend.services.synthetic_match import SyntheticMatchSpec, _simulate, generate_synthetic_match
from backend.services.team_classification import TeamConfig
from backend.services.tracking_types import (
    TEAM_A,
    TEAM_B,
    TEAM_REFEREE,
    BallDetections,
    PlayerTrack,
    TrackingResult,
)

pytest.importorskip("cv2")
pytest.importorskip("scipy")

SPEC = SyntheticMatchSpec(duration_s=36.0, goals=[(8.0, "right"), (24.0, "left")])
TEAMS = [TeamConfig(name="Reds", color_hex="#dc2828"), TeamConfig(name="Cyans", color_hex="#28c8dc")]


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    gt = generate_synthetic_match(tmp_path_factory.mktemp("stats") / "match.mp4", SPEC)
    tr = gt.tracking
    ball = build_ball_track(tr.ball.to_samples(), tr.frame_size)
    geo = estimate_field_geometry(tr.all_player_positions(), tr.frame_size)
    goals = [GoalEvent(t=t, side=side, confidence=0.95, reason="scripted") for t, side in gt.goal_times_s]
    segments = analyze_game_states(ball, geo, 0.0, tr.duration_s, goal_events=goals)
    x0, y0, x1, y1 = gt.pitch_bounds_px
    calib = calibrate_from_corners([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    players, teams = compute_match_stats(tr, ball, segments, goals, calib, TEAMS)
    return {"gt": gt, "ball": ball, "geo": geo, "goals": goals, "segments": segments, "calib": calib,
            "players": players, "teams": teams}


def _reference_distance(track, calib) -> float:
    m = calib.to_pitch(track.foot_xy().astype(np.float64))
    return float(np.hypot(*np.diff(m, axis=0).T).sum())


def test_player_distances_match_ground_truth(match) -> None:
    tr = match["gt"].tracking
    players = match["players"]["players"]
    assert len(players) == 10  # 2 x 5 outfield, referee excluded
    assert all(p["team"] in (TEAM_A, TEAM_B) for p in players)
    for p in players:
        ref = _reference_distance(tr.players[p["track_id"]], match["calib"])
        assert p["distance_m"] > 0
        assert abs(p["distance_m"] - ref) <= 0.10 * ref, (p["track_id"], p["distance_m"], ref)
        assert 0.0 < p["top_speed_mps"] <= 12.0
        assert 0.0 < p["avg_speed_mps"] < p["top_speed_mps"]
        assert p["minutes_tracked"] == pytest.approx(SPEC.duration_s / 60.0, abs=0.02)
        assert p["sprint_distance_m"] <= p["high_intensity_distance_m"] + 1e-6 <= p["distance_m"] + 1e-6


def test_auto_calibration_distances_are_plausible(match) -> None:
    tr = match["gt"].tracking
    calib = calibrate_auto(tr, match["ball"], match["segments"], field_geometry=match["geo"])
    assert calib.source == "auto"
    doc = compute_player_stats(tr, match["ball"], match["segments"], calib, goal_events=match["goals"])
    for p in doc["players"]:
        ref = _reference_distance(tr.players[p["track_id"]], calib)
        assert abs(p["distance_m"] - ref) <= 0.10 * ref
        assert 0.0 < p["top_speed_mps"] <= 12.0


def test_heatmaps_thirds_and_series(match) -> None:
    for p in match["players"]["players"]:
        hm = p["heatmap"]
        grid = np.asarray(hm["grid"])
        assert grid.shape == (hm["bins_y"], hm["bins_x"]) == (14, 21)
        assert grid.sum() == pytest.approx(1.0, abs=1e-3)
        assert sum(p["time_in_thirds_pct"].values()) == pytest.approx(100.0, abs=0.2)
        assert 0 < len(p["speed_series"]) <= SPEC.duration_s + 1
        ts = [s["t"] for s in p["speed_series"]]
        assert all(b - a >= 1.0 - 1e-6 for a, b in zip(ts, ts[1:]))
    # Team A defends left: its players never spend time in their attacking third
    # in this scripted match (homes at 18-50% of the pitch length).
    reds = [p for p in match["players"]["players"] if p["team"] == TEAM_A]
    assert all(p["time_in_thirds_pct"]["attacking"] < 5.0 for p in reds)


def test_goal_attribution_and_sides(match) -> None:
    teams = match["teams"]
    assert teams["teams"]["0"]["defends_first_half"] == "left"
    assert teams["teams"]["1"]["defends_first_half"] == "right"
    attribution = {g["side"]: g["team"] for g in teams["goal_attribution"]}
    # Team A (0) defends left, so a goal into the RIGHT goal is team A's.
    assert attribution == {"right": TEAM_A, "left": TEAM_B}
    assert teams["teams"]["0"]["goals"] == 1 and teams["teams"]["1"]["goals"] == 1
    for key in ("0", "1"):
        t = teams["teams"][key]
        assert t["shots"] >= t["shots_on_target"] >= t["goals"]
    assert teams["teams"]["0"]["name"] == "Reds"


def _simulated_possession(match, analysis) -> float:
    """Reference: team of the last player the simulated ball was redirected by."""
    pos, balls, _vis, _goals, meta = _simulate(SPEC)
    v = np.diff(balls, axis=0)
    touches = []
    for f in range(1, len(v)):
        a, b = v[f - 1], v[f]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if nb * SPEC.fps < 50:
            continue
        ang = np.degrees(np.arccos(np.clip(a @ b / max(na * nb, 1e-9), -1, 1))) if na > 1e-6 else 180.0
        if ang > 10 or (nb - na) * SPEC.fps > 200:
            d = np.hypot(*(pos[f] - balls[f]).T)
            d[[i for i, m in enumerate(meta) if m["team"] == TEAM_REFEREE]] = np.inf
            i = int(np.argmin(d))
            if d[i] <= 16:
                touches.append((f / SPEC.fps, meta[i]["team"]))
    tt = np.asarray([t for t, _ in touches])
    team = np.asarray([k for _, k in touches])
    in_play = analysis.possession_team != -3
    k = np.searchsorted(tt, analysis.possession_t, side="right") - 1
    ref = np.where(k >= 0, team[np.clip(k, 0, None)], -1)
    ok = in_play & (ref >= 0)
    return 100.0 * float(np.mean(ref[ok] == TEAM_A))


def test_possession_is_consistent_and_close_to_simulation(match) -> None:
    teams = match["teams"]["teams"]
    p0, p1 = teams["0"]["possession_pct"], teams["1"]["possession_pct"]
    assert 0.0 <= p0 <= 100.0 and 0.0 <= p1 <= 100.0
    assert p0 + p1 == pytest.approx(100.0, abs=0.1)
    q = match["teams"]["quality"]
    assert q["team_label_coverage_pct"] == pytest.approx(100.0)
    assert 50.0 <= q["ball_coverage_pct"] <= 100.0
    assert q["possession_known_pct"] > 50.0
    an = analyze_match(match["gt"].tracking, match["ball"], match["segments"], match["calib"],
                       goal_events=match["goals"])
    reference = _simulated_possession(match, an)
    # The synthetic plays ping-pong passes at 40-60 m/s inside crowds; the
    # touch chain recovers most but not all of them (measured: within ~5-8
    # points on 90 s matches).
    assert abs(p0 - reference) <= 12.0, (p0, reference)
    passes = teams["0"]["passes"] + teams["1"]["passes"]
    assert passes > 5
    assert all(0.0 <= teams[k]["pass_accuracy_pct"] <= 100.0 for k in teams)


def test_timeline_quality_and_contract_keys(match, tmp_path) -> None:
    teams = match["teams"]
    tl = teams["timeline"]
    assert tl["bin_s"] == 60
    n = len(tl["possession_pct_team0"])
    assert n == 1  # 36 s match
    assert len(tl["momentum"]) == len(tl["shots_team0"]) == len(tl["shots_team1"]) == n
    assert all(-1.0 <= m <= 1.0 for m in tl["momentum"])
    assert sum(tl["shots_team0"]) == teams["teams"]["0"]["shots"]

    p_path, t_path = write_stats(tmp_path, match["players"], teams)
    assert p_path.name == PLAYER_STATS_FILENAME and t_path.name == TEAM_STATS_FILENAME
    players_doc = json.loads(p_path.read_text())
    teams_doc = json.loads(t_path.read_text())
    assert {"generated_at", "pitch_calibration", "focus_player_track_id", "players"} <= set(players_doc)
    assert {"source", "confidence", "pitch_length_m", "pitch_width_m", "image_corners_px",
            "homography"} <= set(players_doc["pitch_calibration"])
    player_keys = {"track_id", "team", "team_name", "label", "jersey_number", "minutes_tracked", "distance_m",
                   "top_speed_mps", "avg_speed_mps", "sprints", "sprint_distance_m", "high_intensity_distance_m",
                   "touches", "passes_attempted", "passes_completed", "shots", "time_in_thirds_pct", "heatmap",
                   "speed_series"}
    for p in players_doc["players"]:
        assert player_keys <= set(p)
        assert set(p["time_in_thirds_pct"]) == {"defensive", "middle", "attacking"}
        assert {"bins_x", "bins_y", "grid"} <= set(p["heatmap"])
    assert {"generated_at", "teams", "timeline", "goal_attribution", "quality"} <= set(teams_doc)
    team_keys = {"name", "color_hex", "defends_first_half", "possession_pct", "passes", "pass_accuracy_pct",
                 "shots", "shots_on_target", "goals", "corners", "territory_pct", "avg_speed_mps", "distance_m",
                 "play_speed"}
    assert set(teams_doc["teams"]) == {"0", "1"}
    for t in teams_doc["teams"].values():
        assert team_keys <= set(t)
        assert set(t["play_speed"]) == {"ball_speed_mean_mps", "progression_mps", "passes_per_minute"}
        assert sum(t["territory_pct"].values()) == pytest.approx(100.0, abs=0.2)
    assert {"bin_s", "possession_pct_team0", "momentum", "shots_team0", "shots_team1"} <= set(teams_doc["timeline"])
    assert {"t", "side", "team"} <= set(teams_doc["goal_attribution"][0])
    assert {"team_label_coverage_pct", "ball_coverage_pct"} <= set(teams_doc["quality"])
    loaded_players, loaded_teams = load_stats(tmp_path)
    assert loaded_players == players_doc and loaded_teams == teams_doc


def test_team_stats_for_catalog(match) -> None:
    cat = team_stats_for_catalog(match["teams"])
    assert set(cat) == {"home", "away"}
    home = match["teams"]["teams"]["0"]
    assert cat["home"]["goals"] == home["goals"]
    assert cat["home"]["possession"] == home["possession_pct"]
    assert cat["home"]["total_shots"] == home["shots"]
    assert cat["away"]["total_passes"] == match["teams"]["teams"]["1"]["passes"]
    assert set(cat["home"]) == {"goals", "possession", "total_shots", "shots_on_target", "total_passes",
                                "pass_accuracy", "corners"}
    assert team_stats_for_catalog({}) == {"home": {}, "away": {}}


# ---------------------------------------------------------------------------
# Hand-built tracking (no video): halftime swap, corners, unknown teams, speed
# ---------------------------------------------------------------------------

W, H = 1920, 1080
CORNERS = [[100.0, 200.0], [1820.0, 200.0], [1820.0, 900.0], [100.0, 900.0]]  # top-down, 16.38 px/m


def _track(track_id: int, t: np.ndarray, x: np.ndarray, y: np.ndarray, team: int) -> PlayerTrack:
    return PlayerTrack(track_id, t.astype(np.float32), (x - 10).astype(np.float32), (y - 50).astype(np.float32),
                       (x + 10).astype(np.float32), y.astype(np.float32), np.ones(t.size, np.float32), team=team)


def test_halftime_swap_drives_sides_goals_and_corners() -> None:
    fps, dur = 10.0, 200.0
    t = np.arange(int(dur * fps)) / fps
    half = t >= 100.0
    players = {}
    for i in range(4):
        # Team 0 on the left in the first half, on the right after the break.
        x0 = np.where(half, 1400.0, 500.0) + 30 * i
        x1 = np.where(half, 500.0, 1400.0) + 30 * i
        players[i] = _track(i, t, x0, np.full(t.size, 400.0 + 80 * i), TEAM_A)
        players[10 + i] = _track(10 + i, t, x1, np.full(t.size, 420.0 + 80 * i), TEAM_B)
    players[99] = _track(99, t, np.full(t.size, 960.0), np.full(t.size, 500.0), -1)
    tracking = TrackingResult(fps=fps, frame_size=(W, H), duration_s=dur, players=players,
                              ball=BallDetections.empty())
    calib = calibrate_from_corners(CORNERS)
    goals = [GoalEvent(t=50.0, side="left", confidence=0.9, reason=""),
             GoalEvent(t=150.0, side="left", confidence=0.9, reason="")]
    corners = [SetPieceEvent(kind="corner_kick", t_start=60.0, t_kick=62.0, x=1815.0, y=205.0, side=None),
               SetPieceEvent(kind="corner_kick", t_start=160.0, t_kick=162.0, x=1815.0, y=205.0, side="right")]
    cfg = MatchStatsConfig(min_track_s=1.0)
    teams = compute_team_stats_v2(tracking, None, None, goals, calib, None, set_piece_events=corners, config=cfg)
    assert teams["half_time_s"] == pytest.approx(100.0)
    assert teams["teams"]["0"]["defends_first_half"] == "left"
    assert teams["teams"]["0"]["defends_second_half"] == "right"
    assert [g["team"] for g in teams["goal_attribution"]] == [TEAM_B, TEAM_A]
    assert teams["teams"]["0"]["goals"] == 1 and teams["teams"]["1"]["goals"] == 1
    # Corner at the right in the first half: team 0 attacks right -> team 0's corner;
    # second half: team 1 attacks right.
    assert teams["teams"]["0"]["corners"] == 1 and teams["teams"]["1"]["corners"] == 1
    # No ball -> no possession information, reported honestly.
    assert teams["teams"]["0"]["possession_pct"] is None
    assert teams["quality"]["ball_coverage_pct"] == 0.0

    players_doc = compute_player_stats(tracking, None, None, calib, config=cfg)
    unknown = [p for p in players_doc["players"] if p["track_id"] == 99]
    assert unknown and unknown[0]["team_unknown"] is True and unknown[0]["team_name"] is None
    # Thirds are relative to the attacking direction: team 0 sits at x=500px
    # (left) then x=1400px (right) while attacking right then left: the same
    # (defensive-side) third in both halves.
    p0 = next(p for p in players_doc["players"] if p["track_id"] == 0)
    assert p0["time_in_thirds_pct"]["defensive"] > 95.0
    # Stationary-ish players: tiny distance despite the teleport at halftime
    # (gap-free track but a > 12 m/s jump is clipped).
    assert p0["distance_m"] < 2.0


def test_speed_sprint_and_gap_handling() -> None:
    fps = 25.0
    calib = calibrate_from_corners(CORNERS)
    px_per_m = (CORNERS[1][0] - CORNERS[0][0]) / 105.0
    # 10 s jog at 3 m/s, 3 s sprint at 8 m/s, 5 s missing, 10 s at 3 m/s.
    seg_speed = [(10.0, 3.0), (3.0, 8.0)]
    t_list, x_list = [], []
    t_cur, x_cur = 0.0, 300.0
    for dur, v in seg_speed:
        n = int(dur * fps)
        t_list.append(t_cur + np.arange(n) / fps)
        x_list.append(x_cur + v * px_per_m * np.arange(n) / fps)
        t_cur += n / fps
        x_cur = x_list[-1][-1] + v * px_per_m / fps
    t_cur += 5.0
    x_cur += 40 * px_per_m  # jumped 40 m while unobserved: must NOT count
    n = int(10 * fps)
    t_list.append(t_cur + np.arange(n) / fps)
    x_list.append(x_cur + 3.0 * px_per_m * np.arange(n) / fps)
    t = np.concatenate(t_list)
    x = np.concatenate(x_list)
    tracking = TrackingResult(fps=fps, frame_size=(W, H), duration_s=float(t[-1]) + 0.04,
                              players={1: _track(1, t, x, np.full(t.size, 550.0), TEAM_A)},
                              ball=BallDetections.empty())
    doc = compute_player_stats(tracking, None, None, calib)
    p = doc["players"][0]
    expected = 10 * 3.0 + 3 * 8.0 + 10 * 3.0
    assert p["distance_m"] == pytest.approx(expected, rel=0.05)
    assert p["top_speed_mps"] == pytest.approx(8.0, abs=0.4)
    assert p["sprints"] == 1
    assert p["sprint_distance_m"] == pytest.approx(24.0, rel=0.2)
    assert p["high_intensity_distance_m"] == pytest.approx(24.0, rel=0.2)
    assert p["minutes_tracked"] == pytest.approx(23.0 / 60.0, abs=0.01)


def test_ninety_minute_match_is_fast() -> None:
    rng = np.random.default_rng(0)
    fps, dur = 30.0, 5400.0
    n = int(dur * fps)
    t = np.arange(n) / fps
    players = {}
    for i in range(25):
        x = np.clip(200 + rng.random() * 1500 + np.cumsum(rng.normal(0, 1.5, n)), 60, 1860)
        y = np.clip(300 + rng.random() * 500 + np.cumsum(rng.normal(0, 1.0, n)), 250, 1000)
        team = TEAM_REFEREE if i == 24 else i % 2
        players[i] = _track(i, t, x, y, team)
    bx = np.clip(960 + np.cumsum(rng.normal(0, 4, n)), 60, 1860)
    by = np.clip(600 + np.cumsum(rng.normal(0, 3, n)), 250, 1000)
    ones = np.ones(n, np.float32)
    ball_det = BallDetections(t.astype(np.float32), bx.astype(np.float32), by.astype(np.float32), ones * 8,
                              ones * 8, ones)
    tracking = TrackingResult(fps=fps, frame_size=(W, H), duration_s=dur, players=players, ball=ball_det)
    # A hand-built BallTrack-like object keeps the test about match_stats speed.
    from backend.services.game_tracking import BallTrack, BallTrackConfig

    ball = BallTrack(times=t, xs=bx, ys=by, frame_size=(W, H), config=BallTrackConfig())
    calib = calibrate_from_corners(CORNERS)
    start = time.perf_counter()
    players_doc, teams_doc = compute_match_stats(tracking, ball, None, [], calib, None)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, elapsed
    assert len(players_doc["players"]) == 24
    assert teams_doc["quality"]["ball_coverage_pct"] > 99.0
