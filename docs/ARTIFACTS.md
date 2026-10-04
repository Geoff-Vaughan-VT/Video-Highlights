# Run Artifacts Contract

Every processing run writes its outputs into one run directory
(`VH_OUTPUT_ROOT/<job_id>/`). The files below are the contract between the
engine, the Studio API/UI, the stats and event engines, and the renderers.
All producers and consumers in this repo build against this document.

Timebase: unless a key says `_source`, times are seconds from the start of
the **processing window** (trimmed video). `trim_offset_seconds` converts to
the original file: `t_source = t + trim_offset_seconds`.

Coordinates: **source-video pixels** (original frame size), even when
detection ran on a proxy.

## Media

| File | Producer | Notes |
|---|---|---|
| `proxy_1080p.mp4` | frame_source | Single ffmpeg pass over the source (hwaccel when available). H.264 yuv420p, `+faststart`, audio AAC. Used for detection, team colours, cards, audio analysis **and browser playback**. Height configurable (`proxy_height`, default 1080; `fast` profile uses 720). |
| `audio_analysis.wav` | frame_source | Mono 16 kHz PCM extracted in the same pass; all audio analysis reads this, never the 4K source. |
| `full_follow_ball_zoom.mp4` | camera_render | Final game-camera movie. **Output height = `output_height` (default 1080)**, never the source size. Encoded straight from the source by ffmpeg (crop+scale filter driven by `camera_crops.txt`), audio muxed in the same command. |
| `highlight_NN.mp4` | event clips | Cut from the final movie (follow modes) or source (wide) with stream copy where possible. |
| `highlights_reel.mp4` | broadcast | ffmpeg concat/xfade. No moviepy. |
| `debug_camera_wide.mp4` | camera_render | Optional (off by default). 1280 wide, rendered from the proxy. |
| `thumbs/NNNN.jpg` | frame_source | Optional thumbnails every 10 s from the proxy for the UI scrub bar. |

## Tracking (`tracks.npz` + `tracks_meta.json`)

Written/read by `backend/services/tracking_types.py::TrackingResult.save/load`.
`tracks.npz` holds `players` `[track_id, t, x1, y1, x2, y2, conf]` float32 and
`ball` `[t, x, y, w, h, conf]` float32 (raw detections, unfiltered).
`tracks_meta.json` holds fps, frame size, duration, trim offset, proxy scale,
focus_track_id, detector info, timings, and per-track metadata (team,
team_confidence, jersey_color_hex, jersey_number, label, source_track_ids).

`analysis_tracking.json` (legacy manifest) stays as the human-readable
summary and keeps `tracking.target_track` / `tracking.ball_track` for the
on-demand clip API.

## Camera (`camera_decisions.jsonl`, `camera_crops.txt`)

`camera_decisions.jsonl`: one row per output frame
`{index, t, t_source, center_x, center_y, zoom, state, focus, reason, confidence, ball_x, ball_y, ball_source, target_x, target_y}`.

`camera_crops.txt`: ffmpeg `sendcmd` script, one command per frame
`<t> crop w <w>, crop h <h>, crop x <x>, crop y <y>;` in source pixels (even
integers). Produced by the planner, consumed by the renderer.

`camera_quality.json`: smoothness metrics for the plan
`{pan_speed_p95_cropw_per_s, pan_accel_p95, zoom_rate_p95_per_s, zoom_reversals_per_min, hard_snaps, ball_in_frame_fraction}`.

## Game analysis (`analysis_game_states.json`)

Unchanged shape: `field_geometry`, `ball_track_stats`, `state_summary_s`,
`segments[]`, `goal_events[]`, `set_piece_events[]`, `card_events[]`.

## Events (`analysis_events.json`)

```json
{
  "generated_at": "...", "trim_offset_seconds": 0.0,
  "events": [
    {
      "id": "ev_0001", "type": "goal|shot|save|chance|corner_kick|free_kick|penalty_kick|goal_kick|kickoff|yellow_card|red_card|sprint|dribble|turnover|foul_candidate",
      "t": 812.4, "t_start": 798.0, "t_end": 821.5,
      "team": 0, "team_name": "HOME", "side": "left|right|null",
      "player_track_id": 7, "secondary_track_id": null,
      "confidence": 0.86, "excitement": 0.91,
      "reason": "human readable", "evidence": {"...": "..."},
      "sources": ["ball_tracking", "audio", "vision", "motion"]
    }
  ],
  "reel_plan": {"target_duration_s": 300, "selected_event_ids": ["ev_0001"], "total_duration_s": 287.5},
  "summary": {"counts_by_type": {"goal": 2}, "per_team": {"0": {"shots": 9}, "1": {"shots": 4}}}
}
```

`analysis_bookmarks.json/.csv` remain for backwards compatibility and are
derived from `events` (one bookmark per reel-selected event).

## Stats (`analysis_player_stats.json`, `analysis_team_stats.json`)

Units are SI. `pitch_calibration.confidence` tells the UI how much to trust
metres (auto-estimated from player spread vs user-supplied corners).

```json
{
  "generated_at": "...",
  "pitch_calibration": {"source": "auto|manual", "confidence": 0.6, "pitch_length_m": 105, "pitch_width_m": 68,
                        "image_corners_px": [[x,y],[x,y],[x,y],[x,y]], "homography": [[...],[...],[...]]},
  "focus_player_track_id": 7,
  "players": [
    {"track_id": 7, "team": 0, "team_name": "HOME", "label": null, "jersey_number": null,
     "minutes_tracked": 71.2, "distance_m": 8420.5, "top_speed_mps": 8.1, "avg_speed_mps": 1.9,
     "sprints": 23, "sprint_distance_m": 610.0, "high_intensity_distance_m": 1410.0,
     "touches": 41, "passes_attempted": 30, "passes_completed": 24, "shots": 2,
     "time_in_thirds_pct": {"defensive": 22.0, "middle": 51.0, "attacking": 27.0},
     "heatmap": {"bins_x": 21, "bins_y": 14, "grid": [[0.0]]},
     "speed_series": [{"t": 0.0, "v": 1.2}]  // ≤ 1 Hz
    }
  ]
}
```

```json
{
  "generated_at": "...",
  "teams": {
    "0": {"name": "HOME", "color_hex": "#d32f2f", "defends_first_half": "left",
          "possession_pct": 54.2, "passes": 310, "pass_accuracy_pct": 78.1,
          "shots": 9, "shots_on_target": 4, "goals": 2, "corners": 5,
          "territory_pct": {"defensive": 30.1, "middle": 40.2, "attacking": 29.7},
          "avg_speed_mps": 1.8, "distance_m": 90000.0,
          "play_speed": {"ball_speed_mean_mps": 6.1, "progression_mps": 1.4, "passes_per_minute": 7.2}},
    "1": {"...": "..."}
  },
  "timeline": {"bin_s": 60, "possession_pct_team0": [50.0], "momentum": [0.1], "shots_team0": [0], "shots_team1": [0]},
  "goal_attribution": [{"t": 812.4, "side": "left", "team": 1}],
  "quality": {"team_label_coverage_pct": 92.0, "ball_coverage_pct": 71.0}
}
```

## Progress (`progress.json`)

Written by the engine every few seconds so the API can show an ETA without
parsing logs:
`{stage, stage_index, stage_count, progress, stage_progress, eta_s, elapsed_s, fps_processing, message, device, cancelled}`.

## Profiles

`profile` in the job config picks defaults (overridable per key):

| key | fast | balanced | quality |
|---|---|---|---|
| proxy_height | 720 | 1080 | 1080 |
| inference_imgsz | 960 | 1280 | 1536 |
| vid_stride | 2 | 1 | 1 |
| detector | yolov8n | yolov8s/yolo26s | yolov8m/yolo26m |
| batch_size | auto | auto | auto |
| output_height | 1080 | 1080 | 1440 |
| debug_video | off | off | off |
| ball_tiles (high-res ball pass) | off | off | on |
