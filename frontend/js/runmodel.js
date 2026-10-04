// Normalizes a /v1/studio/runs/{id} summary into one model the match views
// share: teams, events (window timebase), sources with their timebase,
// stats (team stats v2, or v1 converted), labels and data-quality flags.

export const DEFAULT_TEAM_COLORS = ['#e53935', '#29b6f6'];

const TYPES = {
  goal: { label: 'Goal', c: 'goal', icon: 'ball' },
  shot: { label: 'Shot', c: 'shot', icon: 'target' },
  save: { label: 'Save', c: 'save', icon: 'user' },
  chance: { label: 'Chance', c: 'chance', icon: 'sparkle' },
  corner_kick: { label: 'Corner', c: 'setpiece', icon: 'whistle' },
  free_kick: { label: 'Free kick', c: 'setpiece', icon: 'whistle' },
  penalty_kick: { label: 'Penalty', c: 'setpiece', icon: 'whistle' },
  goal_kick: { label: 'Goal kick', c: 'setpiece', icon: 'whistle' },
  kickoff: { label: 'Kick-off', c: 'setpiece', icon: 'whistle' },
  yellow_card: { label: 'Yellow card', c: 'yellow', icon: 'card' },
  red_card: { label: 'Red card', c: 'red', icon: 'card' },
  sprint: { label: 'Sprint', c: 'run', icon: 'speed' },
  dribble: { label: 'Dribble', c: 'run', icon: 'speed' },
  turnover: { label: 'Turnover', c: 'other', icon: 'rerun' },
  foul: { label: 'Foul', c: 'other', icon: 'whistle' },
  foul_candidate: { label: 'Possible foul', c: 'other', icon: 'whistle' },
};

export function typeInfo(type) {
  const info = TYPES[type] || { label: String(type || 'Event').replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase()), c: 'other', icon: 'tag' };
  return { ...info, color: `var(--ev-${info.c})` };
}

export const typeLabel = (type) => typeInfo(type).label;

// Timebase per source: 'window' views share the processing-window clock;
// 'own' views (reel, clips) have their own and hide timeline markers.
function sources(run) {
  const v = run.videos || {};
  const list = [];
  if (v.movie) list.push({ key: 'movie', label: 'Game camera', file: v.movie, timebase: 'window', wide: false });
  if (v.proxy) list.push({ key: 'wide', label: 'Wide', file: v.proxy, timebase: 'window', wide: true });
  else if (v.original) list.push({ key: 'wide', label: 'Wide', file: v.original, timebase: 'window', wide: true });
  if (v.reel) list.push({ key: 'reel', label: 'Highlights reel', file: v.reel, timebase: 'own' });
  if (v.montage) list.push({ key: 'montage', label: 'Montage', file: v.montage, timebase: 'own' });
  (v.clips || []).forEach((file, i) => list.push({ key: `clip${i}`, label: `Clip ${i + 1}`, file, timebase: 'own', clip: true }));
  (v.spotlight || []).forEach((file, i) => list.push({ key: `spot${i}`, label: `Spotlight ${i + 1}`, file, timebase: 'own', clip: true }));
  if (v.debug) list.push({ key: 'debug', label: 'Debug', file: v.debug, timebase: 'window', wide: true });
  (v.movies || []).slice(1).forEach((file, i) => list.push({ key: `movie${i + 2}`, label: `Camera ${i + 2}`, file, timebase: 'window' }));
  return list;
}

function normalizeTeamStats(raw, names, colors) {
  const out = { teams: {}, timeline: null, quality: {}, goal_attribution: [], version: 0 };
  if (!raw || typeof raw !== 'object') return out;
  if (raw.teams && !Array.isArray(raw.teams)) {
    out.version = 2;
    for (const key of ['0', '1']) out.teams[key] = { ...(raw.teams[key] || {}) };
    out.timeline = raw.timeline || null;
    out.quality = raw.quality || {};
    out.goal_attribution = raw.goal_attribution || [];
  } else if (Array.isArray(raw.teams)) {
    out.version = 1;
    raw.teams.slice(0, 2).forEach((team, i) => {
      out.teams[String(i)] = {
        name: team.team || names[i],
        color_hex: team.color || colors[i],
        possession_pct: raw.possession_pct?.[team.team] ?? null,
        goals: raw.goals?.[team.team] ?? null,
      };
    });
  }
  return out;
}

export function buildModel(run) {
  const names = (run.team_names || ['Home', 'Away']).slice(0, 2);
  const colors = [0, 1].map((i) => (run.team_colors || [])[i] || DEFAULT_TEAM_COLORS[i]);
  const teams = [0, 1].map((i) => ({ idx: i, name: names[i] || (i ? 'Away' : 'Home'), color: colors[i] }));
  const events = (run.events || [])
    .map((event) => ({ ...event, t: +event.t || 0, info: typeInfo(event.type) }))
    .sort((a, b) => a.t - b.t);
  const players = new Map();
  for (const player of run.player_stats?.players || []) players.set(Number(player.track_id), player);
  const meta = run.tracks_meta || {};
  const duration = +run.duration_s || +meta.duration_s || (events.length ? events[events.length - 1].t + 10 : 0);
  const model = {
    run,
    runId: run.run_id,
    matchId: run.match_id,
    jobId: run.job_id,
    trim: +run.trim_offset_seconds || 0,
    duration,
    teams,
    score: run.score,
    events,
    reelPlan: run.reel_plan || {},
    players,
    labels: { ...(run.player_labels || {}) },
    calibration: run.player_stats?.pitch_calibration || null,
    // Saved manual calibration (calibration.json): normalized TL, TR, BR, BL.
    manualCalibration: run.calibration || null,
    teamStats: normalizeTeamStats(run.team_stats, names, colors),
    thumbs: run.thumbs || [],
    sources: sources(run),
    frame: { w: +meta.frame_width || 0, h: +meta.frame_height || 0 },
    tracksAvailable: !!run.tracks_available,
  };
  for (const [key, team] of Object.entries(model.teamStats.teams)) {
    if (team.color_hex) teams[+key].color = team.color_hex;
    if (team.name) teams[+key].name = team.name;
  }
  return model;
}

export function teamColor(model, team) {
  return team === 0 || team === 1 ? model.teams[team].color : '#9aa3af';
}

export function teamName(model, team) {
  if (team === 0 || team === 1) return model.teams[team].name;
  if (team === 2) return 'Referee';
  return 'Unassigned';
}

// "#10 Sam Kerr", "#10", or "Player 7" for a tracker id.
export function playerName(model, trackId, fallbackJersey) {
  const label = model.labels[String(trackId)] || {};
  const stats = model.players.get(Number(trackId)) || {};
  const number = label.number || stats.jersey_number || fallbackJersey;
  const name = label.name || stats.label;
  if (name && number != null) return `#${number} ${name}`;
  if (name) return name;
  if (number != null && number !== '') return `#${number}`;
  return `Player ${trackId}`;
}

export function qualityFlags(model) {
  const flags = [];
  const cal = model.calibration;
  if (cal && cal.confidence != null) {
    const c = +cal.confidence;
    flags.push({ key: 'calibration', label: `Pitch calibration ${Math.round(c * 100)}%${cal.source ? ` · ${cal.source}` : ''}`,
      level: c >= 0.75 ? 'ok' : c >= 0.45 ? 'warn' : 'danger',
      tip: 'How well image pixels map to metres. Distances and speeds are only as good as this.' });
  } else {
    flags.push({ key: 'calibration', label: 'No pitch calibration', level: 'danger', tip: 'Metres-based stats are unavailable or image-space only.' });
  }
  const q = model.teamStats.quality || {};
  if (q.team_label_coverage_pct != null) {
    const v = +q.team_label_coverage_pct;
    flags.push({ key: 'teams', label: `Team labels ${Math.round(v)}%`, level: v >= 85 ? 'ok' : v >= 60 ? 'warn' : 'danger',
      tip: 'Share of player detections assigned to a team. Drives possession and attribution.' });
  }
  if (q.ball_coverage_pct != null) {
    const v = +q.ball_coverage_pct;
    flags.push({ key: 'ball', label: `Ball seen ${Math.round(v)}%`, level: v >= 70 ? 'ok' : v >= 45 ? 'warn' : 'danger',
      tip: 'Share of in-play time the ball was located. Low values make possession and passing estimates rough.' });
  }
  return flags;
}
