// Stats tab: data-quality badges, team comparison, possession/momentum
// charts, territory + play speed, shot map, the 15-stat match sheet (stat
// catalog with evidence drill-down) and the sortable per-player table.

import { get } from '../../api.js';
import { donut, momentumChart, possessionBars, stackBar } from '../../charts.js';
import { calibrationMapper } from '../../homography.js';
import { icon } from '../../icons.js';
import { eventLocation, PITCH, pitchSvg } from '../../pitch.js';
import { playerName, qualityFlags, teamColor, teamName } from '../../runmodel.js';
import { $, $$, esc, fmtClock, fmtMs, num } from '../../ui.js';
import { openPlayerCard } from './playercard.js';

function eventCount(model, team, types) {
  return model.events.filter((e) => e.team === team && types.includes(e.type)).length;
}

function teamValue(model, team, key) {
  const t = model.teamStats.teams[String(team)] || {};
  const ps = t.play_speed || {};
  switch (key) {
    case 'possession': return t.possession_pct;
    case 'passes': return t.passes;
    case 'pass_acc': return t.pass_accuracy_pct;
    case 'shots': return t.shots ?? (model.events.length ? eventCount(model, team, ['shot', 'goal']) : null);
    case 'on_target': return t.shots_on_target;
    case 'goals': return t.goals ?? model.score?.[String(team)];
    case 'corners': return t.corners ?? (model.events.length ? eventCount(model, team, ['corner_kick']) : null);
    case 'distance': return t.distance_m != null ? t.distance_m / 1000 : null;
    case 'avg_speed': return t.avg_speed_mps != null ? t.avg_speed_mps * 3.6 : null;
    case 'ball_speed': return ps.ball_speed_mean_mps;
    case 'progression': return ps.progression_mps;
    case 'ppm': return ps.passes_per_minute;
    default: return null;
  }
}

const ROWS = [
  ['goals', 'Goals', 0], ['shots', 'Shots', 0], ['on_target', 'On target', 0], ['possession', 'Possession %', 0],
  ['passes', 'Passes', 0], ['pass_acc', 'Pass accuracy %', 0], ['corners', 'Corners', 0],
  ['distance', 'Distance (km)', 1], ['avg_speed', 'Avg speed (km/h)', 1],
  ['ball_speed', 'Ball speed (m/s)', 1], ['progression', 'Progression (m/s)', 2], ['ppm', 'Passes / min', 1],
];

function comparison(model) {
  const [a, b] = model.teams;
  return `<div class="cmphead"><div><span class="swatch" style="background:${esc(a.color)}"></span> ${esc(a.name)}</div><div class="faint xs">vs</div>
    <div class="b">${esc(b.name)} <span class="swatch" style="background:${esc(b.color)}"></span></div></div>
    <div class="cmp">${ROWS.map(([key, label, digits]) => {
      const va = teamValue(model, 0, key);
      const vb = teamValue(model, 1, key);
      const known = va != null || vb != null;
      const total = (+va || 0) + (+vb || 0) || 1;
      return `<div class="rowc ${known ? '' : 'na'}" title="${known ? '' : 'Not measured in this run'}">
        <div class="va">${num(va, digits)}</div>
        <div class="half l"><i style="width:${known ? (100 * (+va || 0)) / total : 0}%;background:${esc(a.color)}"></i></div>
        <div class="lab">${esc(label)}</div>
        <div class="half"><i style="width:${known ? (100 * (+vb || 0)) / total : 0}%;background:${esc(b.color)}"></i></div>
        <div class="vb">${num(vb, digits)}</div></div>`;
    }).join('')}</div>`;
}

function possessionPanel(model) {
  const a = teamValue(model, 0, 'possession');
  const b = teamValue(model, 1, 'possession');
  if (a == null && b == null) {
    return `<div class="panel"><div class="panel-title">Possession</div><div class="note">Possession was not measured in this run.</div></div>`;
  }
  return `<div class="panel"><div class="panel-title">Possession</div>
    <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;justify-content:center">
      ${donut({ values: [a || 0, b ?? 100 - (a || 0)], colors: model.teams.map((t) => t.color), label: `${Math.round(a || 0)}%`, sublabel: model.teams[0].name, size: 160 })}
      <div class="legend-inline" style="flex-direction:column;gap:8px">
        ${model.teams.map((t, i) => `<span><span class="swatch" style="background:${esc(t.color)}"></span>${esc(t.name)} <b class="num">${num(i ? b : a, 1)}%</b></span>`).join('')}
      </div></div></div>`;
}

function timelinePanels(model) {
  const tl = model.teamStats.timeline;
  if (!tl?.possession_pct_team0?.length && !tl?.momentum?.length) return '';
  const binS = +tl.bin_s || 60;
  const colors = model.teams.map((t) => t.color);
  const goals = model.events.filter((e) => e.type === 'goal').map((e) => ({ x: e.t, color: teamColor(model, e.team), title: `Goal ${teamName(model, e.team)} ${fmtClock(e.t)}` }));
  return `<div class="cols-2">
    ${tl.possession_pct_team0?.length ? `<div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Possession by period</div>
      <div class="legend-inline">${model.teams.map((t) => `<span><span class="swatch" style="background:${esc(t.color)}"></span>${esc(t.name)}</span>`).join('')}</div></div>
      ${possessionBars({ values: tl.possession_pct_team0, binS, colors })}<div class="note">Bars above the line: ${esc(model.teams[0].name)} had more of the ball in that ${binS >= 60 ? `${Math.round(binS / 60)}-minute` : `${Math.round(binS)}-second`} period.</div></div>` : ''}
    ${tl.momentum?.length ? `<div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Momentum</div>
      <div class="legend-inline"><span>${icon('ball', 'sm')} goals</span></div></div>
      ${momentumChart({ values: tl.momentum, binS, colors, markers: goals })}<div class="note">Positive: ${esc(model.teams[0].name)} on top (territory, shots, pressure).</div></div>` : ''}
  </div>`;
}

function territoryPanel(model) {
  const rows = model.teams.map((t) => {
    const terr = model.teamStats.teams[String(t.idx)]?.territory_pct;
    if (!terr) return '';
    return `<div style="margin-bottom:12px"><div class="small" style="margin-bottom:4px;font-weight:600">${esc(t.name)}</div>${stackBar([
      { label: 'Defensive third', value: terr.defensive, color: '#64748b' },
      { label: 'Middle third', value: terr.middle, color: '#94a3b8' },
      { label: 'Attacking third', value: terr.attacking, color: t.color }])}</div>`;
  }).join('');
  return `<div class="panel"><div class="panel-title">Territory <span class="faint xs">own → opposition third</span></div>
    ${rows || '<div class="note">Territory needs pitch calibration and team labels.</div>'}</div>`;
}

function shotMap(model) {
  const shots = model.events.filter((e) => ['shot', 'goal', 'save', 'chance'].includes(e.type));
  const placed = [];
  const mapper = calibrationMapper(model);
  let mode = null;
  for (const e of shots) {
    const loc = eventLocation(e, mapper, model.frame);
    if (!loc) continue;
    mode = mode === 'image' || loc.mode === 'image' ? 'image' : loc.mode;
    placed.push({ e, x: Math.max(0, Math.min(PITCH.L, loc.xy[0])), y: Math.max(0, Math.min(PITCH.W, loc.xy[1])) });
  }
  const dots = placed.map(({ e, x, y }) => {
    const color = teamColor(model, e.team);
    const r = e.type === 'goal' ? 1.9 : 1.3;
    return `<g><circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="${e.type === 'goal' ? esc(color) : 'none'}" stroke="${esc(color)}" stroke-width="0.6"/>
      ${e.type === 'goal' ? `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r + 1}" fill="none" stroke="#fff" stroke-width="0.25" opacity=".8"/>` : ''}
      <title>${esc(e.info.label)} · ${esc(teamName(model, e.team))} · ${fmtClock(e.t)}</title></g>`;
  }).join('');
  const pending = model.calibration?.source !== 'manual';
  const note = mode === 'image' ? 'Image-space positions (no pitch calibration) — approximate.'
    : mode === 'manual' ? `Projected through your manual pitch calibration${pending ? ' (re-analyze to update the metres-based stats too)' : ''}.`
      : mode === 'calibrated' ? `Projected through the ${model.calibration?.source || 'auto'} pitch calibration.` : '';
  return `<div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Shot map</div>
    <div class="legend-inline"><span><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="var(--text-2)"/></svg>Goal</span>
    <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--text-2)" stroke-width="1.5"/></svg>Shot / chance</span></div></div>
    ${placed.length ? pitchSvg(dots, { label: 'Shot map' }) : `<div class="empty" style="padding:32px">${shots.length ? 'Shots were detected but carry no location.' : 'No shots detected.'}</div>`}
    ${note ? `<div class="note">${esc(note)}</div>` : ''}</div>`;
}

// Calibration badge: what the analysis used (analysis_player_stats.pitch_calibration)
// plus a shortcut into the Watch tab's calibration mode when it was not manual.
function calibrationBadge(model) {
  const cal = model.calibration;
  const saved = model.manualCalibration;
  const source = cal?.source || null;
  const conf = cal?.confidence != null ? +cal.confidence : null;
  const level = conf == null ? 'danger' : conf >= 0.75 ? 'ok' : conf >= 0.45 ? 'warn' : 'danger';
  const label = cal ? `Pitch calibration · ${source === 'manual' ? 'manual' : 'auto'}${conf != null ? ` ${Math.round(conf * 100)}%` : ''}` : 'No pitch calibration';
  const tip = source === 'manual' ? 'Metres come from the pitch corners you clicked.'
    : 'Estimated from player spread. Click the 4 pitch corners for accurate distances, speeds, passes and goal detection.';
  const pending = saved && source !== 'manual';
  const canWatch = model.sources.some((s) => s.wide);
  return `<span class="calbadge"><span class="badge ${level}" id="calbadge" title="${esc(tip)}"><span class="dot"></span>${esc(label)}</span>
    ${pending ? `<span class="badge info" title="Saved ${esc(saved.updated_at || '')}">Manual corners saved · re-analyze to apply</span>` : ''}
    ${source !== 'manual' && canWatch ? `<button class="btn-ghost sm" id="calgo">${icon('pitch', 'sm')} ${pending ? 'Review calibration' : 'Calibrate'}</button>` : ''}</span>`;
}

/* ---------------- stat catalog (DB-derived match sheet) ---------------- */

function statTile(stat) {
  const fmt = (value) => (value == null ? '–' : (stat.unit === 'percent' ? `${value}%` : `${value}`));
  if (!stat.available) {
    const why = {
      no_completed_analysis: 'awaiting analysis', not_detected_by_pipeline: 'coming soon',
      team_stats_artifact_missing: 'needs a full run', not_available_for_source: `not available from ${stat.raw?.source_label || 'this source'}`,
    }[stat.reason] || stat.reason || 'unavailable';
    return `<div class="stat na"><div class="k">${esc(stat.label)}</div><div class="vals"><span>–</span><span class="mid">|</span><span>–</span></div>
      <div class="why">${esc(why)}</div></div>`;
  }
  const linked = (stat.event_ids || []).length ? ' linked' : '';
  return `<div class="stat${linked}" data-key="${esc(stat.key)}" title="${esc(stat.method || '')}" ${linked ? 'tabindex="0" role="button"' : ''}>
    <div class="k">${esc(stat.label)}</div>
    <div class="vals"><span>${fmt(stat.home)}</span><span class="mid">home · away</span><span>${fmt(stat.away)}</span></div>
    ${stat.unattributed ? `<div class="why">+${stat.unattributed} unattributed</div>` : ''}</div>`;
}

async function loadCatalog(ctx, box) {
  try {
    const data = await get(`/matches/${ctx.matchId}/stats`);
    if (!box.isConnected) return;
    const note = data.analysis.has_completed_job ? ''
      : `<div class="warnnote">${icon('info', 'sm')}<span>No completed analysis yet — stats appear after the first run finishes.</span></div>`;
    const sourceNote = data.analysis.source_supported_stat_count < 15
      ? `<div class="warnnote">${icon('alert', 'sm')}<span>Source: ${esc(data.analysis.source_label)} — only ${data.analysis.source_supported_stat_count} of 15 stats are computable. Upload the raw file to get the full sheet.</span></div>` : '';
    box.innerHTML = `<div class="panel-title">Match sheet <span class="faint xs">${esc(data.teams.home || 'Home')} · ${esc(data.teams.away || 'Away')}</span></div>
      <div class="statgrid">${data.stats.map(statTile).join('')}</div>
      <div class="note">Counted from reviewed events. Greyed stats can’t be computed from this footage yet — never shown as a fake 0. Click a stat to see its evidence.</div>
      ${sourceNote}${note}<div id="drill"></div>`;
    $$('.stat.linked', box).forEach((tile) => {
      const open = () => drill(ctx, box, data.stats.find((s) => s.key === tile.dataset.key), tile);
      tile.onclick = open;
      tile.onkeydown = (event) => { if (event.key === 'Enter') open(); };
    });
  } catch (error) {
    box.innerHTML = `<div class="panel-title">Match sheet</div><div class="errnote">${esc(error.message)}</div>`;
  }
}

async function drill(ctx, box, stat, tile) {
  const selected = tile.classList.toggle('selected');
  $$('.stat.selected', box).forEach((other) => { if (other !== tile) other.classList.remove('selected'); });
  const target = $('#drill', box);
  if (!selected) { target.innerHTML = ''; return; }
  target.innerHTML = '<div class="sk sk-line"></div>';
  if (!ctx.dbEvents) ctx.dbEvents = (await get(`/matches/${ctx.matchId}/events?limit=500`)).items;
  const ids = new Set(stat.event_ids || []);
  const rows = ctx.dbEvents.filter((e) => ids.has(e.event_id)).sort((a, b) => a.occurred_at_ms - b.occurred_at_ms);
  target.innerHTML = `<div class="divider"></div><div class="small" style="font-weight:600;margin-bottom:8px">Evidence for ${esc(stat.label)} (${rows.length})</div>
    <div class="tablewrap"><table><thead><tr><th>Source time</th><th>Type</th><th>Team</th><th>Confidence</th><th></th></tr></thead><tbody>
    ${rows.map((e) => `<tr><td class="num">${fmtMs(e.occurred_at_ms)}</td><td>${esc(e.event_type)}</td><td>${esc(e.team_id || '—')}</td>
      <td class="num">${Math.round((+e.confidence || 0) * 100)}%</td>
      <td class="r"><button class="btn-ghost sm" data-w="${e.occurred_at_ms / 1000 - ctx.model.trim}">${icon('play', 'sm')} Watch</button></td></tr>`).join('')}
    </tbody></table></div>`;
  $$('[data-w]', target).forEach((button) => { button.onclick = () => ctx.switchTab('watch', { seek: Math.max(0, +button.dataset.w) }); });
}

/* ---------------- per-player table ---------------- */

const PCOLS = [
  ['name', 'Player', (m, p) => playerName(m, p.track_id, p.jersey_number), 's'],
  ['team', 'Team', (m, p) => teamName(m, p.team), 's'],
  ['minutes', 'Min', (m, p) => p.minutes_tracked, 1],
  ['distance', 'Dist km', (m, p) => (p.distance_m != null ? p.distance_m / 1000 : null), 2],
  ['top', 'Top km/h', (m, p) => (p.top_speed_mps != null ? p.top_speed_mps * 3.6 : null), 1],
  ['sprints', 'Sprints', (m, p) => p.sprints, 0],
  ['touches', 'Touches', (m, p) => p.touches, 0],
  ['passes', 'Passes', (m, p) => p.passes_attempted, 0],
  ['passpct', 'Pass %', (m, p) => (p.passes_attempted ? (100 * (p.passes_completed || 0)) / p.passes_attempted : null), 0],
  ['shots', 'Shots', (m, p) => p.shots, 0],
];

function playerTable(ctx, box) {
  const model = ctx.model;
  const st = ctx.playerTable || (ctx.playerTable = { sort: 'distance', dir: -1, team: 'all' });
  const players = [...model.players.values()].filter((p) => st.team === 'all' || String(p.team) === st.team);
  const col = PCOLS.find((c) => c[0] === st.sort) || PCOLS[3];
  players.sort((a, b) => {
    const va = col[2](model, a);
    const vb = col[2](model, b);
    if (col[3] === 's') return String(va).localeCompare(String(vb)) * st.dir;
    return ((va ?? -1e9) - (vb ?? -1e9)) * st.dir;
  });
  const maxDist = Math.max(1, ...[...model.players.values()].map((p) => p.distance_m || 0));
  box.innerHTML = `<div class="charthead"><div class="panel-title" style="margin:0">Players <span class="faint xs">${model.players.size} tracked</span></div>
    <div class="seg" role="group" aria-label="Team filter">
      <button data-team="all" class="${st.team === 'all' ? 'active' : ''}">All</button>
      ${model.teams.map((t) => `<button data-team="${t.idx}" class="${st.team === String(t.idx) ? 'active' : ''}"><span class="swatch" style="background:${esc(t.color)}"></span>${esc(t.name)}</button>`).join('')}
    </div></div>
    ${players.length ? `<div class="tablewrap"><table><thead><tr>${PCOLS.map(([key, label, , d]) =>
      `<th class="sortable ${d === 's' ? '' : 'r'} ${st.sort === key ? 'sorted' : ''}" data-sort="${key}" aria-sort="${st.sort === key ? (st.dir > 0 ? 'ascending' : 'descending') : 'none'}">${label}${st.sort === key ? (st.dir > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('')}</tr></thead>
    <tbody>${players.map((p) => `<tr class="clickable" data-tid="${p.track_id}" tabindex="0">${PCOLS.map(([key, , fn, d]) => {
      const v = fn(model, p);
      if (key === 'name') return `<td><span style="display:inline-flex;align-items:center;gap:8px"><span class="swatch" style="background:${esc(teamColor(model, p.team))}"></span><b>${esc(v)}</b></span></td>`;
      if (key === 'team') return `<td class="muted">${esc(v)}</td>`;
      if (key === 'distance') return `<td class="r"><span style="display:inline-flex;gap:8px;align-items:center">${num(v, d)}<span class="meter"><i style="width:${(100 * (p.distance_m || 0)) / maxDist}%;background:${esc(teamColor(model, p.team))}"></i></span></span></td>`;
      return `<td class="r">${num(v, d)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>`
    : '<div class="empty" style="padding:32px">No per-player stats in this run.</div>'}`;
  $$('[data-sort]', box).forEach((th) => {
    th.onclick = () => {
      if (st.sort === th.dataset.sort) st.dir *= -1; else { st.sort = th.dataset.sort; st.dir = PCOLS.find((c) => c[0] === st.sort)[3] === 's' ? 1 : -1; }
      playerTable(ctx, box);
    };
  });
  $$('[data-team]', box).forEach((button) => { button.onclick = () => { st.team = button.dataset.team; playerTable(ctx, box); }; });
  $$('tr[data-tid]', box).forEach((row) => {
    const open = () => openPlayerCard(ctx, +row.dataset.tid, {
      onFilterEvents: (tid) => { ctx.eventFilter = { type: '', team: '', player: String(tid), reel: false }; ctx.switchTab('watch'); },
    });
    row.onclick = open;
    row.onkeydown = (event) => { if (event.key === 'Enter') open(); };
  });
}

export function renderStatsTab(body, ctx) {
  const model = ctx.model;
  const flags = qualityFlags(model).filter((f) => f.key !== 'calibration');
  body.innerHTML = `
    <div class="charthead" style="margin-bottom:16px">
      <div class="quality">${calibrationBadge(model)}${flags.map((f) => `<span class="badge ${f.level}" title="${esc(f.tip)}"><span class="dot"></span>${esc(f.label)}</span>`).join('')}</div>
      <div class="faint xs">${model.teamStats.version ? `Team stats v${model.teamStats.version}` : 'No team stats artifact'} · ${model.players.size} players</div>
    </div>
    <div class="cols-split">
      ${possessionPanel(model)}
      <div class="panel"><div class="panel-title">Team comparison</div>${comparison(model)}</div>
    </div>
    ${timelinePanels(model)}
    <div class="cols-2">${territoryPanel(model)}${shotMap(model)}</div>
    ${ctx.matchId ? '<div class="panel" id="catalog"><div class="panel-title">Match sheet</div><div class="sk sk-line"></div><div class="sk sk-line"></div></div>' : ''}
    <div class="panel" id="ptable"></div>`;
  playerTable(ctx, $('#ptable', body));
  const calgo = $('#calgo', body);
  if (calgo) calgo.onclick = () => { window.scrollTo(0, 0); ctx.switchTab('watch', { calibrate: true }); };
  if (ctx.matchId) loadCatalog(ctx, $('#catalog', body));
  const onLabels = () => playerTable(ctx, $('#ptable', body));
  window.addEventListener('vh-labels-changed', onLabels);
  return () => window.removeEventListener('vh-labels-changed', onLabels);
}
