// Player card drawer: identity + label editing, physical stats, heatmap,
// speed series, the player's events, and "Track this player" actions.

import { get } from '../../api.js';
import { lineChart, stackBar } from '../../charts.js';
import { icon } from '../../icons.js';
import { heatmapLayer, pitchSvg } from '../../pitch.js';
import { playerName, teamColor, teamName } from '../../runmodel.js';
import { $, esc, fmtClock, num, openDrawer, toast } from '../../ui.js';
import { rerunForPlayer, saveLabel } from './actions.js';

function kpi(label, value, unit = '', detail = '') {
  return `<div class="kpi"><div class="l">${esc(label)}</div><div class="v">${value}${unit ? `<small>${esc(unit)}</small>` : ''}</div>
    ${detail ? `<div class="d">${detail}</div>` : ''}</div>`;
}

function statsHtml(model, p) {
  if (!p) {
    return `<div class="warnnote">${icon('info', 'sm')}<span>No per-player stats for this track. It may be a short fragment,
      a referee, or the stats stage did not run. Tracking can still follow it.</span></div>`;
  }
  const passPct = p.passes_attempted ? Math.round((100 * (p.passes_completed || 0)) / p.passes_attempted) : null;
  const metresOk = (model.calibration?.confidence ?? 0) >= 0.3;
  const thirds = p.time_in_thirds_pct || {};
  return `<div class="kpis" style="grid-template-columns:repeat(2,1fr)">
      ${kpi('Distance', num((p.distance_m || 0) / 1000, 2), 'km', metresOk ? '' : 'uncalibrated')}
      ${kpi('Top speed', num((p.top_speed_mps || 0) * 3.6, 1), 'km/h', `${num(p.top_speed_mps, 1)} m/s`)}
      ${kpi('Sprints', num(p.sprints), '', p.sprint_distance_m != null ? `${num(p.sprint_distance_m)} m sprinting` : '')}
      ${kpi('Minutes tracked', num(p.minutes_tracked, 1), 'min', p.avg_speed_mps != null ? `avg ${num(p.avg_speed_mps * 3.6, 1)} km/h` : '')}
      ${kpi('Touches', num(p.touches))}
      ${kpi('Passes', `${num(p.passes_completed)}/${num(p.passes_attempted)}`, '', passPct != null ? `${passPct}% completed` : '')}
      ${kpi('Shots', num(p.shots), '', p.goals ? `${p.goals} goal${p.goals > 1 ? 's' : ''}` : '')}
      ${kpi('High intensity', num(p.high_intensity_distance_m), 'm')}
    </div>
    ${Object.keys(thirds).length ? `<div class="label">Time in thirds (own → opposition)</div>${stackBar([
      { label: 'Defensive', value: thirds.defensive, color: '#64748b' },
      { label: 'Middle', value: thirds.middle, color: '#94a3b8' },
      { label: 'Attacking', value: thirds.attacking, color: 'var(--accent)' }])}` : ''}
    ${p.heatmap?.grid?.length ? `<div class="label">Heatmap</div>${pitchSvg(heatmapLayer(p.heatmap, teamColor(model, p.team)), { label: 'Player heatmap' })}` : ''}
    <div class="label">Speed</div><div id="pc_speed">${p.has_speed_series || p.speed_series ? '<div class="sk" style="height:120px"></div>' : '<div class="note">No speed series recorded.</div>'}</div>`;
}

export function openPlayerCard(ctx, trackId, { team: seenTeam, jersey, onFilterEvents, onSeek } = {}) {
  const model = ctx.model;
  const p = model.players.get(Number(trackId));
  const team = p?.team ?? seenTeam;
  const label = model.labels[String(trackId)] || {};
  const myEvents = model.events.filter((e) => e.player_track_id === Number(trackId) || e.secondary_track_id === Number(trackId));
  const number = label.number || p?.jersey_number || jersey || '';
  const canRerun = !!ctx.jobId && model.tracksAvailable;
  const drawer = openDrawer({
    title: `<div class="pcard-head"><div class="jersey" style="--c:${esc(teamColor(model, team))}">${esc(number || '?')}</div>
      <div><h3>${esc(playerName(model, trackId, jersey))}</h3><div class="faint xs">${esc(teamName(model, team))} · track ${esc(trackId)}</div></div></div>`,
    html: `
      <details class="adv" style="margin-bottom:16px"><summary>${icon('chevron', 'sm')} Name this player</summary><div class="body">
        <div class="row"><div><label for="pc_name">Name</label><input type="text" id="pc_name" maxlength="64" value="${esc(label.name || '')}" placeholder="e.g. Sam Kerr"></div>
        <div><label for="pc_num">Shirt number</label><input type="text" id="pc_num" maxlength="4" value="${esc(label.number || '')}" placeholder="${esc(p?.jersey_number ?? '')}"></div></div>
        <button class="btn2 sm" id="pc_save" style="margin-top:12px">${icon('tag', 'sm')} Save label</button></div></details>
      ${statsHtml(model, p)}
      <div class="label">Events (${myEvents.length})</div>
      ${myEvents.length ? `<div>${myEvents.map((e) => `<button class="btn-ghost" data-seek="${e.t}" style="width:100%;justify-content:flex-start">
        <span class="swatch" style="background:${e.info.color}"></span>${esc(e.info.label)}<span class="faint" style="margin-left:auto">${fmtClock(e.t)}</span></button>`).join('')}</div>`
        : '<div class="note">No events attributed to this player.</div>'}`,
    footer: `<button class="btn" id="pc_track" ${canRerun ? '' : 'disabled'} title="${canRerun ? 'Re-render the game camera following this player (tracking is reused)' : 'Needs a run linked to a job with tracks.npz'}">${icon('target', 'sm')} Track this player</button>
      <button class="btn2" id="pc_spot" ${canRerun ? '' : 'disabled'}>${icon('sparkle', 'sm')} Spotlight reel</button>
      ${onFilterEvents ? `<button class="btn-ghost" id="pc_filter">${icon('filter', 'sm')} Show events</button>` : ''}`,
  });
  const el = drawer.el;

  $('#pc_save', el).onclick = async () => {
    try {
      await saveLabel(ctx, trackId, { name: $('#pc_name', el).value, number: $('#pc_num', el).value });
      drawer.close();
      openPlayerCard(ctx, trackId, { team: seenTeam, jersey, onFilterEvents, onSeek });
      el.dispatchEvent(new CustomEvent('vh-label', { bubbles: true }));
      window.dispatchEvent(new CustomEvent('vh-labels-changed'));
    } catch (error) { toast(error.message, 'err'); }
  };
  const runFor = (mode) => async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      const job = await rerunForPlayer(ctx, trackId, mode);
      $('.body', el).insertAdjacentHTML('afterbegin', `<div class="panel" style="border-color:var(--accent)">
        <b>Queued.</b> <span class="muted small">Job ${esc(job.job_id)} will render the ${mode === 'spotlight' ? 'spotlight reel' : 'follow-cam movie'} without re-detecting.</span>
        <div style="margin-top:8px"><a href="#jobs/${esc(job.job_id)}">Follow progress →</a></div></div>`);
    } catch (error) { toast(error.message, 'err'); button.disabled = false; }
  };
  $('#pc_track', el).onclick = runFor('track');
  $('#pc_spot', el).onclick = runFor('spotlight');
  if (onFilterEvents) $('#pc_filter', el).onclick = () => { onFilterEvents(trackId); drawer.close(); };
  el.querySelectorAll('[data-seek]').forEach((button) => {
    button.onclick = () => { if (onSeek) onSeek(+button.dataset.seek); else ctx.switchTab?.('watch', { seek: +button.dataset.seek }); drawer.close(); };
  });

  if (p && (p.has_speed_series || p.speed_series)) {
    const load = p.speed_series ? Promise.resolve({ player: p })
      : get(`/studio/runs/${encodeURIComponent(ctx.runId)}/player-stats/${encodeURIComponent(trackId)}`);
    load.then(({ player }) => {
      const box = $('#pc_speed', el);
      if (!box) return;
      const series = (player.speed_series || []).map((s) => ({ x: +s.t, y: (+s.v || 0) * 3.6 }));
      box.innerHTML = series.length ? lineChart({
        series: [{ values: series, color: teamColor(model, team), fill: true }], height: 150,
        xFmt: (v) => fmtClock(v), yFmt: (v) => `${Math.round(v)}`, label: 'Speed (km/h)', xMax: model.duration,
      }) + '<div class="note">km/h, 1 Hz</div>' : '<div class="note">No speed series recorded.</div>';
    }).catch(() => { const box = $('#pc_speed', el); if (box) box.innerHTML = '<div class="note">Speed series unavailable.</div>'; });
  }
  return drawer;
}
