// Report tab: the AI match report (safe Markdown) plus run diagnostics —
// camera smoothness vs. the acceptance thresholds, game states, card crops.

import { runFileUrl } from '../../api.js';
import { icon } from '../../icons.js';
import { renderMarkdown } from '../../markdown.js';
import { emptyState, esc, fmtDuration, num, titleCase } from '../../ui.js';

const CAMERA_CHECKS = [
  ['pan_speed_p95_cropw_per_s', 'Pan speed p95', (v) => `${num(v, 2)} crop-w/s`, (v) => v < 0.9, '< 0.9'],
  ['zoom_rate_p95_per_s', 'Zoom rate p95', (v) => `${num(v, 2)}×/s`, (v) => v < 0.15, '< 0.15'],
  ['hard_snaps', 'Hard snaps', (v) => num(v), (v) => +v === 0, '0'],
  ['ball_in_frame_fraction', 'Ball in frame', (v) => `${num(v * 100, 1)}%`, (v) => v > 0.97, '> 97%'],
  ['zoom_reversals_per_min', 'Zoom reversals', (v) => `${num(v, 1)}/min`, () => true, ''],
  ['pan_accel_p95', 'Pan accel p95', (v) => num(v, 2), () => true, ''],
];

function cameraPanel(quality) {
  const rows = CAMERA_CHECKS.filter(([key]) => quality?.[key] != null);
  if (!rows.length) return '';
  return `<div class="panel"><div class="panel-title">Game camera quality</div>
    <table><tbody>${rows.map(([key, label, fmt, pass, target]) => {
      const ok = pass(+quality[key]);
      return `<tr><td>${esc(label)}</td><td class="r num">${fmt(+quality[key])}</td>
        <td class="r">${target ? `<span class="badge ${ok ? 'ok' : 'warn'}" title="target ${esc(target)}">${ok ? 'pass' : `target ${esc(target)}`}</span>` : ''}</td></tr>`;
    }).join('')}</tbody></table></div>`;
}

export function renderReportTab(body, ctx) {
  const run = ctx.model.run;
  const states = Object.entries(run.state_summary_s || {});
  const totalState = states.reduce((s, [, v]) => s + (+v || 0), 0) || 1;
  body.innerHTML = `<div class="cols-split rev">
    <div class="panel" style="padding:32px">${run.match_report
      ? `<article class="md">${renderMarkdown(run.match_report)}</article>`
      : emptyState('report', 'No match report', 'Enable the AI match report when starting a run (needs a local Ollama or an API key).')}</div>
    <div>
      ${cameraPanel(run.camera_quality)}
      ${states.length ? `<div class="panel"><div class="panel-title">Game states</div>${states.map(([key, value]) => `
        <div style="margin-bottom:8px"><div class="charthead" style="margin:0"><span class="small">${esc(titleCase(key))}</span><span class="small num">${fmtDuration(value)}</span></div>
        <div class="bar"><i style="width:${(100 * value) / totalState}%"></i></div></div>`).join('')}</div>` : ''}
      ${(run.card_crops || []).length ? `<div class="panel"><div class="panel-title"><span class="ttl-i">${icon('card', 'sm')} Card review crops</span></div>
        <div class="crops">${run.card_crops.map((c) => `<img src="${esc(runFileUrl(ctx.runId, c))}" alt="${esc(c)}" loading="lazy">`).join('')}</div></div>` : ''}
      <div class="panel"><div class="panel-title">Run</div><dl class="kv">
        <dt>Run</dt><dd class="mono">${esc(ctx.runId)}</dd>
        ${run.job_id ? `<dt>Job</dt><dd><a href="#jobs/${esc(run.job_id)}">${esc(run.job_id)}</a></dd>` : ''}
        ${run.video_path ? `<dt>Source</dt><dd class="mono">${esc(run.video_path)}</dd>` : ''}
        <dt>Window</dt><dd>${fmtDuration(ctx.model.trim)} → ${fmtDuration(ctx.model.trim + ctx.model.duration)}</dd>
        <dt>Events</dt><dd>${ctx.model.events.length} (${esc(run.events_source)})</dd>
        <dt>Tracks</dt><dd>${run.tracks_available ? `${num(run.tracks_meta?.player_track_count)} players · ${num(run.tracks_meta?.fps, 0)} fps` : 'not saved'}</dd>
      </dl></div>
    </div></div>`;
}
