// Jobs: one joined list (/studio/jobs) + a detail pane for the selected job
// with the stage pipeline, ETA (/studio/jobs/{id}/progress), per-stage
// timings, cancel / rerun / delete, log viewer with level filter, and
// completion-notification delivery. Polling stops on navigation.

import { del, get, post } from '../api.js';
import { icon } from '../icons.js';
import { go } from '../router.js';
import {
  $, $$, confirmDialog, emptyState, esc, every, fmtDate, fmtDuration, fmtRel, isCurrent, routeToken, setMain, skLines,
  statusBadge, titleCase, toast,
} from '../ui.js';

const ACTIVE = ['queued', 'claimed', 'running', 'cancel_requested'];
const DEFAULT_STAGES = ['queued', 'proxy', 'tracking', 'teams', 'analysis', 'camera_plan', 'render', 'clips_reel'];
const st = { selected: null, level: '', items: [], gpu: null, logsOpen: true };

function describe(config = {}) {
  const parts = [];
  if (config.profile) parts.push(titleCase(config.profile));
  if (config.camera_mode) parts.push(config.camera_mode.replace(/_/g, ' '));
  if (config.focus_track_id != null) parts.push(`player ${config.focus_track_id}`);
  if (config.reuse_tracking_from_job) parts.push('reused tracking');
  return parts.join(' · ');
}

function progressOf(job) {
  const live = job.live || {};
  return Math.max(+job.progress || 0, +live.progress || 0);
}

function jobRow(job) {
  const live = job.live || {};
  const pct = Math.round(100 * progressOf(job));
  const active = ACTIVE.includes(job.status);
  return `<div class="jobrow ${st.selected === job.job_id ? 'sel' : ''}" data-job="${esc(job.job_id)}" tabindex="0" role="button" aria-label="Job for ${esc(job.match_name || job.match_id)}">
    <div style="min-width:0"><div style="font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(job.match_name || job.match_id)}</div>
      <div class="faint xs">${esc(describe(job.config)) || '<span class="mono">' + esc(job.job_id) + '</span>'}</div></div>
    <div>${statusBadge(job.status)}</div>
    <div><div class="bar ${job.status === 'failed' ? 'danger' : job.status === 'completed' ? 'ok' : ''}"><i style="width:${pct}%"></i></div>
      <div class="faint xs" style="margin-top:4px">${active ? `${esc(titleCase(live.stage || job.stage || job.status))} · ${pct}%${live.eta_s ? ` · ${fmtDuration(live.eta_s)} left` : ''}` : esc(job.error_message || titleCase(job.stage || ''))}</div></div>
    <div class="faint xs">${esc(fmtRel(job.created_at))}</div>
    <div>${icon('chevron', 'sm')}</div></div>`;
}

function renderList() {
  const box = $('#joblist');
  if (!box) return;
  const active = st.items.filter((j) => ACTIVE.includes(j.status)).length;
  $('#jobsub').textContent = st.items.length ? `${st.items.length} recent · ${active} active` : 'Processing queue';
  box.innerHTML = st.items.length ? st.items.map(jobRow).join('')
    : `<div style="padding:16px">${emptyState('jobs', 'No jobs yet', 'Start a run and its progress shows up here.', `<a class="btn" href="#create">${icon('plus', 'sm')} New run</a>`)}</div>`;
  $$('[data-job]', box).forEach((row) => {
    const open = () => { if (st.selected !== row.dataset.job) go('jobs', row.dataset.job); };
    row.onclick = open;
    row.onkeydown = (event) => { if (event.key === 'Enter') open(); };
  });
}

function pipeline(progress, job) {
  const stages = Array.isArray(progress.stages) && progress.stages.length
    ? progress.stages.map((s) => (typeof s === 'string' ? s : s.name)) : DEFAULT_STAGES;
  const current = progress.stage || job.stage;
  let idx = stages.indexOf(current);
  if (idx < 0 && progress.stage_index != null) idx = +progress.stage_index;
  const done = job.status === 'completed';
  const timings = progress.timings || {};
  return `<div class="pipeline" aria-label="Pipeline">${stages.map((name, i) => {
    const cls = done || i < idx ? 'done' : i === idx && ACTIVE.includes(job.status) ? 'cur' : '';
    const sp = i === idx && progress.stage_progress != null ? Math.round(100 * progress.stage_progress) : null;
    const t = timings[name];
    return `<div class="ps ${cls}"><b>${esc(titleCase(name))}</b>${cls === 'done' ? (t != null ? fmtDuration(t) : 'done')
      : cls === 'cur' ? `${sp != null ? `${sp}%` : 'running'}<div class="bar"><i style="width:${sp ?? 30}%"></i></div>` : '<span class="faint">waiting</span>'}</div>`;
  }).join('')}</div>`;
}

function timingsChart(timings) {
  const entries = Object.entries(timings || {}).filter(([, v]) => +v > 0);
  if (!entries.length) return '';
  const total = entries.reduce((s, [, v]) => s + +v, 0);
  const max = Math.max(...entries.map(([, v]) => +v));
  return `<div class="panel"><div class="panel-title">Stage timings <span class="faint xs">${fmtDuration(total)} total</span></div>
    ${entries.map(([k, v]) => `<div style="display:grid;grid-template-columns:120px 1fr 64px;gap:12px;align-items:center;margin-bottom:6px">
      <span class="small">${esc(titleCase(k))}</span><div class="bar"><i style="width:${(100 * v) / max}%"></i></div><span class="small num" style="text-align:right">${fmtDuration(v)}</span></div>`).join('')}</div>`;
}

function deviceBadge(progress) {
  const device = progress.device || '';
  if (device) {
    const gpu = /cuda|mps|gpu|nvenc|metal/i.test(device);
    return `<span class="badge ${gpu ? 'ok' : 'warn'}">${icon(gpu ? 'gpu' : 'cpu', 'sm')}${esc(device)}</span>`;
  }
  const g = st.gpu;
  if (!g) return '';
  if (g.ready) return `<span class="badge ok">${icon('gpu', 'sm')}${esc(g.torch?.devices?.[0] || 'CUDA GPU')}</span>`;
  return `<span class="badge warn" title="${esc(g.recommendation || '')}">${icon('cpu', 'sm')}No CUDA GPU detected</span>`;
}

async function renderDetail(tok) {
  const box = $('#jobdetail');
  if (!box || !st.selected) return;
  let progress;
  let logs;
  try {
    [progress, logs] = await Promise.all([
      get(`/studio/jobs/${st.selected}/progress`),
      get(`/jobs/${st.selected}/logs?limit=300${st.level ? `&level=${st.level}` : ''}`),
    ]);
  } catch (error) {
    if (isCurrent(tok) && $('#jobdetail')) box.innerHTML = `<div class="panel">${emptyState('alert', 'Job not found', esc(error.message))}</div>`;
    return;
  }
  if (!isCurrent(tok) || !$('#jobdetail')) return;
  const job = st.items.find((j) => j.job_id === st.selected) || { job_id: st.selected, status: progress.status, stage: progress.job_stage, config: {} };
  const active = ACTIVE.includes(progress.status);
  const pct = Math.round(100 * Math.max(+progress.progress || 0, +progress.job_progress || 0));
  const logScroll = $('#logbox')?.scrollTop ?? 0;
  box.innerHTML = `<div class="panel">
      <div class="charthead"><div><div class="eyebrow">Job</div><h2 style="margin-top:2px">${esc(job.match_name || 'Processing job')}</h2>
        <div class="faint xs mono">${esc(st.selected)}</div></div>
        <div class="btnrow">${statusBadge(progress.status)}${deviceBadge(progress)}</div></div>
      ${pipeline(progress, { ...job, status: progress.status, stage: progress.job_stage })}
      <div class="kpis" style="grid-template-columns:repeat(4,minmax(0,1fr));margin:0">
        <div class="kpi"><div class="l">Overall</div><div class="v">${pct}<small>%</small></div><div class="bar" style="margin-top:6px"><i style="width:${pct}%"></i></div></div>
        <div class="kpi"><div class="l">${active ? 'Time left' : 'Took'}</div><div class="v">${active ? (progress.eta_s != null ? fmtDuration(progress.eta_s) : '—') : (progress.elapsed_s != null ? fmtDuration(progress.elapsed_s) : '—')}</div>
          <div class="d">${active && progress.elapsed_s != null ? `${fmtDuration(progress.elapsed_s)} elapsed` : ''}</div></div>
        <div class="kpi"><div class="l">Speed</div><div class="v">${progress.fps_processing ? `${Math.round(progress.fps_processing)}<small>fps</small>` : '—'}</div></div>
        <div class="kpi"><div class="l">Stage</div><div class="v" style="font-size:16px">${esc(titleCase(progress.stage || progress.job_stage || '—'))}</div>
          <div class="d">${progress.source === 'progress_file' ? 'live from engine' : 'from job log'}</div></div>
      </div>
      ${progress.message ? `<div class="note" style="margin-top:12px">${esc(progress.message)}</div>` : ''}
      ${progress.error_message ? `<div class="errnote">${icon('alert', 'sm')}<span>${esc(progress.error_message)}</span></div>` : ''}
      <div class="divider"></div>
      <div class="btnrow">
        ${progress.run_id && !active ? `<a class="btn" href="#matches/${esc(progress.match_id)}?run=${encodeURIComponent(progress.run_id)}">${icon('film', 'sm')} Review results</a>` : ''}
        <a class="btn2" href="#matches/${esc(progress.match_id)}">Open match</a>
        ${active ? `<button class="btn2 danger" id="j_cancel" ${progress.status === 'cancel_requested' ? 'disabled' : ''}>${icon('stop', 'sm')} Cancel</button>` : ''}
        <button class="btn2" id="j_rerun">${icon('rerun', 'sm')} Rerun</button>
        ${!['running', 'claimed'].includes(progress.status) ? `<button class="btn-ghost danger" id="j_delete">${icon('trash', 'sm')} Delete</button>` : ''}
        <button class="btn-ghost" id="j_notify">${icon('mail', 'sm')} Notifications</button>
      </div><div id="j_notes"></div></div>
    ${!active ? timingsChart(progress.timings) : ''}
    <div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Log</div>
      <div class="seg" role="group" aria-label="Log level">${[['', 'All'], ['info', 'Info'], ['warning', 'Warnings'], ['error', 'Errors'], ['debug', 'Debug']].map(([k, l]) =>
        `<button data-lv="${k}" aria-pressed="${st.level === k}">${l}</button>`).join('')}</div></div>
      <div class="log" id="logbox">${logLines(logs.items || [])}</div></div>`;
  $('#logbox').scrollTop = logScroll;
  bindDetail(progress);
}

function logLines(items) {
  if (!items.length) return '<span class="faint">No entries for this filter.</span>';
  return items.slice().reverse().map((entry) => {
    const d = entry.data || {};
    const extra = [d.sub_stage, d.device,
      d.processed_tracking_frames != null ? `frames ${d.processed_tracking_frames}/${d.estimated_tracking_frames}` : null,
      d.written_frames != null ? `rendered ${d.written_frames}/${d.total_frames}` : null,
      d.progress != null ? `${Math.round(100 * d.progress)}%` : null].filter(Boolean).join(' · ');
    return `<div class="lv-${esc(entry.level)}"><span class="ts">${esc((entry.created_at || '').slice(11, 19))}</span> ${esc((entry.level || '').toUpperCase().padEnd(7))} ${esc(entry.stage || '')}: ${esc(entry.message || '')}${extra ? ` <span class="faint">(${esc(extra)})</span>` : ''}</div>`;
  }).join('');
}

function bindDetail(progress) {
  const tok = routeToken();
  $$('[data-lv]').forEach((b) => { b.onclick = () => { st.level = b.dataset.lv; renderDetail(tok); }; });
  const cancel = $('#j_cancel');
  if (cancel) {
    cancel.onclick = async () => {
      if (!await confirmDialog({ title: 'Cancel this job?', body: 'Processing stops at the next checkpoint. Partial results are kept.', confirmLabel: 'Cancel job', danger: true })) return;
      try { await post(`/jobs/${st.selected}/cancel`, {}); toast('Cancel requested', 'warn'); refresh(); } catch (error) { toast(error.message, 'err'); }
    };
  }
  $('#j_rerun').onclick = async () => {
    try { const job = await post(`/jobs/${st.selected}/rerun`, { config_overrides: {}, reason: 'jobs_page_rerun' }); toast('Rerun queued'); go('jobs', job.job_id); } catch (error) { toast(error.message, 'err'); }
  };
  const remove = $('#j_delete');
  if (remove) {
    remove.onclick = async () => {
      if (!await confirmDialog({ title: 'Delete this job?', body: 'Removes the job, its logs and events from the database. Output files on disk are kept.', confirmLabel: 'Delete', danger: true })) return;
      try { await del(`/jobs/${st.selected}`); toast('Job deleted'); go('jobs'); } catch (error) { toast(error.message, 'err'); }
    };
  }
  $('#j_notify').onclick = async () => {
    const box = $('#j_notes');
    try {
      const { items } = await get(`/jobs/${st.selected}/notifications`);
      box.innerHTML = items.length ? `<div class="tablewrap" style="margin-top:12px"><table><thead><tr><th>When</th><th>To</th><th>Subject</th><th>Status</th><th>Backend</th></tr></thead>
        <tbody>${items.map((item) => `<tr><td>${esc(fmtDate(item.created_at))}</td><td>${esc(item.recipient || '—')}</td><td>${esc(item.subject)}</td>
          <td><span class="badge ${item.status === 'sent' ? 'ok' : item.status === 'failed' ? 'danger' : ''}">${esc(item.status)}</span>${item.error_message ? `<div class="note">${esc(item.error_message)}</div>` : ''}</td>
          <td>${esc(item.backend)}</td></tr>`).join('')}</tbody></table></div>`
        : '<div class="note">No notifications recorded — they are sent when the job finishes.</div>';
    } catch (error) { box.innerHTML = `<div class="errnote">${esc(error.message)}</div>`; }
  };
  void progress;
}

let refresh = () => {};

export async function renderJobs(jobId) {
  const tok = routeToken();
  st.selected = jobId || null;
  setMain(`<div class="page">
    <div class="pagehead"><div><div class="eyebrow">Studio</div><h1>Jobs</h1><div class="sub" id="jobsub">Processing queue</div></div>
      <div class="actions" id="gpubadge"></div></div>
    <div class="jobs-layout">
      <div class="panel flush"><div class="panel-title">Recent jobs <span class="faint xs">auto-refreshing</span></div><div id="joblist"><div style="padding:16px">${skLines(6)}</div></div></div>
      <div id="jobdetail">${st.selected ? `<div class="panel">${skLines(8)}</div>` : `<div class="panel">${emptyState('jobs', 'Select a job', 'Pick a job on the left to see its pipeline, ETA and logs.')}</div>`}</div>
    </div></div>`);
  get('/health/gpu').then((gpu) => {
    st.gpu = gpu;
    if (isCurrent(tok) && $('#gpubadge')) $('#gpubadge').innerHTML = deviceBadge({});
  }).catch(() => {});

  refresh = async () => {
    try {
      const data = await get('/studio/jobs?limit=60');
      if (!isCurrent(tok)) return;
      st.items = data.items || [];
      if (!st.selected && st.items.length && window.innerWidth > 900) st.selected = st.items[0].job_id;
      renderList();
      await renderDetail(tok);
    } catch (error) {
      if (isCurrent(tok) && $('#joblist')) $('#joblist').innerHTML = `<div style="padding:16px">${emptyState('alert', 'Could not load jobs', esc(error.message))}</div>`;
    }
  };
  await refresh();
  // One list call + one progress call + one log call per tick; faster while active.
  let ticks = 0;
  every(2500, () => {
    ticks += 1;
    const anyActive = st.items.some((j) => ACTIVE.includes(j.status));
    if (anyActive || ticks % 6 === 0) refresh();
  });
}
