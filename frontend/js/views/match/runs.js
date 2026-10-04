// Runs tab: every processing job for this match with its settings, and
// actions to open, rerun, follow or delete it.

import { del, post } from '../../api.js';
import { icon } from '../../icons.js';
import { go } from '../../router.js';
import { confirmDialog, emptyState, esc, fmtRel, statusBadge, toast } from '../../ui.js';

function describe(config) {
  const parts = [];
  if (config.profile) parts.push(config.profile);
  if (config.camera_mode) parts.push(config.camera_mode.replace(/_/g, ' '));
  if (config.focus_track_id != null) parts.push(`player ${config.focus_track_id}`);
  if (config.reuse_tracking_from_job) parts.push('reused tracking');
  if (config.player_spotlight_reel) parts.push('spotlight');
  if (config.trim_start || config.trim_end) parts.push(`${config.trim_start || '0:00'}–${config.trim_end || 'end'}`);
  return parts.join(' · ') || 'default settings';
}

export function renderRunsTab(body, ctx) {
  if (!ctx.jobs.length) {
    body.innerHTML = emptyState('jobs', 'No runs yet', 'Start one from New run.', `<a class="btn" href="#create">${icon('plus', 'sm')} New run</a>`);
    return;
  }
  body.innerHTML = `<div class="panel flush"><div class="panel-title">Processing runs <span class="faint xs">${ctx.jobs.length}</span></div>
    ${ctx.jobs.map((job) => `<div class="jobrow ${job.run_id && job.run_id === ctx.runId ? 'sel' : ''}" data-job="${esc(job.job_id)}">
      <div style="min-width:0"><div style="font-weight:600">${esc(describe(job.config || {}))}</div>
        <div class="faint xs mono">${esc(job.job_id)}</div></div>
      <div>${statusBadge(job.status)}</div>
      <div><div class="bar ${job.status === 'failed' ? 'danger' : job.status === 'completed' ? 'ok' : ''}"><i style="width:${Math.round(100 * (job.progress || 0))}%"></i></div>
        <div class="faint xs" style="margin-top:4px">${esc(job.stage || '')}${job.error_message ? ` · ${esc(job.error_message)}` : ''}</div></div>
      <div class="faint xs">${esc(fmtRel(job.created_at))}</div>
      <div class="btnrow" style="gap:2px">
        ${job.run_id && job.run_id !== ctx.runId && !['queued', 'claimed', 'running'].includes(job.status) ? `<button class="btn2 sm" data-open="${esc(job.run_id)}">Review</button>` : ''}
        ${job.run_id === ctx.runId ? '<span class="badge accent">Viewing</span>' : ''}
        <button class="iconbtn" data-rerun="${esc(job.job_id)}" title="Rerun" aria-label="Rerun">${icon('rerun', 'sm')}</button>
        <button class="iconbtn" data-del="${esc(job.job_id)}" title="Delete" aria-label="Delete">${icon('trash', 'sm')}</button>
      </div></div>`).join('')}</div>`;
  body.onclick = async (event) => {
    const open = event.target.closest('[data-open]');
    if (open) { go('matches', ctx.matchId, { run: open.dataset.open }); return; }
    const rerun = event.target.closest('[data-rerun]');
    if (rerun) {
      try { const job = await post(`/jobs/${rerun.dataset.rerun}/rerun`, { config_overrides: {}, reason: 'studio_rerun' }); toast('Rerun queued'); go('jobs', job.job_id); } catch (error) { toast(error.message, 'err'); }
      return;
    }
    const remove = event.target.closest('[data-del]');
    if (remove) {
      if (!await confirmDialog({ title: 'Delete this run?', body: 'Removes the job, its logs and events from the database. Files on disk are kept.', confirmLabel: 'Delete', danger: true })) return;
      try { await del(`/jobs/${remove.dataset.del}`); toast('Run deleted'); go('matches', ctx.matchId); } catch (error) { toast(error.message, 'err'); }
      return;
    }
    const row = event.target.closest('[data-job]');
    if (row && !event.target.closest('button')) go('jobs', row.dataset.job);
  };
}
