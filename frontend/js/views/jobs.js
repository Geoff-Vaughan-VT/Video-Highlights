// Jobs queue: stage-level progress, elapsed-vs-SLA messaging, live logs,
// and delivery state of completion notifications.

import { get } from '../api.js';
import { $, $$, esc, fmtDate, fmtHours, setMain } from '../ui.js';

let timer = null;
let logsFor = null;
let policy = null;

export async function renderJobs() {
  clearInterval(timer);
  logsFor = null;
  setMain(`
    <h1>Jobs</h1>
    <div class="sub" id="jobsub">Processing queue — auto-refreshes while running</div>
    <div id="jl">Loading…</div>
    <div id="jlog"></div>`);
  try { policy = policy || await get('/matches/upload-policy'); } catch { policy = null; }
  if (policy?.processing_sla_hours?.length === 2) {
    $('#jobsub').textContent =
      `Processing queue — most matches finish in ${policy.processing_sla_hours[0]}–${policy.processing_sla_hours[1]} hours`;
  }
  await load();
  timer = setInterval(() => { load(); refreshLogs(); }, 5000);
}

function slaBadge(job) {
  const sla = policy?.processing_sla_hours;
  const started = job.started_at || job.created_at;
  if (['queued', 'claimed', 'running'].includes(job.status) && started) {
    const elapsed = Date.now() - new Date(started).getTime();
    const target = sla?.length === 2 ? ` / target ${sla[0]}–${sla[1]} h` : '';
    return `<small style="color:var(--dim)">elapsed ${fmtHours(elapsed)}${target}</small>`;
  }
  if (job.completed_at && started) {
    const took = new Date(job.completed_at).getTime() - new Date(started).getTime();
    return `<small style="color:var(--dim)">took ${fmtHours(took)}</small>`;
  }
  return '';
}

async function load() {
  // The refresh timer can outlive this view; stop when the table is gone.
  if (!$('#jl')) { clearInterval(timer); return; }
  try {
    const matches = await get('/matches?limit=50');
    const rows = [];
    for (const match of matches.items || []) {
      const jobs = await get(`/matches/${match.match_id}/jobs`);
      for (const job of jobs.items || []) rows.push({ match: match.name, matchId: match.match_id, ...job });
    }
    rows.sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')));
    const active = rows.some((row) => ['queued', 'claimed', 'running'].includes(row.status));
    const list = $('#jl');
    if (!list) { clearInterval(timer); return; } // navigated away mid-fetch
    list.innerHTML = rows.length ? `<div class="tablewrap"><table>
      <thead><tr><th>match</th><th>job</th><th>status</th><th>stage</th><th style="width:200px">progress</th><th>turnaround</th><th></th></tr></thead>
      <tbody>${rows.map((row) => `
        <tr>
          <td><a href="#matches/${encodeURIComponent(row.matchId)}">${esc(row.match || row.matchId)}</a></td>
          <td style="font-family:monospace;font-size:11px" title="created ${esc(fmtDate(row.created_at))}">${esc(row.job_id)}</td>
          <td><span class="status ${esc(row.status)}">${esc(row.status)}</span></td>
          <td>${esc(row.stage || '')}</td>
          <td><div class="bar"><i style="width:${Math.round(100 * (+row.progress || 0))}%"></i></div>
              <small style="color:var(--dim)">${Math.round(100 * (+row.progress || 0))}%</small></td>
          <td>${slaBadge(row)}</td>
          <td style="white-space:nowrap">
            ${row.status === 'completed' ? `<a class="btn2" href="#runs/${encodeURIComponent(row.job_id)}">Review</a> ` : ''}
            <button class="btn2" data-logs="${esc(row.job_id)}">Logs</button>
            <button class="btn2" data-notify="${esc(row.job_id)}" title="Notification delivery">✉</button>
          </td>
        </tr>`).join('')}
      </tbody></table></div>` : '<div class="empty">No jobs yet.</div>';
    $$('button[data-logs]').forEach((button) => { button.onclick = () => { logsFor = button.dataset.logs; refreshLogs(); }; });
    $$('button[data-notify]').forEach((button) => { button.onclick = () => showNotifications(button.dataset.notify); });
    if (!active) clearInterval(timer);
  } catch (error) {
    const list = $('#jl');
    if (list) list.innerHTML = `<div class="empty">${esc(error.message)}</div>`;
    clearInterval(timer);
  }
}

async function showNotifications(jobId) {
  logsFor = null;
  let html;
  try {
    const { items } = await get(`/jobs/${jobId}/notifications`);
    html = items.length ? `<div class="tablewrap"><table>
      <thead><tr><th>when</th><th>to</th><th>subject</th><th>status</th><th>backend</th></tr></thead>
      <tbody>${items.map((item) => `
        <tr><td>${esc(fmtDate(item.created_at))}</td><td>${esc(item.recipient || '—')}</td>
        <td>${esc(item.subject)}</td>
        <td><span class="status ${item.status === 'sent' ? 'completed' : (item.status === 'failed' ? 'failed' : 'queued')}">${esc(item.status)}</span>
            ${item.error_message ? `<div class="note">${esc(item.error_message)}</div>` : ''}</td>
        <td>${esc(item.backend)}</td></tr>`).join('')}
      </tbody></table></div>` : '<div class="note">No notifications recorded — they are sent when the job finishes.</div>';
  } catch (error) {
    html = `<div class="errnote">${esc(error.message)}</div>`;
  }
  $('#jlog').innerHTML = `<div class="panel" style="margin-top:16px">
    <h3>Notifications — ${esc(jobId)} <button class="btn2" style="float:right" id="closepane">close</button></h3>${html}</div>`;
  $('#closepane').onclick = () => { $('#jlog').innerHTML = ''; };
}

async function refreshLogs() {
  if (!logsFor || !$('#jlog')) return;
  let data;
  try { data = await get(`/jobs/${logsFor}/logs?limit=400`); } catch (error) { data = { items: [], error: error.message }; }
  const rows = data.items || [];
  rows.sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')));
  const lines = rows.map((entry) => {
    const detail = entry.data || {};
    const extra = [detail.sub_stage, detail.device,
      detail.processed_tracking_frames != null ? `frames ${detail.processed_tracking_frames}/${detail.estimated_tracking_frames}` : null,
      detail.written_frames != null ? `rendered ${detail.written_frames}/${detail.total_frames}` : null,
      detail.progress != null ? `${Math.round(100 * detail.progress)}%` : null,
    ].filter(Boolean).join(' | ');
    return `${(entry.created_at || '').slice(11, 19)} [${entry.level || ''}] ${entry.stage || ''}: ${entry.message || ''}${extra ? '  (' + extra + ')' : ''}`;
  }).join('\n');
  $('#jlog').innerHTML = `<div class="panel" style="margin-top:16px">
    <h3>Live workflow log — ${esc(logsFor)} <button class="btn2" style="float:right" id="closepane">close</button></h3>
    <pre>${esc(data.error || lines || 'no entries yet')}</pre></div>`;
  $('#closepane').onclick = () => { logsFor = null; $('#jlog').innerHTML = ''; };
}
