// Library (match-centric) and the match review shell. Each review tab lives
// in views/match/*.js and receives a shared context object.

import { del, get, post, runPosterUrl, runThumbUrl } from '../api.js';
import { icon } from '../icons.js';
import { go, parseRoute, setQuery } from '../router.js';
import { buildModel } from '../runmodel.js';
import {
  $, $$, confirmDialog, emptyState, esc, fmtDay, fmtDuration, fmtRel, isCurrent, onLeave, routeToken, setMain, skBlock,
  skCards, statusBadge, toast,
} from '../ui.js';
import { renderReportTab } from './match/report.js';
import { renderReelTab } from './match/reel.js';
import { renderRosterTab } from './match/roster.js';
import { renderRunsTab } from './match/runs.js';
import { renderStatsTab } from './match/stats.js';
import { renderWatchTab } from './match/watch.js';

/* =============================== Library =============================== */

const libState = { q: '', filter: 'all', sort: 'newest', data: null };

function matchStatus(item) {
  const status = item.latest_job?.status;
  if (!status) return 'none';
  if (['queued', 'claimed', 'running', 'cancel_requested'].includes(status)) return 'processing';
  if (status === 'completed' || item.run) return 'ready';
  if (status === 'failed') return 'failed';
  return status;
}

function scoreline(names, colors, score) {
  const s = score ? [score['0'] ?? '–', score['1'] ?? '–'] : null;
  return `<div class="scoreline">
    <div class="t"><span class="swatch" style="background:${esc(colors[0] || '#e53935')}"></span><span>${esc(names[0])}</span></div>
    <div class="sc">${s ? `${esc(s[0])}&thinsp;–&thinsp;${esc(s[1])}` : 'vs'}</div>
    <div class="t r"><span>${esc(names[1])}</span><span class="swatch" style="background:${esc(colors[1] || '#29b6f6')}"></span></div></div>`;
}

function thumbStyle(run) {
  if (!run) return '';
  const url = run.thumb ? runThumbUrl(run.run_id, run.thumb) : runPosterUrl(run.run_id);
  return `style="background-image:url('${esc(url)}')"`;
}

function matchCard(item) {
  const run = item.run;
  const names = run?.team_names || [item.home_team_name || 'Home', item.away_team_name || 'Away'];
  const colors = run?.team_colors || [];
  const counts = run?.event_counts || {};
  const events = Object.values(counts).reduce((a, b) => a + b, 0);
  const job = item.latest_job;
  const live = job && ['queued', 'claimed', 'running', 'cancel_requested'].includes(job.status);
  return `<article class="mcard" tabindex="0" data-open="${esc(item.match_id)}" aria-label="${esc(item.name || 'Match')}">
    <div class="thumb" ${thumbStyle(run)}>
      ${run ? '' : `<div class="ph">${icon('pitch', 'lg')}</div>`}
      <div class="st">${job ? statusBadge(job.status) : '<span class="status">No runs</span>'}</div>
      ${run?.duration_s ? `<div class="dur">${fmtDuration(run.duration_s)}</div>` : ''}
    </div>
    <div class="body">
      <h3>${esc(item.name || 'Untitled match')}</h3>
      <div class="meta">${item.match_date ? `<span>${esc(fmtDay(item.match_date))}</span>` : ''}
        <span>${item.job_count} run${item.job_count === 1 ? '' : 's'}</span>
        ${job ? `<span>updated ${esc(fmtRel(job.created_at))}</span>` : ''}</div>
      ${scoreline(names, colors, run?.score)}
      ${live ? `<div class="bar indeterminate" aria-label="processing"><i style="width:${Math.round((job.progress || 0) * 100)}%"></i></div>` : ''}
      <div class="foot">
        <div class="chips">
          ${counts.goal ? `<span class="chip">${icon('ball', 'sm')}${counts.goal} goal${counts.goal > 1 ? 's' : ''}</span>` : ''}
          ${events ? `<span class="chip">${events} events</span>` : ''}
          ${run?.tracks_available ? `<span class="chip" title="Player tracks available">${icon('users', 'sm')}tracked</span>` : ''}
          ${job?.status === 'failed' ? `<span class="chip" title="${esc(job.error_message || '')}">${icon('alert', 'sm')}failed</span>` : ''}
        </div>
        <div class="btnrow">
          ${job ? `<button class="iconbtn" data-rerun="${esc(job.job_id)}" title="Rerun latest job" aria-label="Rerun">${icon('rerun', 'sm')}</button>
          <button class="iconbtn" data-deljob="${esc(job.job_id)}" title="Delete latest run" aria-label="Delete latest run">${icon('trash', 'sm')}</button>` : ''}
        </div>
      </div>
    </div></article>`;
}

function runCard(run) {
  return `<article class="mcard" tabindex="0" data-run="${esc(run.run_id)}">
    <div class="thumb" ${thumbStyle(run)}><div class="st"><span class="status">Unlinked run</span></div>
      ${run.duration_s ? `<div class="dur">${fmtDuration(run.duration_s)}</div>` : ''}</div>
    <div class="body"><h3 class="mono" style="font-size:13px">${esc(run.run_id)}</h3>
      <div class="meta"><span>${esc(fmtRel(run.modified_at || run.generated_at))}</span></div>
      ${scoreline(run.team_names || ['Home', 'Away'], run.team_colors || [], run.score)}</div></article>`;
}

function renderLibraryGrid() {
  const data = libState.data;
  const q = libState.q.toLowerCase();
  let items = data.matches.filter((item) => {
    const hay = [item.name, item.home_team_name, item.away_team_name, item.match_date, ...(item.run?.team_names || [])].join(' ').toLowerCase();
    if (q && !hay.includes(q)) return false;
    if (libState.filter !== 'all' && matchStatus(item) !== libState.filter) return false;
    return true;
  });
  const key = { newest: (m) => m.latest_job?.created_at || m.created_at, date: (m) => m.match_date || '', name: (m) => (m.name || '').toLowerCase() }[libState.sort];
  items = items.sort((a, b) => (libState.sort === 'name' ? key(a).localeCompare(key(b)) : String(key(b)).localeCompare(String(key(a)))));
  const grid = $('#libgrid');
  if (!grid) return;
  if (!data.matches.length) {
    grid.outerHTML = `<div id="libgrid">${emptyState('film', 'No matches yet',
      'Upload a recording or point at a file on the server. Processing builds the game camera, highlights and stats.',
      '<a class="btn" href="#create">' + icon('upload') + ' Start a new run</a>')}</div>`;
  } else {
    grid.className = items.length ? 'grid' : '';
    grid.innerHTML = items.length ? items.map(matchCard).join('')
      : emptyState('search', 'Nothing matches', 'Try a different search or filter.');
  }
  const unlinked = $('#unlinked');
  if (unlinked) {
    const runs = data.unlinked_runs.filter((run) => !q || run.run_id.toLowerCase().includes(q));
    unlinked.innerHTML = runs.length ? `<div class="pagehead" style="margin:32px 0 16px"><div>
        <div class="eyebrow">Output folder</div><h2>Runs without a match</h2>
        <div class="sub">Found in ${esc(data.output_root)} — open to review; they are not linked to a match in the database.</div></div></div>
      <div class="grid">${runs.map(runCard).join('')}</div>` : '';
  }
  bindLibraryActions();
}

function bindLibraryActions() {
  $$('[data-open]').forEach((card) => {
    const open = () => go('matches', card.dataset.open);
    card.onclick = (event) => { if (!event.target.closest('button')) open(); };
    card.onkeydown = (event) => { if (event.key === 'Enter') open(); };
  });
  $$('[data-run]').forEach((card) => {
    card.onclick = () => go('runs', card.dataset.run);
    card.onkeydown = (event) => { if (event.key === 'Enter') go('runs', card.dataset.run); };
  });
  $$('[data-rerun]').forEach((button) => {
    button.onclick = async () => {
      if (!await confirmDialog({ title: 'Rerun this match?', body: 'A new job is queued with the same settings. Existing results stay available.', confirmLabel: 'Rerun' })) return;
      try {
        const job = await post(`/jobs/${button.dataset.rerun}/rerun`, { config_overrides: {}, reason: 'library_rerun' });
        toast('Rerun queued');
        go('jobs', job.job_id);
      } catch (error) { toast(error.message, 'err'); }
    };
  });
  $$('[data-deljob]').forEach((button) => {
    button.onclick = async () => {
      if (!await confirmDialog({ title: 'Delete the latest run?', body: 'The job, its logs and its events are removed from the database. Files on disk are kept.', confirmLabel: 'Delete', danger: true })) return;
      try {
        await del(`/jobs/${button.dataset.deljob}`);
        toast('Run deleted');
        renderMatches();
      } catch (error) { toast(error.message, 'err'); }
    };
  });
}

export async function renderMatches() {
  const tok = routeToken();
  const { page } = parseRoute();
  setMain(`<div class="page">
    <div class="pagehead"><div><div class="eyebrow">Studio</div><h1>Library</h1><div class="sub" id="libsub">Every match you have processed</div></div>
      <div class="actions"><a class="btn" href="#create">${icon('plus')} New run</a></div></div>
    <div class="toolbar">
      <div class="search">${icon('search', 'sm')}<input type="search" id="libq" placeholder="Search matches or teams" aria-label="Search" value="${esc(libState.q)}"></div>
      <div class="seg" role="group" aria-label="Filter by status">
        ${[['all', 'All'], ['ready', 'Ready'], ['processing', 'Processing'], ['failed', 'Failed'], ['none', 'Not run']].map(([k, l]) =>
          `<button data-f="${k}" class="${libState.filter === k ? 'active' : ''}">${l}</button>`).join('')}
      </div>
      <select id="libsort" aria-label="Sort" style="width:auto">
        <option value="newest">Recently updated</option><option value="date">Match date</option><option value="name">Name</option></select>
    </div>
    <div class="grid" id="libgrid">${skCards(6)}</div>
    <div id="unlinked"></div></div>`);
  $('#libsort').value = libState.sort;
  $('#libq').oninput = (event) => { libState.q = event.target.value; if (libState.data) renderLibraryGrid(); };
  $('#libsort').onchange = (event) => { libState.sort = event.target.value; if (libState.data) renderLibraryGrid(); };
  $$('.seg [data-f]').forEach((button) => {
    button.onclick = () => {
      libState.filter = button.dataset.f;
      $$('.seg [data-f]').forEach((b) => b.classList.toggle('active', b === button));
      if (libState.data) renderLibraryGrid();
    };
  });
  try {
    const data = await get('/studio/library');
    if (!isCurrent(tok)) return;
    libState.data = data;
    const ready = data.matches.filter((m) => m.run).length;
    $('#libsub').textContent = `${data.matches.length} match${data.matches.length === 1 ? '' : 'es'} · ${ready} with results`;
    renderLibraryGrid();
    if (page === 'runs') $('#unlinked')?.scrollIntoView({ behavior: 'smooth' });
  } catch (error) {
    if (!isCurrent(tok)) return;
    $('#libgrid').outerHTML = `<div id="libgrid">${emptyState('alert', 'Could not load the library', esc(error.message))}</div>`;
  }
}

/* ============================ Match review ============================ */

const TABS = [
  ['watch', 'Watch', 'film'],
  ['stats', 'Stats', 'chart'],
  ['reel', 'Reel builder', 'scissors'],
  ['report', 'Report', 'report'],
  ['roster', 'Roster & sharing', 'users'],
  ['runs', 'Runs', 'jobs'],
];
const TAB_RENDER = {
  watch: renderWatchTab, stats: renderStatsTab, reel: renderReelTab, report: renderReportTab, roster: renderRosterTab, runs: renderRunsTab,
};

function header(ctx) {
  const m = ctx.model;
  const match = ctx.match;
  const title = match?.name || (m ? `Run ${ctx.runId}` : 'Match');
  const names = m ? m.teams.map((t) => t.name) : [match?.home_team_name || 'Home', match?.away_team_name || 'Away'];
  const colors = m ? m.teams.map((t) => t.color) : ['#e53935', '#29b6f6'];
  const score = m?.score;
  const runOptions = ctx.jobs.filter((job) => job.run_id).map((job) =>
    `<option value="${esc(job.run_id)}" ${job.run_id === ctx.runId ? 'selected' : ''}>${esc(fmtRel(job.created_at))} · ${esc(job.config.camera_mode || 'run')}${job.config.focus_track_id != null ? ` #${esc(job.config.focus_track_id)}` : ''} · ${esc(job.status)}</option>`).join('');
  return `<div class="crumbs"><a href="#matches">Library</a>${icon('chevron', 'sm')}<span>${esc(title)}</span></div>
    <div class="matchhead">
      <div><h1>${esc(title)}</h1>
        <div class="sub">${match?.match_date ? esc(fmtDay(match.match_date)) + ' · ' : ''}${m ? `${fmtDuration(m.duration)} processed` : 'No processed run yet'}${m?.trim ? ` · starts at ${fmtDuration(m.trim)} of the source` : ''}</div></div>
      <div class="btnrow">
        <div class="scoreboard" aria-label="Score">
          <div class="tm"><span class="swatch" style="background:${esc(colors[0])}"></span>${esc(names[0])}</div>
          <div class="big">${score ? `${esc(score['0'] ?? 0)} – ${esc(score['1'] ?? 0)}` : '– –'}</div>
          <div class="tm">${esc(names[1])}<span class="swatch" style="background:${esc(colors[1])}"></span></div>
        </div>
        ${runOptions && ctx.jobs.filter((j) => j.run_id).length > 1 ? `<select id="runpick" aria-label="Choose run" style="width:auto;max-width:260px">${runOptions}</select>` : ''}
        ${ctx.matchId ? `<button class="btn2" id="sharematch">${icon('share', 'sm')} Share</button>` : ''}
      </div>
    </div>`;
}

function renderShell(ctx) {
  ctx.switchTab = (tab, options) => switchTab(ctx, tab, options);
  onLeave(() => { if (ctx.tabCleanup) ctx.tabCleanup(); ctx.tabCleanup = null; });
  const available = TABS.filter(([key]) => (ctx.matchId || !['roster', 'runs'].includes(key)));
  setMain(`<div class="page">${header(ctx)}
    <div class="tabs" role="tablist">${available.map(([key, label, ico]) =>
      `<button role="tab" data-tab="${key}" aria-selected="${ctx.tab === key}" class="${ctx.tab === key ? 'active' : ''}">${icon(ico, 'sm')}${label}</button>`).join('')}</div>
    <div id="tabbody"></div></div>`);
  $$('[data-tab]').forEach((button) => { button.onclick = () => switchTab(ctx, button.dataset.tab); });
  const picker = $('#runpick');
  if (picker) picker.onchange = () => go('matches', ctx.matchId, { run: picker.value, tab: ctx.tab });
  const share = $('#sharematch');
  if (share) share.onclick = () => switchTab(ctx, 'roster', { share: true });
}

export function switchTab(ctx, tab, options = {}) {
  if (ctx.tabCleanup) { try { ctx.tabCleanup(); } catch { /* ignore */ } ctx.tabCleanup = null; }
  ctx.tab = tab;
  setQuery({ tab: tab === 'watch' ? null : tab });
  $$('[data-tab]').forEach((b) => { b.classList.toggle('active', b.dataset.tab === tab); b.setAttribute('aria-selected', String(b.dataset.tab === tab)); });
  const body = $('#tabbody');
  body.innerHTML = '';
  if (!ctx.model && !['roster', 'runs'].includes(tab)) {
    body.innerHTML = noRunState(ctx);
    return;
  }
  const cleanup = TAB_RENDER[tab](body, ctx, options);
  if (typeof cleanup === 'function') ctx.tabCleanup = cleanup;
}

function noRunState(ctx) {
  const latest = ctx.jobs[0];
  if (latest && ['queued', 'claimed', 'running', 'cancel_requested'].includes(latest.status)) {
    return emptyState('clock', 'Processing in progress',
      `The first run is ${esc(latest.status)}${latest.stage ? ` (${esc(latest.stage)})` : ''}. Results appear here when it finishes.`,
      `<a class="btn2" href="#jobs/${esc(latest.job_id)}">${icon('jobs', 'sm')} Follow progress</a>`);
  }
  if (latest?.status === 'failed') {
    return emptyState('alert', 'The last run failed', esc(latest.error_message || 'See the job log for details.'),
      `<a class="btn2" href="#jobs/${esc(latest.job_id)}">Open job</a>`);
  }
  return emptyState('film', 'No results yet', 'Process this match to get the game camera, highlights and stats.',
    `<a class="btn" href="#create">${icon('upload', 'sm')} New run</a>`);
}

export async function loadRun(ctx, runId) {
  ctx.runId = runId;
  ctx.run = await get(`/studio/runs/${encodeURIComponent(runId)}`);
  ctx.model = buildModel(ctx.run);
  ctx.jobId = ctx.run.job_id;
  return ctx;
}

function newContext(matchId) {
  return {
    matchId, match: null, jobs: [], runId: null, run: null, model: null, jobId: null, tab: 'watch', tabCleanup: null,
    reel: null, // ordered event ids, shared by Watch and Reel tabs
  };
}

function loadingShell() {
  setMain(`<div class="page"><div class="sk" style="height:28px;width:320px;margin-bottom:16px"></div>
    <div class="watch"><div>${skBlock(420)}<div style="height:12px"></div>${skBlock(90)}</div><div>${skBlock(520)}</div></div></div>`);
}

export async function renderMatchDetail(matchId, query = new URLSearchParams()) {
  const tok = routeToken();
  loadingShell();
  const ctx = newContext(matchId);
  try {
    const [match, jobs] = await Promise.all([
      get(`/matches/${encodeURIComponent(matchId)}`),
      get(`/studio/jobs?match_id=${encodeURIComponent(matchId)}&limit=100`),
    ]);
    if (!isCurrent(tok)) return;
    ctx.match = match;
    ctx.jobs = jobs.items || [];
    const wanted = query.get('run');
    const pick = (wanted && ctx.jobs.find((job) => job.run_id === wanted))
      || ctx.jobs.find((job) => job.status === 'completed' && job.run_id)
      || ctx.jobs.find((job) => job.run_id && !['queued', 'claimed', 'running', 'cancel_requested'].includes(job.status));
    if (pick) await loadRun(ctx, pick.run_id);
    if (!isCurrent(tok)) return;
  } catch (error) {
    if (!isCurrent(tok)) return;
    setMain(`<div class="page">${emptyState('alert', 'Could not open this match', esc(error.message), '<a class="btn2" href="#matches">Back to library</a>')}</div>`);
    return;
  }
  ctx.tab = query.get('tab') || 'watch';
  renderShell(ctx);
  switchTab(ctx, ctx.tab);
}

// Run without (or before resolving) a match: used by #runs/<id>.
export async function renderRunReview(runId, query = new URLSearchParams()) {
  const tok = routeToken();
  loadingShell();
  const ctx = newContext(null);
  try {
    await loadRun(ctx, runId);
  } catch (error) {
    if (!isCurrent(tok)) return;
    setMain(`<div class="page">${emptyState('alert', 'Run not found', esc(error.message), '<a class="btn2" href="#matches">Back to library</a>')}</div>`);
    return;
  }
  if (!isCurrent(tok)) return;
  if (ctx.run.match_id) {
    go('matches', ctx.run.match_id, { run: runId, ...(query.get('tab') ? { tab: query.get('tab') } : {}) });
    return;
  }
  ctx.tab = query.get('tab') || 'watch';
  renderShell(ctx);
  switchTab(ctx, ctx.tab);
}
