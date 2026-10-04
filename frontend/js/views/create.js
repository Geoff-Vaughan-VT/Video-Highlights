// New run wizard: 1 Match & video (upload with drag-drop + progress, link,
// or server path) → 2 Teams (names, kit colours, auto-detect) → 3 Processing
// (profile with runtime estimates, camera style, output, window trim,
// advanced) → 4 Review & start. Validates against the upload policy.

import { get, post, uploadFile } from '../api.js';
import { icon } from '../icons.js';
import { go } from '../router.js';
import { $, $$, clamp, esc, fmtBytes, fmtClock, fmtDuration, isCurrent, routeToken, setMain, toggleHtml, toast } from '../ui.js';

const KIT_PRESETS = ['#e53935', '#1e88e5', '#fdd835', '#43a047', '#fb8c00', '#8e24aa', '#ffffff', '#212121', '#00acc1', '#d81b60'];

const PROFILES = {
  fast: { label: 'Fast', blurb: '720p analysis, every other frame. Great for a quick look or a test window; small players and long balls are found less often.', minutes: { gpu: 20, spark: 20, apple: 45, cpu: 360 } },
  balanced: { label: 'Balanced', blurb: '1080p analysis at full frame rate with a mid-size detector. The right choice for most matches.', minutes: { gpu: 35, spark: 35, apple: 80, cpu: 600 } },
  quality: { label: 'Quality', blurb: 'Largest detector, high-res ball pass and 1440p output. Best stats and camera; slowest.', minutes: { gpu: 60, spark: 55, apple: 150, cpu: 1200 } },
};
const HARDWARE = [['gpu', 'RTX 4080-class GPU'], ['spark', 'DGX Spark'], ['apple', 'Apple Studio (M-series)'], ['cpu', 'CPU only']];
const CAMERA_STYLES = {
  broadcast: { label: 'Broadcast', blurb: 'Follows the ball and frames the players in the play, like a TV camera.', zoom: 1.6 },
  tight: { label: 'Tight', blurb: 'Closer on the ball. More detail, less context.', zoom: 2.0 },
  wide: { label: 'Wide', blurb: 'Gentle zoom that keeps most of the pitch in view.', zoom: 1.25 },
};

const blank = () => ({
  step: 1, policy: null, sources: [], hwClass: 'gpu',
  details: { name: '', date: '', email: '' },
  srcMode: 'upload', file: null, fileMeta: {}, localPath: '', localInfo: null, link: '', linkSource: null,
  teams: { home: 'Home', away: 'Away', homeColor: '#e53935', awayColor: '#1e88e5', auto: true },
  proc: { profile: 'balanced', style: 'broadcast', height: 1080, trimStart: 0, trimEnd: null, fullMovie: true, reel: true,
    cards: true, llm: false, scorebug: true, debug: false, model: '', imgsz: '', stride: '' },
});

let st = blank();

export async function renderCreate() {
  const tok = routeToken();
  st = blank();
  setMain(`<div class="page">
    <div class="pagehead"><div><div class="eyebrow">Studio</div><h1>New run</h1>
      <div class="sub">Upload a match and get a game camera, highlights and stats. Sensible defaults — change only what you need.</div></div></div>
    <div class="steps" id="steps"></div>
    <div class="wizard"><div id="wizard"></div><aside id="wsummary"></aside></div></div>`);
  const [policy, sources, gpu] = await Promise.all([
    get('/matches/upload-policy').catch(() => null),
    get('/sources').then((r) => r.sources).catch(() => []),
    get('/health/gpu').catch(() => null),
  ]);
  if (!isCurrent(tok)) return;
  st.policy = policy;
  st.sources = sources;
  if (gpu && !gpu.ready) st.hwClass = /mac/i.test(navigator.userAgent) ? 'apple' : 'cpu';
  renderStep();
}

const sourceDuration = () => st.fileMeta.duration || st.localInfo?.ffprobe?.duration_seconds || null;
const windowLength = () => {
  const d = sourceDuration();
  if (!d) return null;
  return Math.max(0, (st.proc.trimEnd ?? d) - st.proc.trimStart);
};

function slaText() {
  const sla = st.policy?.processing_sla_hours;
  return sla?.length === 2 ? `Most matches finish within ${sla[0]}–${sla[1]} hours of upload on the hosted service.` : '';
}

function stepsBar() {
  const labels = ['Match & video', 'Teams', 'Processing', 'Review'];
  $('#steps').innerHTML = labels.map((label, i) => {
    const n = i + 1;
    const cls = n === st.step ? 'active' : n < st.step ? 'done' : '';
    return `<div class="step ${cls}"><span class="n">${n < st.step ? icon('check', 'sm') : n}</span>${label}</div>`;
  }).join('');
}

function estimateMinutes(profile, hw) {
  const base = PROFILES[profile].minutes[hw];
  const len = windowLength();
  return len ? Math.max(1, Math.round((base * len) / 5400)) : base;
}

function renderSummary() {
  const d = st.details;
  const len = windowLength();
  const src = st.srcMode === 'upload' ? (st.file ? `${st.file.name} · ${fmtBytes(st.file.size)}` : 'No file yet')
    : st.srcMode === 'link' ? (st.link || 'No link yet') : (st.localPath || 'No path yet');
  $('#wsummary').innerHTML = `<div class="panel summary" style="position:sticky;top:24px">
    <div class="panel-title">Summary</div>
    <dl><dt>Match</dt><dd>${esc(d.name || '—')}</dd>
      <dt>Teams</dt><dd><span class="swatch" style="background:${esc(st.teams.homeColor)}"></span> ${esc(st.teams.home)} · ${esc(st.teams.away)} <span class="swatch" style="background:${esc(st.teams.awayColor)}"></span></dd>
      <dt>Video</dt><dd title="${esc(src)}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px">${esc(src.split(/[\\/]/).pop())}</dd>
      ${sourceDuration() ? `<dt>Length</dt><dd>${fmtDuration(sourceDuration())}</dd>` : ''}
      ${len && len < sourceDuration() - 1 ? `<dt>Window</dt><dd>${fmtClock(st.proc.trimStart)} – ${fmtClock(st.proc.trimEnd ?? sourceDuration())}</dd>` : ''}
      <dt>Profile</dt><dd>${PROFILES[st.proc.profile].label}</dd>
      <dt>Camera</dt><dd>${CAMERA_STYLES[st.proc.style].label} · ${st.proc.height}p</dd>
      <dt>Estimate</dt><dd>~${fmtDuration(estimateMinutes(st.proc.profile, st.hwClass) * 60)}</dd></dl>
    <div class="note">${len ? `For ${fmtDuration(len)} of video` : 'For a 90-minute match'} on ${esc(HARDWARE.find(([k]) => k === st.hwClass)[1])}. ${esc(slaText())}</div></div>`;
}

function renderStep() {
  stepsBar();
  ({ 1: stepVideo, 2: stepTeams, 3: stepProcessing, 4: stepReview })[st.step]();
  renderSummary();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function nav(canNext = true, nextLabel = 'Continue') {
  return `<div class="btnrow" style="margin-top:24px;justify-content:space-between">
    ${st.step > 1 ? `<button class="btn2" id="wback">${icon('left', 'sm')} Back</button>` : '<span></span>'}
    <button class="btn" id="wnext" ${canNext ? '' : 'disabled'}>${nextLabel} ${st.step < 4 ? icon('chevron', 'sm') : ''}</button></div>`;
}

function bindNav(onNext) {
  const back = $('#wback');
  if (back) back.onclick = () => { st.step -= 1; renderStep(); };
  $('#wnext').onclick = onNext;
}

/* ---------------- step 1: match & video ---------------- */

function stepVideo() {
  const policy = st.policy;
  const maxLabel = policy ? `${policy.max_upload_gb} GB` : '3 GB';
  const extensions = (policy?.allowed_extensions || ['.mp4', '.mov', '.mkv', '.avi', '.m4v']).join(', ');
  const linkProviders = (st.sources || []).filter((source) => source.kind === 'link' && source.key !== 'other_link');
  $('#wizard').innerHTML = `
    <div class="panel"><div class="panel-title">Match</div>
      <div class="row"><div><label for="c_name">Match name</label><input type="text" id="c_name" value="${esc(st.details.name)}" placeholder="e.g. U16 League — Harbour vs Eastfield"></div>
        <div><label for="c_date">Date</label><input type="date" id="c_date" value="${esc(st.details.date)}"></div></div>
      <label for="c_email">Email me when it’s ready (optional)</label>
      <input type="email" id="c_email" value="${esc(st.details.email)}" placeholder="coach@example.com"></div>
    <div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Video</div>
      <div class="seg" role="group" aria-label="Video source">
        <button data-mode="upload">${icon('upload', 'sm')} Upload</button>
        <button data-mode="local">${icon('disk', 'sm')} Server path</button>
        ${linkProviders.length ? `<button data-mode="link">${icon('link', 'sm')} Link</button>` : ''}</div></div>
      <div id="m_upload">
        <div class="dropzone" id="drop" tabindex="0" role="button" aria-label="Choose a video file">
          ${icon('upload', 'lg')}<div class="big">Drop the match video here, or click to browse</div>
          <div class="small">${esc(extensions)} · up to ${esc(maxLabel)}${policy && !policy.extended_upload_enabled ? ' (larger uploads available as an add-on)' : ''}</div></div>
        <input type="file" id="c_file" accept="video/*" hidden>
        <div class="filemeta" id="filemeta"></div><div id="filenotes"></div></div>
      <div id="m_local" hidden>
        <label for="c_path">Absolute path on the server / container</label>
        <div class="btnrow"><input type="text" id="c_path" placeholder="/data/videos/match.mp4" value="${esc(st.localPath)}" style="flex:1">
          <button class="btn2" id="c_check">Check file</button></div>
        <div class="note">No upload — fastest for long recordings already on the machine running the worker.</div><div id="pathnotes"></div></div>
      <div id="m_link" hidden>
        <label for="c_link">Public video link</label><input type="text" id="c_link" placeholder="https://www.youtube.com/watch?v=…" value="${esc(st.link)}">
        <div id="linknotes"></div>
        <div class="note">Supported: ${linkProviders.map((s) => esc(s.label)).join(', ')}. Raw file uploads always give the most complete statistics.</div></div>
    </div>${nav(false)}`;

  const setMode = (mode) => {
    st.srcMode = mode;
    $$('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    $('#m_upload').hidden = mode !== 'upload';
    $('#m_local').hidden = mode !== 'local';
    $('#m_link').hidden = mode !== 'link';
    updateNext();
    renderSummary();
  };
  $$('[data-mode]').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });

  const drop = $('#drop');
  drop.onclick = () => $('#c_file').click();
  drop.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('#c_file').click(); } };
  drop.ondragover = (event) => { event.preventDefault(); drop.classList.add('drag'); };
  drop.ondragleave = () => drop.classList.remove('drag');
  drop.ondrop = (event) => { event.preventDefault(); drop.classList.remove('drag'); if (event.dataTransfer.files[0]) acceptFile(event.dataTransfer.files[0]); };
  $('#c_file').onchange = (event) => { if (event.target.files[0]) acceptFile(event.target.files[0]); };
  $('#c_path').oninput = (event) => { st.localPath = event.target.value.trim(); st.localInfo = null; $('#pathnotes').innerHTML = ''; updateNext(); renderSummary(); };
  $('#c_check').onclick = checkPath;
  let linkTimer = null;
  $('#c_link').oninput = (event) => {
    st.link = event.target.value.trim(); st.linkSource = null; updateNext(); renderSummary();
    clearTimeout(linkTimer); linkTimer = setTimeout(classifyLink, 400);
  };
  for (const id of ['c_name', 'c_date', 'c_email']) $(`#${id}`).oninput = () => { readDetails(); renderSummary(); };

  function readDetails() {
    st.details = { name: $('#c_name').value.trim(), date: $('#c_date').value, email: $('#c_email').value.trim() };
  }

  function updateNext() {
    const ok = st.srcMode === 'local' ? !!st.localPath : st.srcMode === 'link' ? !!st.link : !!st.file && !st.fileMeta.blocked;
    $('#wnext').disabled = !ok;
  }

  async function checkPath() {
    if (!st.localPath) return;
    const notes = $('#pathnotes');
    notes.innerHTML = '<div class="note">Checking…</div>';
    try {
      const info = await post('/matches/assets/inspect-local', { path: st.localPath });
      st.localInfo = info;
      const probe = info.ffprobe || {};
      notes.innerHTML = info.ok
        ? `<div class="filemeta"><span class="chip">${icon('ok', 'sm')} Readable</span><span class="chip">${fmtBytes(info.size_bytes)}</span>
            ${probe.duration_seconds ? `<span class="chip">${fmtDuration(probe.duration_seconds)}</span>` : ''}
            ${probe.width ? `<span class="chip">${probe.width}×${probe.height}</span>` : ''}${probe.codec_name ? `<span class="chip">${esc(probe.codec_name)}</span>` : ''}</div>`
        : `<div class="errnote">${icon('alert', 'sm')}<span>${esc(info.message)}</span></div>`;
      renderSummary();
    } catch (error) { notes.innerHTML = `<div class="errnote">${icon('alert', 'sm')}<span>${esc(error.message)}</span></div>`; }
  }

  async function classifyLink() {
    const notes = $('#linknotes');
    if (!notes || !st.link) { if (notes) notes.innerHTML = ''; return; }
    try {
      const { detected } = await get(`/sources?url=${encodeURIComponent(st.link)}`);
      st.linkSource = detected;
      if (!detected) { notes.innerHTML = `<div class="warnnote">${icon('alert', 'sm')}<span>That does not look like a video URL.</span></div>`; return; }
      const missing = detected.unsupported_stats || [];
      notes.innerHTML = `<div class="note"><b>${esc(detected.label)}</b> — ${detected.supported_stat_count} of ${detected.total_stat_count} stats available. ${esc(detected.notes)}</div>
        ${missing.length ? `<div class="warnnote">${icon('info', 'sm')}<span>Not available from this source: ${missing.map((k) => esc(k.replace(/_/g, ' '))).join(', ')}.</span></div>` : ''}`;
    } catch { notes.innerHTML = ''; }
  }

  async function acceptFile(file) {
    st.file = file;
    st.fileMeta = { blocked: false, warnings: [], errors: [] };
    const meta = st.fileMeta;
    const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`;
    const allowed = policy?.allowed_extensions || ['.mp4', '.mov', '.mkv', '.avi', '.m4v'];
    if (!allowed.includes(ext)) { meta.errors.push(`'${ext}' is not a supported video format (${allowed.join(', ')}).`); meta.blocked = true; }
    if (policy && file.size > policy.max_upload_bytes) {
      meta.errors.push(`File is ${fmtBytes(file.size)} — over the ${policy.max_upload_gb} GB limit for this account.${policy.extended_upload_enabled ? '' : ' Larger uploads are available as a paid add-on.'}`);
      meta.blocked = true;
    }
    await probe(file, meta);
    if (!st.details.name) {
      $('#c_name').value = file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ');
      readDetails();
    }
    renderFileMeta();
    updateNext();
    renderSummary();
  }

  function probe(file, meta) {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      const url = URL.createObjectURL(file);
      const done = () => { URL.revokeObjectURL(url); resolve(); };
      video.onloadedmetadata = () => {
        meta.duration = video.duration; meta.width = video.videoWidth; meta.height = video.videoHeight;
        const minSeconds = policy?.min_duration_seconds || 0;
        if (minSeconds > 0 && video.duration && video.duration < minSeconds) {
          meta.errors.push(`Video is ${fmtClock(video.duration)} long; matches must be at least ${Math.round(minSeconds / 60)} minutes.`);
          meta.blocked = true;
        }
        if (meta.height && meta.height < 1080) meta.warnings.push(`Resolution is ${meta.width}×${meta.height} — 1080p or 4K footage gives much better stats and shirt reading.`);
        done();
      };
      video.onerror = done;
      video.src = url;
    });
  }

  function renderFileMeta() {
    const meta = st.fileMeta;
    $('#filemeta').innerHTML = st.file ? [
      `<span class="chip">${icon('film', 'sm')} ${esc(st.file.name)}</span>`, `<span class="chip">${fmtBytes(st.file.size)}</span>`,
      meta.duration ? `<span class="chip">${fmtDuration(meta.duration)}</span>` : '', meta.width ? `<span class="chip">${meta.width}×${meta.height}</span>` : '',
    ].join('') : '';
    $('#filenotes').innerHTML = (meta.errors || []).map((m) => `<div class="errnote">${icon('alert', 'sm')}<span>${esc(m)}</span></div>`).join('')
      + (meta.warnings || []).map((m) => `<div class="warnnote">${icon('info', 'sm')}<span>${esc(m)}</span></div>`).join('');
  }

  setMode(st.srcMode);
  if (st.file) renderFileMeta();
  if (st.link) classifyLink();
  bindNav(() => {
    readDetails();
    if (!st.details.name) st.details.name = 'Match';
    st.step = 2;
    renderStep();
  });
}

/* ---------------- step 2: teams ---------------- */

function teamBlock(side, label) {
  const name = st.teams[side];
  const color = st.teams[`${side}Color`];
  return `<div class="panel"><div class="panel-title">${label}</div>
    <div class="teampick"><div><label for="t_${side}_c">Kit</label><input type="color" id="t_${side}_c" value="${esc(color)}"></div>
      <div><label for="t_${side}">Name</label><input type="text" id="t_${side}" value="${esc(name)}"></div></div>
    <div class="presets" role="group" aria-label="${label} kit presets">${KIT_PRESETS.map((c) =>
      `<button type="button" data-side="${side}" data-c="${c}" style="background:${c}" aria-label="Kit colour ${c}"></button>`).join('')}</div></div>`;
}

function stepTeams() {
  $('#wizard').innerHTML = `<div class="cols-2">${teamBlock('home', 'Home team')}${teamBlock('away', 'Away team')}</div>
    <div class="panel">${toggleHtml('t_auto', 'Detect kit colours automatically from the video', st.teams.auto)}
      <div class="note">Recommended. A few frames are sampled and the two dominant kits are found; your colours above are used to name them (nearest match) and as a fallback.</div></div>
    ${nav(true)}`;
  for (const side of ['home', 'away']) {
    $(`#t_${side}`).oninput = (event) => { st.teams[side] = event.target.value.trim() || (side === 'home' ? 'Home' : 'Away'); renderSummary(); };
    $(`#t_${side}_c`).oninput = (event) => { st.teams[`${side}Color`] = event.target.value; renderSummary(); };
  }
  $$('[data-c]').forEach((b) => { b.onclick = () => { st.teams[`${b.dataset.side}Color`] = b.dataset.c; $(`#t_${b.dataset.side}_c`).value = b.dataset.c; renderSummary(); }; });
  $('#t_auto').onchange = (event) => { st.teams.auto = event.target.checked; };
  bindNav(() => { st.step = 3; renderStep(); });
}

/* ---------------- step 3: processing ---------------- */

function trimControls() {
  const d = sourceDuration();
  if (!d) {
    return `<div class="row"><div><label for="p_ts">Start (mm:ss)</label><input type="text" id="p_ts" placeholder="00:00" value="${st.proc.trimStart ? fmtClock(st.proc.trimStart) : ''}"></div>
      <div><label for="p_te">End (mm:ss)</label><input type="text" id="p_te" placeholder="end of video" value="${st.proc.trimEnd != null ? fmtClock(st.proc.trimEnd) : ''}"></div></div>`;
  }
  const end = st.proc.trimEnd ?? d;
  return `<div class="range2" id="range2"><div class="rail"></div><div class="sel" id="rsel"></div>
      <input type="range" id="r_a" min="0" max="${Math.floor(d)}" step="1" value="${Math.floor(st.proc.trimStart)}" aria-label="Window start">
      <input type="range" id="r_b" min="0" max="${Math.floor(d)}" step="1" value="${Math.floor(end)}" aria-label="Window end"></div>
    <div class="charthead"><span class="small num" id="r_lab"></span><span class="faint xs">of ${fmtClock(d)}</span></div>`;
}

const parseClock = (text) => {
  const parts = String(text || '').trim().split(':').map(Number);
  if (!parts.length || parts.some((n) => Number.isNaN(n))) return null;
  return parts.reduce((acc, n) => acc * 60 + n, 0);
};

function stepProcessing() {
  const p = st.proc;
  $('#wizard').innerHTML = `
    <div class="panel"><div class="panel-title">Profile</div>
      <div class="choice cols" role="radiogroup">${Object.entries(PROFILES).map(([key, prof]) => `
        <label class="opt"><input type="radio" name="p_prof" value="${key}" ${p.profile === key ? 'checked' : ''}>
          <b>${prof.label}${key === 'balanced' ? ' <span class="badge accent">Recommended</span>' : ''}</b><span class="d">${prof.blurb}</span>
          <span class="d" style="margin-top:4px"><b class="num" data-est="${key}"></b></span></label>`).join('')}</div>
      <div class="charthead" style="margin-top:12px"><span class="faint xs">Estimated processing time for ${windowLength() ? fmtDuration(windowLength()) : 'a 90-minute match'}</span>
        <select id="p_hw" style="width:auto" aria-label="Hardware class">${HARDWARE.map(([k, l]) => `<option value="${k}" ${st.hwClass === k ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
      <table class="esttable"><thead><tr><th>Hardware</th>${Object.values(PROFILES).map((x) => `<th class="r">${x.label}</th>`).join('')}</tr></thead>
        <tbody>${HARDWARE.map(([hk, hl]) => `<tr><td>${hl}</td>${Object.keys(PROFILES).map((pk) => `<td class="r num">${fmtDuration(estimateMinutes(pk, hk) * 60)}</td>`).join('')}</tr>`).join('')}</tbody></table>
    </div>
    <div class="panel"><div class="panel-title">Game camera</div>
      <div class="choice cols" role="radiogroup">${Object.entries(CAMERA_STYLES).map(([key, cam]) => `
        <label class="opt"><input type="radio" name="p_cam" value="${key}" ${p.style === key ? 'checked' : ''}><b>${cam.label}</b><span class="d">${cam.blurb}</span></label>`).join('')}</div>
      <div class="row" style="margin-top:8px"><div><label>Output resolution</label>
        <div class="seg" role="group" aria-label="Output resolution"><button data-h="1080">1080p</button><button data-h="1440">1440p</button></div></div>
        <div><label>&nbsp;</label>${toggleHtml('p_full', 'Render the full game-camera movie', p.fullMovie)}</div></div></div>
    <div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Processing window</div>
      <div class="seg" role="group" aria-label="Window presets"><button data-w="full">Full match</button><button data-w="300">First 5 min</button><button data-w="600">First 10 min (test)</button></div></div>
      ${trimControls()}<div class="note">Process a short window first to validate settings, then rerun the whole match.</div></div>
    <details class="adv"><summary>${icon('chevron', 'sm')} Advanced</summary><div class="body">
      <div class="row3"><div><label for="p_model">Detector</label><select id="p_model">
          <option value="">Auto (from profile)</option><option>yolov8n.pt</option><option>yolov8s.pt</option><option>yolov8m.pt</option><option>yolo26s.pt</option><option>yolo26m.pt</option></select></div>
        <div><label for="p_imgsz">Inference size</label><select id="p_imgsz"><option value="">Auto</option><option>960</option><option>1280</option><option>1536</option></select></div>
        <div><label for="p_stride">Frame stride</label><select id="p_stride"><option value="">Auto</option><option value="1">Every frame</option><option value="2">Every 2nd</option><option value="3">Every 3rd</option></select></div></div>
      <div class="checks" style="margin-top:16px">
        ${toggleHtml('p_reel', 'Highlights reel', p.reel)}${toggleHtml('p_cards', 'Yellow/red card detection', p.cards)}
        ${toggleHtml('p_bug', 'Scorebug overlay', p.scorebug)}${toggleHtml('p_llm', 'AI match report (needs an LLM)', p.llm)}
        ${toggleHtml('p_debug', 'Debug video (slower)', p.debug)}</div></div></details>
    ${nav(true)}`;

  const refreshEst = () => Object.keys(PROFILES).forEach((k) => { $(`[data-est="${k}"]`).textContent = `~${fmtDuration(estimateMinutes(k, st.hwClass) * 60)}`; });
  refreshEst();
  $$('input[name=p_prof]').forEach((r) => { r.onchange = () => { p.profile = r.value; if (r.value === 'quality' && p.height === 1080) setHeight(1440); renderSummary(); }; });
  $$('input[name=p_cam]').forEach((r) => { r.onchange = () => { p.style = r.value; renderSummary(); }; });
  $('#p_hw').onchange = (event) => { st.hwClass = event.target.value; refreshEst(); renderSummary(); };
  const setHeight = (h) => { p.height = h; $$('[data-h]').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.h === h))); renderSummary(); };
  $$('[data-h]').forEach((b) => { b.onclick = () => setHeight(+b.dataset.h); });
  setHeight(p.height);
  $('#p_full').onchange = (e) => { p.fullMovie = e.target.checked; };
  for (const [id, key] of [['p_reel', 'reel'], ['p_cards', 'cards'], ['p_bug', 'scorebug'], ['p_llm', 'llm'], ['p_debug', 'debug']]) $(`#${id}`).onchange = (e) => { p[key] = e.target.checked; };
  for (const [id, key] of [['p_model', 'model'], ['p_imgsz', 'imgsz'], ['p_stride', 'stride']]) { $(`#${id}`).value = p[key]; $(`#${id}`).onchange = (e) => { p[key] = e.target.value; }; }

  const d = sourceDuration();
  const syncRange = () => {
    if (!d) return;
    const a = +$('#r_a').value;
    const b = +$('#r_b').value;
    p.trimStart = Math.min(a, b);
    p.trimEnd = Math.max(a, b) >= Math.floor(d) ? null : Math.max(a, b);
    const end = p.trimEnd ?? d;
    $('#rsel').style.left = `${(100 * p.trimStart) / d}%`;
    $('#rsel').style.width = `${(100 * (end - p.trimStart)) / d}%`;
    $('#r_lab').textContent = `${fmtClock(p.trimStart)} → ${fmtClock(end)} · ${fmtDuration(end - p.trimStart)}`;
    refreshEst();
    renderSummary();
  };
  if (d) { $('#r_a').oninput = syncRange; $('#r_b').oninput = syncRange; syncRange(); } else {
    $('#p_ts').onchange = (e) => { p.trimStart = parseClock(e.target.value) || 0; };
    $('#p_te').onchange = (e) => { p.trimEnd = e.target.value.trim() ? parseClock(e.target.value) : null; };
  }
  $$('[data-w]').forEach((b) => {
    b.onclick = () => {
      p.trimStart = 0;
      p.trimEnd = b.dataset.w === 'full' ? null : +b.dataset.w;
      if (d) { $('#r_a').value = 0; $('#r_b').value = p.trimEnd == null ? Math.floor(d) : clamp(p.trimEnd, 0, Math.floor(d)); syncRange(); } else {
        $('#p_ts').value = ''; $('#p_te').value = p.trimEnd == null ? '' : fmtClock(p.trimEnd);
      }
    };
  });
  bindNav(() => { st.step = 4; renderStep(); });
}

/* ---------------- step 4: review & start ---------------- */

const hhmmss = (s) => { const t = Math.round(s); return `${String(Math.floor(t / 3600)).padStart(2, '0')}:${String(Math.floor((t % 3600) / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`; };

export function buildConfig(state = st) {
  const p = state.proc;
  const t = state.teams;
  const config = {
    profile: p.profile,
    camera_mode: 'follow_ball',
    camera_style: p.style,
    zoom_factor: CAMERA_STYLES[p.style].zoom,
    output_height: p.height,
    render_full_follow_cam: p.fullMovie,
    broadcast_reel: p.reel,
    detect_cards: p.cards,
    llm_report: p.llm,
    scorebug: p.scorebug,
    debug_video: p.debug,
    team_left: t.home,
    team_right: t.away,
    team_left_color: t.homeColor,
    team_right_color: t.awayColor,
    auto_detect_team_colors: t.auto,
  };
  if (p.trimStart > 0) config.trim_start = hhmmss(p.trimStart);
  if (p.trimEnd != null) config.trim_end = hhmmss(p.trimEnd);
  if (p.model) config.yolo_model = p.model;
  if (p.imgsz) config.inference_imgsz = +p.imgsz;
  if (p.stride) config.vid_stride = +p.stride;
  if (state.details.email) config.notify_email = state.details.email;
  return config;
}

function stepReview() {
  const config = buildConfig();
  $('#wizard').innerHTML = `<div class="panel"><div class="panel-title">Ready to start</div>
      <p class="muted small" style="margin-bottom:12px">We’ll create the match, ${st.srcMode === 'upload' ? 'upload the video' : st.srcMode === 'local' ? 'register the server file' : 'queue the link'}, and queue processing. You can leave this page — progress is on the Jobs page${st.details.email ? `, and we’ll email ${esc(st.details.email)}` : ''}.</p>
      <details class="adv"><summary>${icon('chevron', 'sm')} Job configuration</summary><div class="body"><pre>${esc(JSON.stringify(config, null, 2))}</pre></div></details>
      <div id="prog" style="margin-top:16px"></div></div>${nav(true, `${icon('play', 'sm')} Start processing`)}`;
  bindNav(submit);
}

async function submit() {
  const button = $('#wnext');
  const back = $('#wback');
  const prog = $('#prog');
  button.disabled = true;
  if (back) back.disabled = true;
  const say = (html) => { prog.innerHTML = html; };
  try {
    say('<div class="small">Creating match…</div><div class="bar indeterminate" style="margin-top:8px"><i></i></div>');
    const d = st.details;
    const match = await post('/matches', {
      name: d.name || 'Match', home_team_name: st.teams.home, away_team_name: st.teams.away, match_date: d.date || null,
      source_video_path: st.srcMode === 'link' ? st.link : st.srcMode === 'local' ? st.localPath : st.file?.name || '',
      metadata: d.email ? { notify_email: d.email } : {},
    });
    if (st.srcMode === 'local') {
      say('<div class="small">Registering server file…</div><div class="bar indeterminate" style="margin-top:8px"><i></i></div>');
      await post(`/matches/${match.match_id}/assets/register-local`, { path: st.localPath });
    } else if (st.srcMode === 'upload') {
      const started = Date.now();
      say('<div class="charthead"><span class="small">Uploading…</span><span class="small num" id="uppct">0%</span></div><div class="bar"><i id="upbar" style="width:0"></i></div><div class="note" id="uprate"></div>');
      await uploadFile(`/matches/${match.match_id}/assets/upload`, st.file, (fraction) => {
        const bar = $('#upbar');
        if (!bar) return;
        bar.style.width = `${Math.round(fraction * 100)}%`;
        $('#uppct').textContent = `${Math.round(fraction * 100)}%`;
        const secs = (Date.now() - started) / 1000;
        const rate = (fraction * st.file.size) / Math.max(secs, 0.1);
        const left = rate > 0 ? ((1 - fraction) * st.file.size) / rate : 0;
        $('#uprate').textContent = `${fmtBytes(rate)}/s · ${fmtDuration(left)} left`;
      });
    }
    say('<div class="small">Queuing the job…</div><div class="bar indeterminate" style="margin-top:8px"><i></i></div>');
    const job = await post(`/matches/${match.match_id}/jobs`, { config: buildConfig() });
    toast('Processing queued');
    go('jobs', job.job_id);
  } catch (error) {
    say(`<div class="errnote">${icon('alert', 'sm')}<span>${esc(error.message)}</span></div>`);
    button.disabled = false;
    if (back) back.disabled = false;
  }
}
