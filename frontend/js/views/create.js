// Guided upload: a 3-step wizard with drag-and-drop, pre-flight validation
// against the server upload policy (size caps, formats, minimum length),
// upload progress, and turnaround (SLA) messaging once the job is queued.

import { get, post, uploadFile } from '../api.js';
import { $, esc, fmtBytes, fmtSeconds, setMain } from '../ui.js';

const state = { step: 1, policy: null, sources: [], file: null, fileMeta: {}, localPath: '', link: '', linkSource: null };

export async function renderCreate() {
  state.step = 1; state.file = null; state.fileMeta = {}; state.localPath = ''; state.link = ''; state.linkSource = null;
  setMain(`
    <h1>Create</h1>
    <div class="sub">Upload a match and let the analysis run — nothing to configure unless you want to</div>
    <div class="steps" id="steps"></div>
    <form id="wizard" onsubmit="return false"></form>`);
  try {
    state.policy = await get('/matches/upload-policy');
  } catch {
    state.policy = null;
  }
  try {
    state.sources = (await get('/sources')).sources;
  } catch {
    state.sources = [];
  }
  renderStep();
}

function stepsBar() {
  const labels = ['1 · Match details', '2 · Video', '3 · Processing'];
  $('#steps').innerHTML = labels.map((label, index) => {
    const number = index + 1;
    const cls = number === state.step ? 'active' : (number < state.step ? 'done' : '');
    return `<span class="step ${cls}">${label}</span>`;
  }).join('');
}

function renderStep() {
  stepsBar();
  if (state.step === 1) return stepDetails();
  if (state.step === 2) return stepVideo();
  return stepProcessing();
}

/* ---- step 1: match details ---- */

function stepDetails() {
  const previous = state.details || {};
  $('#wizard').innerHTML = `
    <div class="panel"><h3>Match</h3>
      <div class="row">
        <div><label>Match name</label><input type="text" id="c_name" value="${esc(previous.name || 'Match')}" required></div>
        <div><label>Date</label><input type="date" id="c_date" value="${esc(previous.date || '')}"></div>
      </div>
      <div class="row">
        <div><label>Home team</label><input type="text" id="c_left" value="${esc(previous.home || 'HOME')}"></div>
        <div><label>Away team</label><input type="text" id="c_right" value="${esc(previous.away || 'AWAY')}"></div>
      </div>
      <label>Email me when it's ready (optional)</label>
      <input type="email" id="c_email" value="${esc(previous.email || '')}" placeholder="coach@example.com">
      <div class="note">We'll send a note when processing finishes${slaText()}.</div>
    </div>
    <button class="btn" id="next1">Continue</button>`;
  $('#next1').onclick = () => {
    state.details = {
      name: $('#c_name').value.trim() || 'Match',
      date: $('#c_date').value,
      home: $('#c_left').value.trim() || 'HOME',
      away: $('#c_right').value.trim() || 'AWAY',
      email: $('#c_email').value.trim(),
    };
    state.step = 2;
    renderStep();
  };
}

function slaText() {
  const sla = state.policy?.processing_sla_hours;
  return sla?.length === 2 ? ` — most matches take ${sla[0]}–${sla[1]} hours` : '';
}

/* ---- step 2: video source with validation ---- */

function stepVideo() {
  const policy = state.policy;
  const maxLabel = policy ? `${policy.max_upload_gb} GB` : '3 GB';
  const extensions = (policy?.allowed_extensions || ['.mp4', '.mov', '.mkv', '.avi', '.m4v']).join(', ');
  const linkProviders = (state.sources || []).filter((source) => source.kind === 'link' && source.key !== 'other_link');
  $('#wizard').innerHTML = `
    <div class="panel"><h3>Video source</h3>
      <div class="checks">
        <label><input type="radio" name="c_src" id="c_src_up" checked> Upload a file</label>
        <label><input type="radio" name="c_src" id="c_src_link"> Paste a video link</label>
        <label><input type="radio" name="c_src" id="c_src_local"> Local file on the server (no upload)</label>
      </div>
      <div id="uprow">
        <div class="dropzone" id="drop">
          <div class="big">Drag a match video here, or click to choose</div>
          <div>${esc(extensions)} · up to ${esc(maxLabel)}${policy && !policy.extended_upload_enabled ? ' (larger caps available as an add-on)' : ''}</div>
        </div>
        <input type="file" id="c_file" accept="video/*" style="display:none">
        <div class="filemeta" id="filemeta"></div>
        <div id="filenotes"></div>
      </div>
      <div id="linkrow" style="display:none">
        <label>Public video link</label>
        <input type="text" id="c_link" placeholder="https://www.youtube.com/watch?v=…" value="${esc(state.link)}">
        <div id="linknotes"></div>
        <div class="note">Supported: ${linkProviders.map((source) => esc(source.label)).join(', ')}.
          Raw file uploads always produce the most complete statistics.</div>
      </div>
      <div id="pathrow" style="display:none">
        <label>Absolute path on the server / container</label>
        <input type="text" id="c_path" placeholder="/data/videos/match.mp4" value="${esc(state.localPath)}">
        <div class="note">Skips the upload — point at a file already on the machine running the worker.</div>
      </div>
    </div>
    <button class="btn2" id="back2">← Back</button>
    <button class="btn" id="next2" disabled>Continue</button>`;

  const currentMode = () => $('#c_src_local').checked ? 'local' : ($('#c_src_link').checked ? 'link' : 'upload');
  const setSource = () => {
    const mode = currentMode();
    $('#uprow').style.display = mode === 'upload' ? '' : 'none';
    $('#linkrow').style.display = mode === 'link' ? '' : 'none';
    $('#pathrow').style.display = mode === 'local' ? '' : 'none';
    updateNext();
  };
  $('#c_src_up').onchange = setSource;
  $('#c_src_link').onchange = setSource;
  $('#c_src_local').onchange = setSource;
  $('#c_path').oninput = () => { state.localPath = $('#c_path').value.trim(); updateNext(); };

  let linkTimer = null;
  $('#c_link').oninput = () => {
    state.link = $('#c_link').value.trim();
    state.linkSource = null;
    updateNext();
    clearTimeout(linkTimer);
    linkTimer = setTimeout(classifyLink, 400);
  };
  if (state.link) classifyLink();

  async function classifyLink() {
    const notes = $('#linknotes');
    if (!notes || !state.link) { if (notes) notes.innerHTML = ''; return; }
    try {
      const { detected } = await get(`/sources?url=${encodeURIComponent(state.link)}`);
      state.linkSource = detected;
      if (!detected) { notes.innerHTML = '<div class="warnnote">That does not look like a video URL.</div>'; return; }
      const missing = detected.unsupported_stats || [];
      notes.innerHTML = `
        <div class="note"><b>${esc(detected.label)}</b> — ${detected.supported_stat_count} of
          ${detected.total_stat_count} stats available. ${esc(detected.notes)}</div>
        ${missing.length ? `<div class="warnnote">Not available from this source:
          ${missing.map((key) => esc(key.replace(/_/g, ' '))).join(', ')}.
          Upload the raw file instead to get all ${detected.total_stat_count}.</div>` : ''}`;
    } catch {
      notes.innerHTML = '';
    }
  }

  const drop = $('#drop');
  drop.onclick = () => $('#c_file').click();
  drop.ondragover = (event) => { event.preventDefault(); drop.classList.add('drag'); };
  drop.ondragleave = () => drop.classList.remove('drag');
  drop.ondrop = (event) => {
    event.preventDefault();
    drop.classList.remove('drag');
    if (event.dataTransfer.files[0]) acceptFile(event.dataTransfer.files[0]);
  };
  $('#c_file').onchange = (event) => { if (event.target.files[0]) acceptFile(event.target.files[0]); };

  $('#back2').onclick = () => { state.step = 1; renderStep(); };
  $('#next2').onclick = () => { state.step = 3; renderStep(); };

  function updateNext() {
    const mode = currentMode();
    if (mode === 'local') $('#next2').disabled = !state.localPath;
    else if (mode === 'link') $('#next2').disabled = !state.link;
    else $('#next2').disabled = !state.file || state.fileMeta.blocked;
  }

  async function acceptFile(file) {
    state.file = file;
    state.fileMeta = { blocked: false, warnings: [], errors: [] };
    const meta = state.fileMeta;
    const extension = ('.' + (file.name.split('.').pop() || '')).toLowerCase();
    const allowed = policy?.allowed_extensions || ['.mp4', '.mov', '.mkv', '.avi', '.m4v'];
    if (!allowed.includes(extension)) {
      meta.errors.push(`'${extension}' is not a supported video format (${allowed.join(', ')}).`);
      meta.blocked = true;
    }
    if (policy && file.size > policy.max_upload_bytes) {
      meta.errors.push(`File is ${fmtBytes(file.size)} — over the ${policy.max_upload_gb} GB limit for this account.`
        + (policy.extended_upload_enabled ? '' : ' Larger uploads are available as a paid add-on.'));
      meta.blocked = true;
    }
    await probeDuration(file, meta);
    renderFileMeta();
    updateNext();
  }

  function probeDuration(file, meta) {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      const url = URL.createObjectURL(file);
      const done = () => { URL.revokeObjectURL(url); resolve(); };
      video.onloadedmetadata = () => {
        meta.duration = video.duration;
        meta.width = video.videoWidth;
        meta.height = video.videoHeight;
        const minSeconds = policy?.min_duration_seconds || 0;
        if (minSeconds > 0 && video.duration && video.duration < minSeconds) {
          meta.errors.push(`Video is ${fmtSeconds(video.duration)} long; matches must be at least ${Math.round(minSeconds / 60)} minutes.`);
          meta.blocked = true;
        }
        if (meta.height && meta.height < 1080) {
          meta.warnings.push(`Resolution is ${meta.width}×${meta.height} — 1080p or 4K footage produces much better stats and jersey reading.`);
        }
        done();
      };
      video.onerror = done;
      video.src = url;
    });
  }

  function renderFileMeta() {
    const meta = state.fileMeta;
    $('#filemeta').innerHTML = state.file ? [
      `<span class="chip">${esc(state.file.name)}</span>`,
      `<span class="chip">${fmtBytes(state.file.size)}</span>`,
      meta.duration ? `<span class="chip">${fmtSeconds(meta.duration)}</span>` : '',
      meta.width ? `<span class="chip">${meta.width}×${meta.height}</span>` : '',
    ].filter(Boolean).join('') : '';
    $('#filenotes').innerHTML =
      meta.errors.map((message) => `<div class="errnote">✕ ${esc(message)}</div>`).join('') +
      meta.warnings.map((message) => `<div class="warnnote">⚠ ${esc(message)}</div>`).join('');
  }
}

/* ---- step 3: processing options + submit ---- */

function stepProcessing() {
  $('#wizard').innerHTML = `
    <div class="panel"><h3>Camera &amp; highlights</h3>
      <div class="row3">
        <div><label>Camera mode</label><select id="c_cam">
          <option value="follow_ball" selected>Game camera (follow ball)</option>
          <option value="follow_action">Follow player + action</option>
          <option value="follow_player">Follow player</option>
          <option value="wide">Wide (no crop)</option></select></div>
        <div><label>Zoom</label><input type="number" id="c_zoom" value="1.8" step="0.1" min="1" max="3"></div>
        <div><label>Speed</label><select id="c_speed"><option value="quality" selected>Best quality</option><option value="fast">Fast (~2x)</option></select></div>
      </div>
      <div class="checks">
        <label><input type="checkbox" id="c_full" checked> Full game-camera movie</label>
        <label><input type="checkbox" id="c_reel" checked> Broadcast reel</label>
        <label><input type="checkbox" id="c_teams" checked> Team stats by jersey color</label>
        <label><input type="checkbox" id="c_autocolors" checked> Auto-detect jersey colors</label>
        <label><input type="checkbox" id="c_cards" checked> Yellow/red card detection</label>
        <label><input type="checkbox" id="c_llm" checked> AI match report</label>
        <label><input type="checkbox" id="c_bug" checked> Scorebug overlay</label>
        <label><input type="checkbox" id="c_debug"> Debug video</label>
      </div>
    </div>
    <div class="panel"><h3>Processing window (optional)</h3>
      <div class="viewbar" style="margin:2px 0 10px">
        <button type="button" class="btn2" data-window=",">Full match</button>
        <button type="button" class="btn2" data-window="00:00,05:00">First 5 min</button>
        <button type="button" class="btn2" data-window="00:00,10:00">First 10 min (test run)</button>
      </div>
      <div class="row">
        <div><label>Window start (MM:SS)</label><input type="text" id="c_tstart"></div>
        <div><label>Window end (MM:SS)</label><input type="text" id="c_tend"></div>
      </div>
      <div class="note">Process a short window first to validate settings, then re-run the full match.</div>
    </div>
    <button class="btn2" id="back3">← Back</button>
    <button class="btn" id="go">Start analysis</button>
    <div id="prog" style="margin-top:14px"></div>`;

  document.querySelectorAll('button[data-window]').forEach((button) => {
    button.onclick = () => {
      const [start, end] = button.dataset.window.split(',');
      $('#c_tstart').value = start; $('#c_tend').value = end;
    };
  });
  $('#back3').onclick = () => { state.step = 2; renderStep(); };
  $('#go').onclick = submit;
}

async function submit() {
  const button = $('#go');
  const progress = $('#prog');
  button.disabled = true;
  const details = state.details;
  try {
    progress.innerHTML = 'Creating match…';
    const metadata = details.email ? { notify_email: details.email } : {};
    const match = await post('/matches', {
      name: details.name,
      home_team_name: details.home,
      away_team_name: details.away,
      match_date: details.date || null,
      // A pasted link is stored as the source path and classified server-side.
      source_video_path: state.link || state.localPath || state.file?.name || '',
      metadata,
    });

    if (state.link) {
      progress.innerHTML = 'Queuing link-based analysis…';
    } else if (state.localPath) {
      progress.innerHTML = 'Registering local video…';
      await post(`/matches/${match.match_id}/assets/register-local`, { path: state.localPath });
    } else {
      progress.innerHTML = 'Uploading… <div class="bar"><i id="upbar" style="width:0"></i></div>';
      await uploadFile(`/matches/${match.match_id}/assets/upload`, state.file, (fraction) => {
        const bar = $('#upbar');
        if (bar) bar.style.width = `${Math.round(fraction * 100)}%`;
      });
    }

    progress.innerHTML = 'Starting job…';
    const config = {
      camera_mode: $('#c_cam').value,
      zoom_factor: +$('#c_zoom').value,
      render_full_follow_cam: $('#c_full').checked,
      broadcast_reel: $('#c_reel').checked,
      detect_cards: $('#c_cards').checked,
      debug_video: $('#c_debug').checked,
      scorebug: $('#c_bug').checked,
      team_left: details.home,
      team_right: details.away,
      vid_stride: $('#c_speed').value === 'fast' ? 2 : 1,
      inference_imgsz: $('#c_speed').value === 'fast' ? 736 : 960,
      llm_report: $('#c_llm').checked,
    };
    if (details.email) config.notify_email = details.email;
    if ($('#c_teams').checked) config.auto_detect_team_colors = $('#c_autocolors').checked;
    if ($('#c_tstart').value.trim()) config.trim_start = $('#c_tstart').value.trim();
    if ($('#c_tend').value.trim()) config.trim_end = $('#c_tend').value.trim();

    const job = await post(`/matches/${match.match_id}/jobs`, { config });
    progress.innerHTML = `✅ Job <b>${esc(job.job_id)}</b> queued${esc(slaText())}.
      <a href="#jobs">Track progress →</a> · <a href="#matches/${encodeURIComponent(match.match_id)}">Open match →</a>
      ${details.email ? `<div class="note">We'll email ${esc(details.email)} when it's done (check spam the first time).</div>` : ''}`;
  } catch (error) {
    progress.innerHTML = `<div class="errnote">✕ ${esc(error.message)}</div>`;
  }
  button.disabled = false;
}
