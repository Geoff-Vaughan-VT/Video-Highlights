// Video player with source selector and custom controls.
//
// Timebase: events use the processing-window clock. Window sources (game
// camera, wide proxy, debug) share it; reel/clips have their own clock.
// `shift` converts window time to the element's time: normally 0, but if a
// wide proxy turns out to cover the whole source (its duration ~ trim +
// window), shift = trim offset.

import { runFileUrl } from '../../api.js';
import { icon } from '../../icons.js';
import { $, $$, clamp, esc, fmtClock } from '../../ui.js';

const RATES = [0.25, 0.5, 1, 1.5, 2];

export function createPlayer(root, model, { initial } = {}) {
  const sources = model.sources;
  const listeners = { time: [], source: [], meta: [] };
  const state = { source: null, shift: 0, pendingSeek: null, autoplay: false };

  root.innerHTML = `
    <div class="charthead" style="margin-bottom:8px">
      <div class="seg" id="srcseg" role="group" aria-label="Video source">
        ${sources.filter((s) => !s.clip).map((s) => `<button data-src="${esc(s.key)}">${esc(s.label)}</button>`).join('')}
        ${sources.some((s) => s.clip) ? `<select id="clipsel" aria-label="Clips" style="width:auto;border:0;background:none;padding:6px 8px;font-size:13px">
          <option value="">Clips…</option>${sources.filter((s) => s.clip).map((s) => `<option value="${esc(s.key)}">${esc(s.label)}</option>`).join('')}</select>` : ''}
      </div>
      <div class="btnrow" id="playerextra"></div>
    </div>
    <div class="stage" id="stage">
      <video id="vid" preload="metadata" playsinline></video>
      <div class="overlay" id="overlay"></div>
      <div class="flash" id="flash"></div>
      ${sources.length ? '' : `<div class="center-msg">${icon('film', 'lg')}<div>No playable video in this run yet.</div></div>`}
    </div>
    <div class="controls" role="toolbar" aria-label="Playback">
      <button class="iconbtn" data-k="prev" title="Previous event (P)" aria-label="Previous event">${icon('prev')}</button>
      <button class="iconbtn" data-k="back" title="Back 5 s (J / ←)" aria-label="Back 5 seconds">${icon('back5')}</button>
      <button class="iconbtn" data-k="play" title="Play / pause (Space / K)" aria-label="Play">${icon('play')}</button>
      <button class="iconbtn" data-k="fwd" title="Forward 5 s (L / →)" aria-label="Forward 5 seconds">${icon('fwd5')}</button>
      <button class="iconbtn" data-k="next" title="Next event (N)" aria-label="Next event">${icon('next')}</button>
      <button class="iconbtn" data-k="stepb" title="Previous frame (,)" aria-label="Previous frame">${icon('stepb', 'sm')}</button>
      <button class="iconbtn" data-k="stepf" title="Next frame (.)" aria-label="Next frame">${icon('stepf', 'sm')}</button>
      <div class="tc" id="tc" aria-live="off">00:00</div>
      <div class="spacer"></div>
      <select id="rate" aria-label="Playback speed">${RATES.map((r) => `<option value="${r}" ${r === 1 ? 'selected' : ''}>${r}×</option>`).join('')}</select>
      <button class="iconbtn" data-k="mute" title="Mute (M)" aria-label="Mute">${icon('volume')}</button>
      <input type="range" id="vol" min="0" max="1" step="0.05" value="1" aria-label="Volume">
      <button class="iconbtn" data-k="full" title="Fullscreen (F)" aria-label="Fullscreen">${icon('fullscreen')}</button>
    </div>`;

  const video = $('#vid', root);
  const stage = $('#stage', root);
  const tc = $('#tc', root);
  const fps = +model.run.tracks_meta?.fps || 25;

  const emit = (type, payload) => listeners[type].forEach((fn) => fn(payload));
  const isWindow = () => state.source?.timebase === 'window';
  const windowTime = () => (isWindow() ? video.currentTime - state.shift : null);

  function flash(name) {
    const node = $('#flash', root);
    node.innerHTML = icon(name, 'lg');
    node.classList.add('show');
    requestAnimationFrame(() => setTimeout(() => node.classList.remove('show'), 250));
  }

  function renderTime() {
    const t = video.currentTime || 0;
    if (isWindow()) {
      const w = t - state.shift;
      tc.innerHTML = `${fmtClock(w)}<small title="Source-file timecode (window start + trim offset)">SRC ${fmtClock(w + model.trim)}</small>`;
    } else {
      tc.innerHTML = `${fmtClock(t)}<small>/ ${fmtClock(video.duration || 0)}</small>`;
    }
    $('[data-k=play]', root).innerHTML = icon(video.paused ? 'play' : 'pause');
    $('[data-k=play]', root).setAttribute('aria-label', video.paused ? 'Play' : 'Pause');
  }

  function setSource(key, { keepTime = true } = {}) {
    const next = sources.find((s) => s.key === key);
    if (!next || next === state.source) return;
    const carry = keepTime && isWindow() && next.timebase === 'window' ? windowTime() : null;
    state.autoplay = !video.paused;
    state.source = next;
    state.shift = 0;
    state.pendingSeek = carry;
    video.src = runFileUrl(model.runId, next.file);
    $$('[data-src]', root).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.src === key)));
    const sel = $('#clipsel', root);
    if (sel) sel.value = next.clip ? key : '';
    emit('source', next);
  }

  video.addEventListener('loadedmetadata', () => {
    const src = state.source;
    if (src?.wide && model.trim > 5 && video.duration > model.duration + model.trim * 0.8) state.shift = model.trim;
    if (state.pendingSeek != null) video.currentTime = clamp(state.pendingSeek + state.shift, 0, video.duration || 1e9);
    state.pendingSeek = null;
    if (state.autoplay) video.play().catch(() => {});
    emit('meta', { duration: video.duration, width: video.videoWidth, height: video.videoHeight });
    renderTime();
  });
  video.addEventListener('error', () => {
    const msg = document.createElement('div');
    msg.className = 'center-msg';
    msg.id = 'vid_err';
    msg.innerHTML = `<div>${icon('alert', 'lg')}<div style="margin-top:8px">This browser could not play “${esc(state.source?.label || 'video')}”.</div>
      <div class="faint xs" style="margin-top:4px">H.264/AAC needs Chrome, Edge, Safari or Firefox with system codecs. <a href="${runFileUrl(model.runId, state.source?.file || '')}" download>Download the file</a></div></div>`;
    $('#vid_err', root)?.remove();
    stage.appendChild(msg);
  });
  video.addEventListener('loadstart', () => $('#vid_err', root)?.remove());
  video.addEventListener('timeupdate', () => { renderTime(); emit('time', windowTime()); });
  video.addEventListener('seeked', () => { renderTime(); emit('time', windowTime()); });
  video.addEventListener('play', renderTime);
  video.addEventListener('pause', renderTime);
  video.addEventListener('click', () => api.toggle());
  video.addEventListener('dblclick', () => api.fullscreen());

  const api = {
    video,
    stage,
    overlay: $('#overlay', root),
    extra: $('#playerextra', root),
    get source() { return state.source; },
    get shift() { return state.shift; },
    windowTime,
    isWindow,
    on(type, fn) { listeners[type].push(fn); },
    setSource,
    play() { video.play().catch(() => {}); },
    pause() { video.pause(); },
    toggle() { if (video.paused) { video.play().catch(() => {}); flash('play'); } else { video.pause(); flash('pause'); } },
    // Seek to a window time; switches to a window source when on a reel/clip.
    seekWindow(t, { play = false, preroll = 0 } = {}) {
      if (!isWindow()) {
        const fallback = sources.find((s) => s.key === 'movie') || sources.find((s) => s.timebase === 'window');
        if (!fallback) return;
        state.autoplay = play;
        setSource(fallback.key, { keepTime: false });
        state.pendingSeek = Math.max(0, t - preroll);
        return;
      }
      video.currentTime = clamp(t - preroll + state.shift, 0, video.duration || 1e9);
      if (play) video.play().catch(() => {});
    },
    nudge(seconds) { video.currentTime = clamp(video.currentTime + seconds, 0, video.duration || 1e9); },
    step(frames) { video.pause(); video.currentTime = clamp(video.currentTime + frames / fps, 0, video.duration || 1e9); },
    setRate(rate) { video.playbackRate = rate; $('#rate', root).value = String(rate); },
    fullscreen() {
      if (document.fullscreenElement) document.exitFullscreen();
      else stage.requestFullscreen?.().catch(() => {});
    },
    toggleMute() {
      video.muted = !video.muted;
      $('[data-k=mute]', root).innerHTML = icon(video.muted ? 'mute' : 'volume');
    },
    destroy() {
      video.pause();
      video.removeAttribute('src');
      video.load();
    },
  };

  $$('[data-src]', root).forEach((button) => { button.onclick = () => setSource(button.dataset.src); });
  const clipSel = $('#clipsel', root);
  if (clipSel) clipSel.onchange = () => { if (clipSel.value) { state.autoplay = true; setSource(clipSel.value, { keepTime: false }); } };
  $('#rate', root).onchange = (event) => { video.playbackRate = +event.target.value; };
  $('#vol', root).oninput = (event) => { video.volume = +event.target.value; video.muted = false; };
  root.addEventListener('click', (event) => {
    const key = event.target.closest('[data-k]')?.dataset.k;
    if (!key) return;
    if (key === 'play') api.toggle();
    if (key === 'back') api.nudge(-5);
    if (key === 'fwd') api.nudge(5);
    if (key === 'stepb') api.step(-1);
    if (key === 'stepf') api.step(1);
    if (key === 'mute') api.toggleMute();
    if (key === 'full') api.fullscreen();
    if (key === 'prev' || key === 'next') root.dispatchEvent(new CustomEvent('vh-event-nav', { detail: key }));
  });

  const start = sources.find((s) => s.key === initial) || sources.find((s) => s.key === 'movie') || sources[0];
  if (start) setSource(start.key);
  renderTime();
  return api;
}
