// "Pick player" mode: pause the wide view, fetch tracked boxes at the
// current window time, overlay clickable team-coloured boxes scaled to the
// rendered video, and open the player card on click.

import { get } from '../../api.js';
import { icon } from '../../icons.js';
import { playerName, teamColor, teamName } from '../../runmodel.js';
import { $, esc, fmtClock, toast } from '../../ui.js';
import { openPlayerCard } from './playercard.js';

export function createPicker(ctx, player, { onFilterEvents }) {
  const model = ctx.model;
  const overlay = player.overlay;
  const state = { on: false, data: null, selected: null, seq: 0 };

  // Rendered video rectangle inside the stage (object-fit: contain).
  function videoRect() {
    const v = player.video;
    const sw = player.stage.clientWidth;
    const sh = player.stage.clientHeight;
    const vw = v.videoWidth || 16;
    const vh = v.videoHeight || 9;
    const scale = Math.min(sw / vw, sh / vh);
    return { x: (sw - vw * scale) / 2, y: (sh - vh * scale) / 2, w: vw * scale, h: vh * scale };
  }

  function draw() {
    if (!state.on) { overlay.innerHTML = ''; overlay.classList.remove('picking'); return; }
    overlay.classList.add('picking');
    const data = state.data;
    const hint = `<div class="pickhint">${icon('target', 'sm')}<span>${data ? (data.players.length
      ? `Click a player · ${data.players.length} tracked at ${fmtClock(data.t)}` : `Nobody tracked at ${fmtClock(data.t)} — scrub a little`) : 'Loading players…'}</span><kbd>Esc</kbd></div>`;
    if (!data) { overlay.innerHTML = hint; return; }
    const r = videoRect();
    const fw = data.frame_width || player.video.videoWidth;
    const fh = data.frame_height || player.video.videoHeight;
    const sx = r.w / fw;
    const sy = r.h / fh;
    const boxes = data.players.map((p) => {
      const left = r.x + p.x1 * sx;
      const top = r.y + p.y1 * sy;
      const width = Math.max(10, (p.x2 - p.x1) * sx);
      const height = Math.max(14, (p.y2 - p.y1) * sy);
      const name = p.name || (p.number ? `#${p.number}` : `${p.track_id}`);
      return `<button class="pbox ${state.selected === p.track_id ? 'sel' : ''}" data-tid="${p.track_id}"
        style="left:${left}px;top:${top}px;width:${width}px;height:${height}px;--c:${esc(teamColor(model, p.team))}"
        aria-label="${esc(playerName(model, p.track_id, p.number))}, ${esc(teamName(model, p.team))}"><span>${esc(name)}</span></button>`;
    }).join('');
    overlay.innerHTML = boxes + hint;
  }

  async function load() {
    const t = player.windowTime();
    if (t == null) return;
    const seq = ++state.seq;
    state.data = null;
    draw();
    try {
      const data = await get(`/studio/runs/${encodeURIComponent(ctx.runId)}/tracks?t=${t.toFixed(2)}&tolerance=0.4`);
      if (seq !== state.seq || !state.on) return;
      state.data = data;
      draw();
    } catch (error) {
      toast(error.message, 'err');
      stop();
    }
  }

  function start() {
    if (!model.tracksAvailable) { toast('This run has no tracks.npz, so players cannot be picked.', 'warn'); return; }
    const wide = model.sources.find((s) => s.wide);
    if (!wide) { toast('Player picking needs the wide view (proxy video), which this run does not have.', 'warn'); return; }
    state.on = true;
    player.pause();
    if (player.source?.key !== wide.key) {
      player.setSource(wide.key);
      player.video.addEventListener('loadedmetadata', () => { if (state.on) load(); }, { once: true });
    } else {
      load();
    }
    btn?.classList.add('on');
    btn?.setAttribute('aria-pressed', 'true');
  }

  function stop() {
    state.on = false;
    state.seq += 1;
    state.data = null;
    draw();
    btn?.classList.remove('on');
    btn?.setAttribute('aria-pressed', 'false');
  }

  overlay.addEventListener('click', (event) => {
    const box = event.target.closest('.pbox');
    if (!box) return;
    event.stopPropagation();
    const tid = +box.dataset.tid;
    const seen = state.data?.players.find((p) => p.track_id === tid);
    state.selected = tid;
    draw();
    openPlayerCard(ctx, tid, {
      team: seen?.team, jersey: seen?.number, onFilterEvents,
      onSeek: (t) => { stop(); player.seekWindow(t, { play: true, preroll: 3 }); },
    });
  });

  const onSeeked = () => { if (state.on) load(); };
  const onPlay = () => { if (state.on) stop(); };
  player.video.addEventListener('seeked', onSeeked);
  player.video.addEventListener('play', onPlay);
  const ro = new ResizeObserver(draw);
  ro.observe(player.stage);

  // Toolbar button above the player.
  player.extra.insertAdjacentHTML('beforeend', `<button class="btn2 sm" id="pickbtn" aria-pressed="false"
    ${model.tracksAvailable ? '' : 'disabled title="No player tracks in this run"'}>${icon('target', 'sm')} Pick player <kbd>T</kbd></button>`);
  const btn = $('#pickbtn', player.extra);
  btn.onclick = () => (state.on ? stop() : start());

  return {
    get active() { return state.on; },
    start,
    stop,
    toggle() { return state.on ? stop() : start(); },
    destroy() { ro.disconnect(); player.video.removeEventListener('seeked', onSeeked); player.video.removeEventListener('play', onPlay); },
  };
}
