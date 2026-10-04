// Scrubbable match timeline: event markers (coloured by type), possession
// ribbon and momentum sparkline from team stats, hover tooltip with the
// nearest thumbnail, click/drag to seek, and a thumbnail strip.

import { runThumbUrl } from '../../api.js';
import { $, $$, clamp, esc, fmtClock } from '../../ui.js';
import { playerName, teamName } from '../../runmodel.js';

function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

function hexToRgba(hex, alpha) {
  const h = String(hex || '#888').replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h.padEnd(6, '0');
  const n = parseInt(full.slice(0, 6), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

export function createTimeline(root, model, { onSeek, onEvent }) {
  const tl = model.teamStats.timeline;
  const poss = tl?.possession_pct_team0 || [];
  const mom = tl?.momentum || [];
  const binS = +tl?.bin_s || 60;
  const D = Math.max(1, model.duration);
  const state = { t: 0, hidden: false, filter: null, dragging: false };

  root.innerHTML = `<div class="timeline" id="tl">
    <div class="tl-head">
      <div class="legend">
        ${poss.length ? `<span><span class="swatch" style="background:${esc(model.teams[0].color)}"></span><span class="swatch" style="background:${esc(model.teams[1].color)};margin-left:-4px"></span>Possession</span>` : ''}
        ${mom.length ? '<span>Momentum</span>' : ''}
        <span>${model.events.length} events</span>
      </div>
      <div id="tlnote"></div>
    </div>
    <div class="tl-body" id="tlbody" role="slider" tabindex="0" aria-label="Match timeline" aria-valuemin="0" aria-valuemax="${Math.round(D)}" aria-valuenow="0">
      <div class="markers" id="tlmarkers">${model.events.map((e, i) => `<button class="mk ${e.type === 'goal' ? 'big' : ''}" data-i="${i}"
        style="left:${(100 * clamp(e.t / D, 0, 1)).toFixed(3)}%;--c:${e.info.color}" aria-label="${esc(e.info.label)} at ${fmtClock(e.t)}"></button>`).join('')}</div>
      ${poss.length ? '<div class="lane"><canvas id="tlposs" height="10"></canvas></div>' : ''}
      ${mom.length ? '<div class="lane"><canvas id="tlmom" height="30"></canvas></div>' : ''}
      <div class="playhead" id="tlhead" style="left:0"></div>
      <div class="hoverline" id="tlhover"></div>
      <div class="tip" id="tltip"></div>
    </div>
    <div class="ticks">${Array.from({ length: 7 }, (_, i) => `<span>${fmtClock((D * i) / 6)}</span>`).join('')}</div>
    ${model.thumbs.length ? `<div class="thumbstrip">${model.thumbs.slice(0, 48).map((th) =>
      `<button data-t="${th.t}" title="${fmtClock(th.t)}" aria-label="Seek to ${fmtClock(th.t)}"><img loading="lazy" src="${esc(runThumbUrl(model.runId, th.name))}" alt=""></button>`).join('')}</div>` : ''}
  </div>`;

  const body = $('#tlbody', root);
  const head = $('#tlhead', root);
  const hover = $('#tlhover', root);
  const tip = $('#tltip', root);

  function drawPossession() {
    const canvas = $('#tlposs', root);
    if (!canvas) return;
    const w = canvas.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, w * dpr);
    canvas.height = 10 * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    poss.forEach((value, i) => {
      const v = +value;
      const x0 = ((i * binS) / D) * w;
      const x1 = Math.min(w, (((i + 1) * binS) / D) * w);
      const team = v >= 50 ? 0 : 1;
      const strength = Math.abs(v - 50) / 50;
      ctx.fillStyle = hexToRgba(model.teams[team].color, 0.25 + strength * 0.75);
      ctx.fillRect(x0, 1, Math.max(1, x1 - x0 - 0.5), 8);
    });
  }

  function drawMomentum() {
    const canvas = $('#tlmom', root);
    if (!canvas) return;
    const w = canvas.clientWidth;
    const h = 30;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, w * dpr);
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.strokeStyle = cssVar('--line-strong');
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    const pts = mom.map((v, i) => [(((i + 0.5) * binS) / D) * w, h / 2 - clamp(+v || 0, -1, 1) * (h / 2 - 2)]);
    for (const [team, sign] of [[0, 1], [1, -1]]) {
      ctx.fillStyle = hexToRgba(model.teams[team].color, 0.55);
      ctx.beginPath();
      ctx.moveTo(pts[0]?.[0] || 0, h / 2);
      for (const [x, y] of pts) ctx.lineTo(x, sign > 0 ? Math.min(y, h / 2) : Math.max(y, h / 2));
      ctx.lineTo(pts[pts.length - 1]?.[0] || 0, h / 2);
      ctx.closePath();
      ctx.fill();
    }
  }

  function redraw() { drawPossession(); drawMomentum(); }

  const timeAt = (clientX) => {
    const rect = body.getBoundingClientRect();
    return clamp((clientX - rect.left) / rect.width, 0, 1) * D;
  };

  function nearestThumb(t) {
    let best = null;
    for (const th of model.thumbs) if (th.t <= t + 5 && (!best || th.t > best.t)) best = th;
    return best;
  }

  function showTip(clientX, event) {
    const rect = body.getBoundingClientRect();
    const x = clamp(clientX - rect.left, 0, rect.width);
    const t = timeAt(clientX);
    hover.style.display = 'block';
    hover.style.left = `${x}px`;
    const th = nearestThumb(t);
    const bin = Math.floor(t / binS);
    let html = th ? `<img src="${esc(runThumbUrl(model.runId, th.name))}" alt="">` : '';
    if (event) {
      html += `<div><b style="color:${event.info.color}">${esc(event.info.label)}</b> · <b>${fmtClock(event.t)}</b></div>
        <div class="faint">${esc(teamName(model, event.team))}${event.player_track_id != null ? ` · ${esc(playerName(model, event.player_track_id))}` : ''}</div>`;
    } else {
      html += `<div><b>${fmtClock(t)}</b> <span class="faint">SRC ${fmtClock(t + model.trim)}</span></div>`;
      if (poss[bin] != null) html += `<div class="faint">Possession ${Math.round(poss[bin])}% – ${Math.round(100 - poss[bin])}%</div>`;
    }
    tip.innerHTML = html;
    tip.style.display = 'block';
    tip.style.left = `${clamp(x, 70, rect.width - 70)}px`;
  }

  function hideTip() { hover.style.display = 'none'; tip.style.display = 'none'; }

  body.addEventListener('pointerdown', (event) => {
    if (event.target.closest('.mk')) return;
    state.dragging = true;
    body.setPointerCapture(event.pointerId);
    onSeek(timeAt(event.clientX), false);
  });
  body.addEventListener('pointermove', (event) => {
    const mk = event.target.closest('.mk');
    showTip(event.clientX, mk ? model.events[+mk.dataset.i] : null);
    if (state.dragging) onSeek(timeAt(event.clientX), false);
  });
  body.addEventListener('pointerup', () => { state.dragging = false; });
  body.addEventListener('pointerleave', hideTip);
  body.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      event.stopPropagation();
      onSeek(clamp(state.t + (event.key === 'ArrowLeft' ? -5 : 5), 0, D), false);
    }
  });
  $$('.mk', root).forEach((mk) => {
    mk.onclick = (event) => { event.stopPropagation(); onEvent(model.events[+mk.dataset.i]); };
  });
  $$('.thumbstrip button', root).forEach((button) => { button.onclick = () => onSeek(+button.dataset.t, false); });

  const ro = new ResizeObserver(redraw);
  ro.observe(body);
  const onTheme = () => redraw();
  window.addEventListener('vh-theme', onTheme);
  redraw();

  return {
    update(t) {
      if (t == null) return;
      state.t = t;
      head.style.left = `${(100 * clamp(t / D, 0, 1)).toFixed(3)}%`;
      body.setAttribute('aria-valuenow', String(Math.round(t)));
      body.setAttribute('aria-valuetext', fmtClock(t));
    },
    setActive(active, note = '') {
      state.hidden = !active;
      $('#tl', root).style.opacity = active ? '1' : '.45';
      head.style.display = active ? '' : 'none';
      $('#tlnote', root).textContent = note;
    },
    // Dim markers that are filtered out of the event list.
    setFilter(visibleIds) {
      $$('.mk', root).forEach((mk) => {
        const ev = model.events[+mk.dataset.i];
        mk.classList.toggle('dim', visibleIds ? !visibleIds.has(ev.id) : false);
      });
    },
    destroy() { ro.disconnect(); window.removeEventListener('vh-theme', onTheme); },
  };
}
