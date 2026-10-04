// "Calibrate pitch" mode on the Watch tab.
//
// Pauses the wide (proxy) view and lets the user click the 4 corners of the
// playing area in order TL, TR, BR, BL (drag to adjust, undo, reset). With 4
// corners the pitch markings are projected onto the frame through a
// client-side homography so the user can see whether the corners are right,
// and the goal mouths the pipeline would derive are drawn (or the user drags
// their own). Everything is kept as normalized [0, 1] coordinates of the
// source frame: saved per run in calibration.json and sent unchanged as the
// job config keys pitch_corners / goal_box_left / goal_box_right on
// "Re-analyze with calibration" (tracking reused, no re-detection).

import { post, put } from '../../api.js';
import { deriveGoalBoxes, isConvexQuad, PITCH_CORNERS_M, pitchPolylines, project, solveHomography } from '../../homography.js';
import { icon } from '../../icons.js';
import { PITCH } from '../../pitch.js';
import { $, esc, fmtRel, toast } from '../../ui.js';

const CORNERS = [
  { key: 'TL', label: 'top-left', hint: 'far-left corner flag' },
  { key: 'TR', label: 'top-right', hint: 'far-right corner flag' },
  { key: 'BR', label: 'bottom-right', hint: 'near-right corner flag' },
  { key: 'BL', label: 'bottom-left', hint: 'near-left corner flag' },
];
const MIN_BOX = 0.004; // normalized; smaller drags are treated as clicks
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const r6 = (v) => Math.round(v * 1e6) / 1e6;

// The rerun payload: exact JobConfig keys the pipeline reads.
export function calibrationOverrides(ctx, doc) {
  return {
    pitch_corners: doc.pitch_corners,
    goal_box_left: doc.goal_box_left || null,
    goal_box_right: doc.goal_box_right || null,
    reuse_tracking_from_job: ctx.jobId,
    output_dir: null,
  };
}

export async function rerunWithCalibration(ctx, doc) {
  if (!ctx.jobId) throw new Error('This run is not linked to a job; open it from its match to re-analyze.');
  return post(`/jobs/${ctx.jobId}/rerun`, {
    config_overrides: calibrationOverrides(ctx, doc),
    reason: 'manual_pitch_calibration',
  });
}

export function createCalibrator(ctx, player, { box, onStart }) {
  const model = ctx.model;
  const overlay = player.overlay;
  const layer = document.createElement('div');
  layer.className = 'layer';
  overlay.appendChild(layer);
  const st = {
    on: false, seeded: false, corners: [], goalMode: 'auto', goals: { left: null, right: null }, drawSide: null,
    history: [], drag: null, dirty: false, busy: false, raf: 0, queued: null,
  };
  const canRerun = () => !!ctx.jobId && model.tracksAvailable;

  const frame = () => ({
    w: model.frame.w || player.video.videoWidth || 1920,
    h: model.frame.h || player.video.videoHeight || 1080,
  });

  // Rendered video rectangle inside the stage (object-fit: contain).
  function videoRect() {
    const v = player.video;
    const sw = player.stage.clientWidth;
    const sh = player.stage.clientHeight;
    const vw = v.videoWidth || frame().w;
    const vh = v.videoHeight || frame().h;
    const scale = Math.min(sw / vw, sh / vh);
    return { x: (sw - vw * scale) / 2, y: (sh - vh * scale) / 2, w: vw * scale, h: vh * scale, sw, sh };
  }

  function toNorm(event) {
    const rect = player.stage.getBoundingClientRect();
    const r = videoRect();
    return [clamp01((event.clientX - rect.left - r.x) / r.w), clamp01((event.clientY - rect.top - r.y) / r.h)];
  }

  const valid = () => st.corners.length === 4 && isConvexQuad(st.corners);

  function autoGoals() {
    if (!valid()) return { left: null, right: null };
    const f = frame();
    const boxes = deriveGoalBoxes(st.corners.map(([x, y]) => [x * f.w, y * f.h]), f.w);
    const norm = (b) => (b ? { x1: clamp01(b.x1 / f.w), y1: clamp01(b.y1 / f.h), x2: clamp01(b.x2 / f.w), y2: clamp01(b.y2 / f.h) } : null);
    return { left: norm(boxes.left), right: norm(boxes.right) };
  }

  function currentGoals() {
    return st.goalMode === 'manual' ? { ...st.goals } : autoGoals();
  }

  function payload() {
    const goals = currentGoals();
    const boxOut = (b) => (b ? { x1: r6(Math.min(b.x1, b.x2)), y1: r6(Math.min(b.y1, b.y2)), x2: r6(Math.max(b.x1, b.x2)), y2: r6(Math.max(b.y1, b.y2)), normalized: true } : null);
    const f = frame();
    return {
      pitch_corners: st.corners.map(([x, y]) => [r6(x), r6(y)]),
      // Auto goal mouths are not pinned: the pipeline derives the same boxes
      // from the corners (and null clears any boxes in the parent config).
      goal_box_left: st.goalMode === 'manual' ? boxOut(goals.left) : null,
      goal_box_right: st.goalMode === 'manual' ? boxOut(goals.right) : null,
      frame_width: f.w,
      frame_height: f.h,
      normalized: true,
    };
  }

  /* ---------------- state helpers ---------------- */

  function snapshot() {
    st.history.push(JSON.stringify({ corners: st.corners, goals: st.goals, goalMode: st.goalMode, drawSide: st.drawSide }));
    if (st.history.length > 200) st.history.shift();
  }

  function undo() {
    const last = st.history.pop();
    if (!last) return;
    Object.assign(st, JSON.parse(last));
    st.dirty = true;
    render();
  }

  function reset() {
    if (!st.corners.length && !st.goals.left && !st.goals.right) return;
    snapshot();
    st.corners = [];
    st.goals = { left: null, right: null };
    st.drawSide = null;
    st.dirty = true;
    render();
  }

  function seed() {
    if (st.seeded) return;
    st.seeded = true;
    const saved = model.manualCalibration;
    if (saved?.pitch_corners?.length === 4) {
      st.corners = saved.pitch_corners.map(([x, y]) => [+x, +y]);
      const strip = (b) => (b ? { x1: +b.x1, y1: +b.y1, x2: +b.x2, y2: +b.y2 } : null);
      st.goals = { left: strip(saved.goal_box_left), right: strip(saved.goal_box_right) };
      st.goalMode = st.goals.left || st.goals.right ? 'manual' : 'auto';
    }
  }

  function analysisCorners() {
    const cal = model.calibration;
    const f = frame();
    if (cal?.image_corners_px?.length !== 4 || !f.w || !f.h) return null;
    return cal.image_corners_px.map(([x, y]) => [clamp01(+x / f.w), clamp01(+y / f.h)]);
  }

  function setGoalMode(mode) {
    if (mode === st.goalMode) return;
    snapshot();
    st.goalMode = mode;
    if (mode === 'manual' && !st.goals.left && !st.goals.right) {
      // Start from the derived boxes; the user moves or redraws them.
      st.goals = autoGoals();
      st.drawSide = st.goals.left ? null : 'left';
    }
    if (mode === 'auto') st.drawSide = null;
    st.dirty = true;
    render();
  }

  /* ---------------- drawing ---------------- */

  function pathD(points) {
    let d = '';
    let pen = false;
    for (const p of points) {
      if (!p) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`;
      pen = true;
    }
    return d;
  }

  function hintText() {
    const n = st.corners.length;
    if (n < 4) {
      const c = CORNERS[n];
      return `Click corner <b>${n + 1} of 4</b> · ${c.key} — ${c.label} (${c.hint})`;
    }
    if (!valid()) return 'Corners cross — keep the order TL, TR, BR, BL. Drag a corner or press Z to undo';
    if (st.goalMode === 'manual' && st.drawSide) return `Drag a box over the <b>${st.drawSide}</b> goal mouth`;
    return 'Drag corners until the projected lines sit on the painted lines';
  }

  function drawOverlay() {
    if (!st.on) { layer.innerHTML = ''; overlay.classList.remove('calibrating'); return; }
    overlay.classList.add('calibrating');
    const r = videoRect();
    const D = ([x, y]) => [r.x + x * r.w, r.y + y * r.h];
    let svg = '';
    const n = st.corners.length;
    if (valid()) {
      const H = solveHomography(PITCH_CORNERS_M, st.corners.map(D));
      if (H) {
        const { outline, lines, spots } = pitchPolylines();
        const proj = (poly) => pathD(poly.map(([x, y]) => project(H, x, y)));
        const d = [...outline, ...lines].map(proj).join('');
        svg += `<path class="pl-halo" d="${d}"/><path class="pl" d="${d}"/>`;
        svg += spots.map(([x, y]) => project(H, x, y)).filter(Boolean)
          .map(([x, y]) => `<circle class="spot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5"/>`).join('');
      }
    }
    if (n >= 2) {
      const pts = st.corners.map(D).map((p) => p.map((v) => v.toFixed(1)).join(',')).join(' ');
      svg += n === 4 ? `<polygon class="quad ${valid() ? '' : 'bad'}" points="${pts}"/>` : `<polyline class="quad open" points="${pts}"/>`;
    }
    const goals = currentGoals();
    for (const side of ['left', 'right']) {
      const b = goals[side];
      if (!b) continue;
      const [x1, y1] = D([Math.min(b.x1, b.x2), Math.min(b.y1, b.y2)]);
      const [x2, y2] = D([Math.max(b.x1, b.x2), Math.max(b.y1, b.y2)]);
      const auto = st.goalMode !== 'manual';
      const tx = side === 'left' ? x1 : x2;
      svg += `<g class="goal ${side} ${auto ? 'auto' : ''} ${st.drawSide === side && st.drag?.kind === 'draw' ? 'drawing' : ''}" data-goal="${side}">
        <rect x="${x1.toFixed(1)}" y="${y1.toFixed(1)}" width="${Math.max(1, x2 - x1).toFixed(1)}" height="${Math.max(1, y2 - y1).toFixed(1)}" rx="2"/>
        <text x="${tx.toFixed(1)}" y="${(y1 - 6).toFixed(1)}" text-anchor="${side === 'left' ? 'start' : 'end'}">${side === 'left' ? 'Left' : 'Right'} goal${auto ? ' · auto' : ''}</text></g>`;
    }
    st.corners.forEach((c, i) => {
      const [x, y] = D(c);
      const dx = c[0] > 0.5 ? -16 : 16;
      const dy = c[1] > 0.5 ? 22 : -16;
      svg += `<g class="ch ${st.drag?.kind === 'corner' && st.drag.i === i ? 'drag' : ''}" data-corner="${i}" role="button" aria-label="Corner ${CORNERS[i].key}">
        <circle class="hit" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="16"/>
        <circle class="dot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="9"/>
        <circle class="pin" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.6"/>
        <text x="${(x + dx).toFixed(1)}" y="${(y + dy).toFixed(1)}" text-anchor="${dx < 0 ? 'end' : 'start'}">${i + 1} · ${CORNERS[i].key}</text></g>`;
    });
    layer.innerHTML = `<svg class="calsvg" width="${r.sw}" height="${r.sh}" viewBox="0 0 ${r.sw} ${r.sh}" aria-hidden="true">${svg}</svg>
      <div class="pickhint calhint">${icon('pitch', 'sm')}<span>${hintText()}</span><kbd>Esc</kbd></div>`;
  }

  function miniPitch() {
    const n = st.corners.length;
    const { L, W } = PITCH;
    const dots = PITCH_CORNERS_M.map(([x, y], i) => `<g class="cd ${i < n ? 'done' : i === n ? 'next' : ''}">
      <circle cx="${x}" cy="${y}" r="5.5"/><text x="${x}" y="${y + 2.4}" text-anchor="middle">${i + 1}</text></g>`).join('');
    return `<svg viewBox="-9 -9 ${L + 18} ${W + 18}" role="img" aria-label="Corner order: top-left, top-right, bottom-right, bottom-left">
      <rect x="0" y="0" width="${L}" height="${W}" class="mp"/><line x1="${L / 2}" y1="0" x2="${L / 2}" y2="${W}" class="mpl"/>
      <circle cx="${L / 2}" cy="${W / 2}" r="9.15" class="mpl"/>
      <rect x="0" y="${(W - 40.32) / 2}" width="16.5" height="40.32" class="mpl"/><rect x="${L - 16.5}" y="${(W - 40.32) / 2}" width="16.5" height="40.32" class="mpl"/>
      ${dots}</svg>`;
  }

  function stepState() {
    if (st.corners.length < 4 || !valid()) return 0;
    if (st.goalMode === 'manual' && (st.drawSide || !st.goals.left || !st.goals.right)) return 1;
    return 2;
  }

  function renderPanel() {
    if (!st.on) { box.hidden = true; box.innerHTML = ''; return; }
    box.hidden = false;
    const n = st.corners.length;
    const ok = valid();
    const step = stepState();
    const saved = model.manualCalibration;
    const cal = model.calibration;
    const auto = analysisCorners();
    const steps = ['Corners', 'Goal mouths', 'Apply'].map((label, i) =>
      `<span class="calstep ${i < step ? 'done' : i === step ? 'cur' : ''}">${i < step ? icon('check', 'sm') : `<b>${i + 1}</b>`} ${label}${i === 0 ? ` <span class="faint">${n}/4</span>` : ''}</span>`).join('');
    let status = '';
    if (n === 4 && !ok) status = `<div class="errnote">${icon('alert', 'sm')}<span>The corners cross or are collinear. Click them in order TL, TR, BR, BL.</span></div>`;
    else if (ok && st.goalMode === 'manual' && (!st.goals.left || !st.goals.right)) status = `<div class="warnnote">${icon('info', 'sm')}<span>A missing goal box is estimated by the pipeline.</span></div>`;
    else if (ok) status = `<div class="oknote">${icon('ok', 'sm')}<span>Looks consistent. The white lines should sit on the painted lines; drag a corner if they drift.</span></div>`;
    const savedNote = saved ? `Saved ${esc(fmtRel(saved.updated_at))}${st.dirty ? ' · unsaved changes' : ''}` : (st.dirty ? 'Not saved yet' : 'No manual calibration saved');
    const analysisNote = cal ? `Analysis used ${esc(cal.source || 'auto')} calibration (${Math.round((+cal.confidence || 0) * 100)}%)` : 'Analysis had no pitch calibration';
    box.innerHTML = `<div class="panel calpanel">
      <div class="calhead"><div class="panel-title" style="margin:0">${icon('pitch', 'sm')} Calibrate pitch</div>
        <div class="calsteps">${steps}</div>
        <button class="iconbtn" data-cal="close" title="Leave calibration (Esc)" aria-label="Leave calibration">${icon('x', 'sm')}</button></div>
      <div class="calbody">
        <div class="calmini">${miniPitch()}</div>
        <div class="calctl">
          <div class="small muted">Click where each touchline meets the goal line, in order
            <b>top-left → top-right → bottom-right → bottom-left</b>, then drag to fine-tune.</div>
          <div class="calrow">
            <button class="btn2 sm" data-cal="undo" ${st.history.length ? '' : 'disabled'} title="Undo (Z)">${icon('rerun', 'sm')} Undo <kbd>Z</kbd></button>
            <button class="btn2 sm" data-cal="reset" ${n || st.goals.left || st.goals.right ? '' : 'disabled'}>${icon('trash', 'sm')} Reset</button>
            ${auto ? `<button class="btn-ghost sm" data-cal="auto" title="Start from the corners the analysis estimated">${icon('sparkle', 'sm')} Start from ${esc(cal.source || 'auto')} estimate</button>` : ''}
            <span class="calsep" aria-hidden="true"></span>
            <span class="small" style="font-weight:600">Goal mouths</span>
            <div class="seg" role="group" aria-label="Goal mouths">
              <button data-gm="auto" aria-pressed="${st.goalMode === 'auto'}">Auto</button>
              <button data-gm="manual" aria-pressed="${st.goalMode === 'manual'}">Draw</button>
            </div>
            ${st.goalMode === 'manual' ? `<button class="btn-ghost sm ${st.drawSide === 'left' ? 'on' : ''}" data-draw="left" ${ok ? '' : 'disabled'}>${icon('plus', 'sm')} ${st.goals.left ? 'Redraw' : 'Draw'} left</button>
              <button class="btn-ghost sm ${st.drawSide === 'right' ? 'on' : ''}" data-draw="right" ${ok ? '' : 'disabled'}>${icon('plus', 'sm')} ${st.goals.right ? 'Redraw' : 'Draw'} right</button>` : ''}
          </div>
          ${status}
          <div class="calrow apply">
            <button class="btn2" data-cal="save" ${ok && !st.busy ? '' : 'disabled'}>${icon('check', 'sm')} Save calibration</button>
            <button class="btn" data-cal="rerun" ${ok && !st.busy && canRerun() && !(st.queued && !st.dirty) ? '' : 'disabled'}
              title="${canRerun() ? 'Re-run analysis (stats, events, goals) with these corners; tracking is reused' : 'Needs a run linked to a job with tracks.npz'}">${icon('rerun', 'sm')} Re-analyze with calibration</button>
            <span class="faint xs">${savedNote} · ${analysisNote}</span>
          </div>
          ${st.queued ? `<div class="panel queued" id="calqueued"><b>Queued.</b>
            <span class="muted small">Job ${esc(st.queued)} re-runs the analysis (stats, events, goal detection) with your pitch corners, reusing this run's tracking — no re-detection.</span>
            <div style="margin-top:8px"><a href="#jobs/${esc(st.queued)}">Follow progress on the Jobs page →</a></div></div>` : ''}
        </div>
      </div></div>`;
  }

  function render() {
    drawOverlay();
    if (!st.drag) renderPanel();
  }

  const drawSoon = () => {
    if (st.raf) return;
    st.raf = requestAnimationFrame(() => { st.raf = 0; drawOverlay(); });
  };

  /* ---------------- pointer interaction ---------------- */

  overlay.addEventListener('pointerdown', (event) => {
    if (!st.on || event.button !== 0) return;
    const p = toNorm(event);
    const handle = event.target.closest('[data-corner]');
    const goal = event.target.closest('[data-goal]');
    if (handle) {
      snapshot();
      st.drag = { kind: 'corner', i: +handle.dataset.corner };
    } else if (st.goalMode === 'manual' && st.drawSide && valid()) {
      snapshot();
      st.drag = { kind: 'draw', side: st.drawSide };
      st.goals[st.drawSide] = { x1: p[0], y1: p[1], x2: p[0], y2: p[1] };
    } else if (goal && st.goalMode === 'manual' && st.goals[goal.dataset.goal]) {
      snapshot();
      st.drag = { kind: 'move', side: goal.dataset.goal, start: p, orig: { ...st.goals[goal.dataset.goal] } };
    } else if (st.corners.length < 4) {
      snapshot();
      st.corners.push(p);
      st.drag = { kind: 'corner', i: st.corners.length - 1 };
    } else {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    overlay.setPointerCapture?.(event.pointerId);
    st.dirty = true;
    drawOverlay();
  });

  overlay.addEventListener('pointermove', (event) => {
    if (!st.drag) return;
    const p = toNorm(event);
    const d = st.drag;
    if (d.kind === 'corner') st.corners[d.i] = p;
    if (d.kind === 'draw') Object.assign(st.goals[d.side], { x2: p[0], y2: p[1] });
    if (d.kind === 'move') {
      const o = d.orig;
      const w = Math.abs(o.x2 - o.x1);
      const h = Math.abs(o.y2 - o.y1);
      const x1 = Math.min(Math.max(0, Math.min(o.x1, o.x2) + p[0] - d.start[0]), 1 - w);
      const y1 = Math.min(Math.max(0, Math.min(o.y1, o.y2) + p[1] - d.start[1]), 1 - h);
      st.goals[d.side] = { x1, y1, x2: x1 + w, y2: y1 + h };
    }
    drawSoon();
  });

  const endDrag = () => {
    const d = st.drag;
    if (!d) return;
    st.drag = null;
    if (d.kind === 'draw') {
      const b = st.goals[d.side];
      if (Math.abs(b.x2 - b.x1) < MIN_BOX || Math.abs(b.y2 - b.y1) < MIN_BOX) {
        undo(); // a click, not a box
        st.drawSide = d.side;
      } else {
        st.goals[d.side] = { x1: Math.min(b.x1, b.x2), y1: Math.min(b.y1, b.y2), x2: Math.max(b.x1, b.x2), y2: Math.max(b.y1, b.y2) };
        st.drawSide = d.side === 'left' && !st.goals.right ? 'right' : null;
      }
    }
    render();
  };
  overlay.addEventListener('pointerup', endDrag);
  overlay.addEventListener('pointercancel', endDrag);

  /* ---------------- panel actions ---------------- */

  async function save() {
    if (!valid()) return null;
    st.busy = true;
    renderPanel();
    try {
      const result = await put(`/studio/runs/${encodeURIComponent(ctx.runId)}/calibration`, payload());
      ctx.run.calibration = result.calibration;
      model.manualCalibration = result.calibration;
      st.dirty = false;
      return result.calibration;
    } finally {
      st.busy = false;
      renderPanel();
    }
  }

  async function reanalyze() {
    const doc = st.dirty || !model.manualCalibration ? await save() : model.manualCalibration;
    if (!doc) return;
    st.busy = true;
    renderPanel();
    try {
      const job = await rerunWithCalibration(ctx, doc);
      st.queued = job.job_id;
      toast('Re-analysis queued — tracking is reused, no re-detection');
    } finally {
      st.busy = false;
      renderPanel();
    }
  }

  box.addEventListener('click', async (event) => {
    const gm = event.target.closest('[data-gm]');
    if (gm) { setGoalMode(gm.dataset.gm); return; }
    const draw = event.target.closest('[data-draw]');
    if (draw && !draw.disabled) {
      st.drawSide = st.drawSide === draw.dataset.draw ? null : draw.dataset.draw;
      render();
      return;
    }
    const action = event.target.closest('[data-cal]')?.dataset.cal;
    if (!action) return;
    try {
      if (action === 'close') stop();
      if (action === 'undo') undo();
      if (action === 'reset') reset();
      if (action === 'auto') {
        snapshot();
        st.corners = analysisCorners() || st.corners;
        st.dirty = true;
        render();
      }
      if (action === 'save') { await save(); toast('Calibration saved for this run'); }
      if (action === 'rerun') await reanalyze();
    } catch (error) {
      toast(error.message, 'err');
    }
  });

  /* ---------------- lifecycle ---------------- */

  function start() {
    const wide = model.sources.find((s) => s.wide);
    if (!wide) { toast('Pitch calibration needs the wide view (proxy video), which this run does not have.', 'warn'); return; }
    onStart?.();
    seed();
    st.on = true;
    player.pause();
    if (player.source?.key !== wide.key) player.setSource(wide.key);
    btn?.classList.add('on');
    btn?.setAttribute('aria-pressed', 'true');
    render();
  }

  function stop() {
    if (!st.on) return;
    st.on = false;
    st.drag = null;
    st.queued = null;
    render();
    btn?.classList.remove('on');
    btn?.setAttribute('aria-pressed', 'false');
  }

  const ro = new ResizeObserver(() => { if (st.on) drawOverlay(); });
  ro.observe(player.stage);
  player.on('meta', () => { if (st.on) drawOverlay(); });

  player.extra.insertAdjacentHTML('beforeend', `<button class="btn2 sm" id="calbtn" aria-pressed="false"
    ${model.sources.some((s) => s.wide) ? 'title="Click the pitch corners for accurate metres and goal detection"' : 'disabled title="No wide view in this run"'}>${icon('pitch', 'sm')} Calibrate pitch <kbd>C</kbd></button>`);
  const btn = $('#calbtn', player.extra);
  btn.onclick = () => (st.on ? stop() : start());

  return {
    get active() { return st.on; },
    start,
    stop,
    undo,
    toggle() { return st.on ? stop() : start(); },
    destroy() {
      ro.disconnect();
      cancelAnimationFrame(st.raf);
      layer.remove();
    },
  };
}
