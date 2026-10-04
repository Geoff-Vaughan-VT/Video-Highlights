// Watch tab: sticky player + timeline on the left, event list on the right,
// player picker overlay, and keyboard shortcuts.

import { icon } from '../../icons.js';
import { $, esc, openDrawer } from '../../ui.js';
import { createCalibrator } from './calibrate.js';
import { createEventList } from './events.js';
import { createPicker } from './picker.js';
import { createPlayer } from './player.js';
import { createTimeline } from './timeline.js';

const PREROLL = 3;

const SHORTCUTS = [
  ['Space / K', 'Play / pause'], ['J / ←', 'Back 5 s (Shift: 15 s)'], ['L / →', 'Forward 5 s (Shift: 15 s)'],
  [', / .', 'Previous / next frame'], ['N / P', 'Next / previous event'], ['1 – 9', 'Jump to event 1–9 in the list'],
  ['T', 'Pick a player'], ['C', 'Calibrate pitch (click the 4 corners)'], ['Z', 'Undo last corner (calibration)'], ['G / W', 'Game camera / wide view'], ['M', 'Mute'], ['F', 'Fullscreen'],
  ['[ / ]', 'Slower / faster'], ['Esc', 'Leave pick / calibration mode'], ['?', 'This sheet'],
];

function showShortcuts() {
  openDrawer({
    title: `${icon('keyboard')} <b>Keyboard shortcuts</b>`,
    html: `<div class="shortcuts">${SHORTCUTS.map(([k, d]) => `<div>${k.split(' / ').map((x) => `<kbd>${esc(x)}</kbd>`).join(' ')}</div><div class="muted">${esc(d)}</div>`).join('')}</div>`,
  });
}

export function renderWatchTab(body, ctx, options = {}) {
  const model = ctx.model;
  body.innerHTML = `<div class="watch">
    <div class="stage-col"><div id="playerbox"></div><div id="calbox" hidden></div><div id="tlbox"></div></div>
    <div id="evbox"></div></div>`;

  let timeline = null;
  let picker = null;
  let calib = null;
  const player = createPlayer($('#playerbox', body), model, { initial: ctx.lastSource });
  const events = createEventList($('#evbox', body), ctx, {
    onPlay: (e) => { picker?.stop(); player.seekWindow(e.t_start != null ? e.t_start + PREROLL : e.t, { play: true, preroll: PREROLL }); },
    onFilter: (ids) => timeline?.setFilter(ids),
  });
  timeline = createTimeline($('#tlbox', body), model, {
    onSeek: (t) => player.seekWindow(t),
    onEvent: (e) => { picker.stop(); player.seekWindow(e.t_start != null ? e.t_start + PREROLL : e.t, { play: true, preroll: PREROLL }); },
  });
  timeline.setFilter(new Set(events.visible.map((e) => e.id)));
  picker = createPicker(ctx, player, { onFilterEvents: (tid) => events.setPlayerFilter(tid), onStart: () => calib?.stop() });
  calib = createCalibrator(ctx, player, { box: $('#calbox', body), onStart: () => picker.stop() });
  player.extra.insertAdjacentHTML('beforeend',
    `<button class="iconbtn" id="kbdhelp" title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">${icon('keyboard')}</button>`);
  $('#kbdhelp', player.extra).onclick = showShortcuts;

  player.on('time', (t) => { timeline.update(t); events.update(t); });
  player.on('source', (src) => {
    ctx.lastSource = src.key;
    timeline.setActive(src.timebase === 'window', src.timebase === 'window' ? '' : 'Reel/clip has its own clock — markers apply to Game camera and Wide');
    if (src.timebase !== 'window') picker.stop();
    if (!src.wide) calib.stop();
  });
  if (player.source) timeline.setActive(player.source.timebase === 'window');

  const step = (dir) => {
    const t = player.windowTime() ?? 0;
    const list = events.visible;
    const target = dir > 0 ? list.find((e) => e.t > t + 1) : [...list].reverse().find((e) => e.t < t - PREROLL - 1);
    if (target) player.seekWindow(target.t, { play: true, preroll: PREROLL });
  };
  body.addEventListener('vh-event-nav', (event) => step(event.detail === 'next' ? 1 : -1));

  const onKey = (event) => {
    if (event.target.closest('input, select, textarea, [contenteditable]') || event.metaKey || event.ctrlKey || event.altKey) return;
    if (document.querySelector('.scrim')) return;
    const k = event.key;
    if (calib.active && ['z', 'Z', 'Backspace'].includes(k)) { event.preventDefault(); calib.undo(); return; }
    const big = event.shiftKey ? 15 : 5;
    const handled = {
      ' ': () => player.toggle(), k: () => player.toggle(), K: () => player.toggle(),
      j: () => player.nudge(-big), J: () => player.nudge(-big), l: () => player.nudge(big), L: () => player.nudge(big),
      ArrowLeft: () => player.nudge(-big), ArrowRight: () => player.nudge(big),
      ',': () => player.step(-1), '.': () => player.step(1), '<': () => player.step(-1), '>': () => player.step(1),
      n: () => step(1), N: () => step(1), p: () => step(-1), P: () => step(-1),
      t: () => picker.toggle(), T: () => picker.toggle(),
      c: () => calib.toggle(), C: () => calib.toggle(),
      m: () => player.toggleMute(), M: () => player.toggleMute(), f: () => player.fullscreen(), F: () => player.fullscreen(),
      g: () => player.setSource('movie'), G: () => player.setSource('movie'), w: () => player.setSource('wide'), W: () => player.setSource('wide'),
      '[': () => player.setRate(Math.max(0.25, player.video.playbackRate / 2)),
      ']': () => player.setRate(Math.min(2, player.video.playbackRate * 2)),
      '?': showShortcuts,
      Escape: () => { picker.stop(); calib.stop(); },
    }[k];
    if (handled) { event.preventDefault(); handled(); return; }
    if (/^[1-9]$/.test(k)) {
      const e = events.visible[+k - 1];
      if (e) { event.preventDefault(); player.seekWindow(e.t, { play: true, preroll: PREROLL }); }
    }
  };
  document.addEventListener('keydown', onKey);

  const onLabels = () => events.refresh();
  window.addEventListener('vh-labels-changed', onLabels);

  if (options.calibrate) calib.start();

  if (options.seek != null) {
    player.video.addEventListener('loadedmetadata', () => player.seekWindow(options.seek, { play: true, preroll: PREROLL }), { once: true });
  }

  return () => {
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('vh-labels-changed', onLabels);
    picker.destroy();
    calib.destroy();
    timeline.destroy();
    player.destroy();
  };
}
