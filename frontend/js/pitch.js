// Pitch SVG (metres, 105 x 68), heatmaps and event maps.

import { esc } from './ui.js';

export const PITCH = { L: 105, W: 68 };

// Pitch outline + markings. `inner` is SVG content in pitch metres.
export function pitchSvg(inner = '', { label = 'Pitch', note = '' } = {}) {
  const { L, W } = PITCH;
  const line = 'stroke="var(--pitch-line)" stroke-width="0.35" fill="none"';
  const box = (x, w, h) => `<rect x="${x}" y="${(W - h) / 2}" width="${w}" height="${h}" ${line}/>`;
  return `<div class="pitch"><svg viewBox="-3 -3 ${L + 6} ${W + 6}" role="img" aria-label="${esc(label)}">
    <rect x="-3" y="-3" width="${L + 6}" height="${W + 6}" rx="2" fill="var(--pitch)"/>
    ${Array.from({ length: 10 }, (_, i) => i % 2 ? `<rect x="${i * L / 10}" y="0" width="${L / 10}" height="${W}" fill="rgba(255,255,255,.025)"/>` : '').join('')}
    <g>${inner}</g>
    <rect x="0" y="0" width="${L}" height="${W}" ${line}/>
    <line x1="${L / 2}" y1="0" x2="${L / 2}" y2="${W}" ${line}/>
    <circle cx="${L / 2}" cy="${W / 2}" r="9.15" ${line}/>
    <circle cx="${L / 2}" cy="${W / 2}" r="0.5" fill="var(--pitch-line)"/>
    ${box(0, 16.5, 40.3)}${box(L - 16.5, 16.5, 40.3)}${box(0, 5.5, 18.3)}${box(L - 5.5, 5.5, 18.3)}
    <rect x="-1.6" y="${(W - 7.32) / 2}" width="1.6" height="7.32" ${line}/>
    <rect x="${L}" y="${(W - 7.32) / 2}" width="1.6" height="7.32" ${line}/>
    <circle cx="11" cy="${W / 2}" r="0.45" fill="var(--pitch-line)"/><circle cx="${L - 11}" cy="${W / 2}" r="0.45" fill="var(--pitch-line)"/>
    ${note ? `<text x="${L / 2}" y="${W + 2.4}" text-anchor="middle" style="fill:var(--pitch-line);font-size:2.2px">${esc(note)}</text>` : ''}
  </svg></div>`;
}

// Heatmap grid -> rects. Accepts grid[bins_y][bins_x] or the transpose.
export function heatmapLayer(heatmap, color = '#c6f04a') {
  if (!heatmap?.grid?.length) return '';
  let grid = heatmap.grid;
  const bx = +heatmap.bins_x || grid[0].length;
  const by = +heatmap.bins_y || grid.length;
  if (grid.length === bx && grid[0].length === by && bx !== by) grid = grid[0].map((_, i) => grid.map((row) => row[i]));
  const max = Math.max(...grid.flat().map(Number), 1e-9);
  const cw = PITCH.L / bx;
  const ch = PITCH.W / by;
  let out = '';
  grid.forEach((row, y) => row.forEach((value, x) => {
    const v = (+value || 0) / max;
    if (v < 0.04) return;
    out += `<rect x="${(x * cw).toFixed(2)}" y="${(y * ch).toFixed(2)}" width="${cw.toFixed(2)}" height="${ch.toFixed(2)}"
      fill="${esc(color)}" opacity="${(0.12 + v * 0.78).toFixed(2)}" rx="0.6"/>`;
  }));
  return `<g style="filter:blur(1.2px)">${out}</g>`;
}

export function applyHomography(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (!w) return null;
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

// Event location in pitch metres, or null. `mode` reports how it was found.
export function eventLocation(event, calibration, frame) {
  const ev = event.evidence || {};
  const metres = ev.pitch_xy_m || ev.location_m || (ev.x_m != null && ev.y_m != null ? [ev.x_m, ev.y_m] : null);
  if (metres) return { xy: metres.map(Number), mode: 'metres' };
  const px = ev.ball_xy || ev.location_px || (ev.ball_x != null && ev.ball_y != null ? [ev.ball_x, ev.ball_y] : null);
  if (!px) return null;
  const H = calibration?.homography;
  if (Array.isArray(H) && H.length === 3) {
    const xy = applyHomography(H, +px[0], +px[1]);
    if (xy && Number.isFinite(xy[0]) && Number.isFinite(xy[1])) return { xy, mode: 'calibrated' };
  }
  if (frame?.w && frame?.h) return { xy: [(+px[0] / frame.w) * PITCH.L, (+px[1] / frame.h) * PITCH.W], mode: 'image' };
  return null;
}
