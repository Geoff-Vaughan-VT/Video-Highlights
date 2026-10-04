// Hand-written SVG charts (no runtime dependencies). Each returns markup
// sized by a viewBox so it scales with its container.

import { esc } from './ui.js';

const W = 640;

// Axis label for a time in seconds: minutes (12') for long spans, mm:ss for short clips.
export function timeTick(totalSeconds) {
  return (v) => (totalSeconds >= 600 ? `${Math.round(v / 60)}'` : `${Math.floor(v / 60)}:${String(Math.round(v % 60)).padStart(2, '0')}`);
}

function ticksFor(max, count = 4) {
  if (!(max > 0)) return [0];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

// Donut for two shares (e.g. possession). values: [a, b].
export function donut({ values, colors, size = 140, label = '', sublabel = '' }) {
  const total = values.reduce((s, v) => s + (+v || 0), 0) || 1;
  const r = 52;
  const c = 2 * Math.PI * r;
  let offset = 0;
  const arcs = values.map((value, i) => {
    const len = ((+value || 0) / total) * c;
    const gap = values.length > 1 ? 2 : 0;
    const arc = `<circle r="${r}" cx="70" cy="70" fill="none" stroke="${esc(colors[i])}" stroke-width="16"
      stroke-dasharray="${Math.max(0, len - gap)} ${c}" stroke-dashoffset="${-offset}" transform="rotate(-90 70 70)"/>`;
    offset += len;
    return arc;
  }).join('');
  return `<svg class="chart" viewBox="0 0 140 140" width="${size}" height="${size}" role="img" aria-label="${esc(label)} ${esc(sublabel)}">
    <circle r="${r}" cx="70" cy="70" fill="none" stroke="var(--surface-3)" stroke-width="16"/>${arcs}
    <text x="70" y="68" text-anchor="middle" style="fill:var(--text);font-size:22px;font-weight:700">${esc(label)}</text>
    <text x="70" y="88" text-anchor="middle" style="font-size:11px">${esc(sublabel)}</text></svg>`;
}

// Line / area chart. series: [{values:[{x,y}], color, fill?, width?}]
export function lineChart({ series, height = 180, xMax, yMin = 0, yMax, xFmt = (v) => v, yFmt = (v) => v,
  zero = false, markers = [], label = 'chart', bands = [] }) {
  const pad = { l: 34, r: 8, t: 8, b: 20 };
  const all = series.flatMap((s) => s.values);
  const xmax = xMax ?? Math.max(1, ...all.map((p) => p.x));
  const ymax = yMax ?? Math.max(1e-6, ...all.map((p) => p.y)) * 1.1;
  const ymin = yMin;
  const sx = (x) => pad.l + (x / xmax) * (W - pad.l - pad.r);
  const sy = (y) => pad.t + (1 - (y - ymin) / (ymax - ymin || 1)) * (height - pad.t - pad.b);
  const yt = ymin < 0 ? [ymin, ymin / 2, 0, ymax / 2, ymax] : ticksFor(ymax);
  const grid = yt.map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${sy(v)}" y2="${sy(v)}"/>
    <text x="${pad.l - 6}" y="${sy(v) + 3}" text-anchor="end">${esc(yFmt(v))}</text>`).join('');
  const xt = ticksFor(xmax, 6).map((v) => `<text x="${sx(v)}" y="${height - 4}" text-anchor="middle">${esc(xFmt(v))}</text>`).join('');
  const bandSvg = bands.map((b) => `<rect x="${sx(b.x0)}" y="${pad.t}" width="${Math.max(1, sx(b.x1) - sx(b.x0))}"
    height="${height - pad.t - pad.b}" fill="${esc(b.color)}" opacity="${b.opacity ?? 0.12}"/>`).join('');
  const paths = series.map((s) => {
    if (!s.values.length) return '';
    const d = s.values.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join('');
    const base = sy(Math.max(ymin, 0));
    const area = s.fill ? `<path d="${d}L${sx(s.values[s.values.length - 1].x).toFixed(1)},${base}L${sx(s.values[0].x).toFixed(1)},${base}Z"
      fill="${esc(s.color)}" opacity="${s.fillOpacity ?? 0.15}"/>` : '';
    return `${area}<path d="${d}" fill="none" stroke="${esc(s.color)}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join('');
  const zeroLine = zero ? `<line class="axis" x1="${pad.l}" x2="${W - pad.r}" y1="${sy(0)}" y2="${sy(0)}"/>` : '';
  const marks = markers.map((m) => `<g><line x1="${sx(m.x)}" x2="${sx(m.x)}" y1="${pad.t}" y2="${height - pad.b}"
      stroke="${esc(m.color)}" stroke-dasharray="3 3" opacity=".8"/>
    <circle cx="${sx(m.x)}" cy="${pad.t + 4}" r="4" fill="${esc(m.color)}"><title>${esc(m.title || '')}</title></circle></g>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${height}" role="img" aria-label="${esc(label)}">
    <g class="grid">${grid}</g>${bandSvg}${zeroLine}${paths}${marks}${xt}</svg>`;
}

// Possession per bin as bars diverging from 50%: up = team 0, down = team 1.
export function possessionBars({ values, binS, colors, height = 170 }) {
  const pad = { l: 34, r: 8, t: 8, b: 20 };
  const n = values.length || 1;
  const bw = (W - pad.l - pad.r) / n;
  const mid = pad.t + (height - pad.t - pad.b) / 2;
  const half = (height - pad.t - pad.b) / 2;
  const bars = values.map((v, i) => {
    const dv = ((+v || 50) - 50) / 50;
    const h = Math.abs(dv) * half;
    const x = pad.l + i * bw + 1;
    const y = dv >= 0 ? mid - h : mid;
    const color = dv >= 0 ? colors[0] : colors[1];
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, bw - 2).toFixed(1)}" height="${Math.max(1, h).toFixed(1)}"
      rx="2" fill="${esc(color)}"><title>${timeTick(values.length * binS)(i * binS)} · ${Math.round(+v)}% / ${Math.round(100 - v)}%</title></rect>`;
  }).join('');
  const total = n * binS;
  const fmt = timeTick(total);
  const labels = ticksFor(total, 6).map((sec) => {
    const x = pad.l + (sec / binS) * bw;
    return `<text x="${x}" y="${height - 4}" text-anchor="middle">${fmt(sec)}</text>`;
  }).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${height}" role="img" aria-label="Possession by period">
    <g class="grid"><line x1="${pad.l}" x2="${W - pad.r}" y1="${pad.t}" y2="${pad.t}"/><line x1="${pad.l}" x2="${W - pad.r}" y1="${height - pad.b}" y2="${height - pad.b}"/></g>
    <line class="axis" x1="${pad.l}" x2="${W - pad.r}" y1="${mid}" y2="${mid}"/>
    <text x="${pad.l - 6}" y="${pad.t + 8}" text-anchor="end">100</text><text x="${pad.l - 6}" y="${mid + 3}" text-anchor="end">50</text>
    <text x="${pad.l - 6}" y="${height - pad.b}" text-anchor="end">100</text>
    ${bars}${labels}</svg>`;
}

// Momentum (-1..1) as a two-coloured area around zero.
export function momentumChart({ values, binS, colors, height = 150, markers = [] }) {
  const pts = values.map((v, i) => ({ x: (i + 0.5) * binS, y: Math.max(-1, Math.min(1, +v || 0)) }));
  const pos = pts.map((p) => ({ x: p.x, y: Math.max(0, p.y) }));
  const neg = pts.map((p) => ({ x: p.x, y: Math.min(0, p.y) }));
  return lineChart({
    series: [
      { values: pos, color: colors[0], fill: true, fillOpacity: 0.35, width: 1.5 },
      { values: neg, color: colors[1], fill: true, fillOpacity: 0.35, width: 1.5 },
    ],
    height, xMax: values.length * binS, yMin: -1, yMax: 1, zero: true, markers,
    xFmt: timeTick(values.length * binS), yFmt: (v) => (v === 0 ? '0' : ''), label: 'Momentum',
  });
}

export function sparkline(values, color, { width = 120, height = 28 } = {}) {
  if (!values.length) return '';
  const max = Math.max(...values.map(Math.abs), 1e-6);
  const pts = values.map((v, i) => `${((i / Math.max(1, values.length - 1)) * width).toFixed(1)},${(height / 2 - (v / max) * (height / 2 - 2)).toFixed(1)}`);
  return `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true">
    <line x1="0" x2="${width}" y1="${height / 2}" y2="${height / 2}" stroke="var(--line-strong)"/>
    <polyline points="${pts.join(' ')}" fill="none" stroke="${esc(color)}" stroke-width="1.5"/></svg>`;
}

// Horizontal stacked share bar (e.g. territory thirds).
export function stackBar(parts) {
  const total = parts.reduce((s, p) => s + (+p.value || 0), 0) || 1;
  return `<div class="thirds">${parts.map((p) => `<div style="width:${(100 * (+p.value || 0)) / total}%;background:${esc(p.color)}"
    title="${esc(p.label)} ${Math.round(+p.value || 0)}%">${(+p.value || 0) >= 12 ? `${Math.round(+p.value)}%` : ''}</div>`).join('')}</div>`;
}
