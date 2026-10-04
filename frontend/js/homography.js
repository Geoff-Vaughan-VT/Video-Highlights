// Small projective-geometry kit for the pitch calibration UI.
//
// solveHomography() is the classic 4-point DLT: fixing h33 = 1 leaves 8
// unknowns, two linear equations per correspondence, solved as an 8x8 system
// by Gaussian elimination with partial pivoting. Pitch metres here use the
// pitch.js convention (origin at the top-left corner flag, x along the
// 105 m touchline, y down the 68 m goal line); the pipeline's
// PitchCalibration is centred instead (x in ±52.5, y in ±34).

import { PITCH } from './pitch.js';

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-12) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      if (!f) continue;
      for (let c = col; c <= n; c += 1) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// 3x3 H with H * [src, 1] ~ [dst, 1] for 4 point pairs, or null if degenerate.
export function solveHomography(src, dst) {
  if (src?.length !== 4 || dst?.length !== 4) return null;
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const h = solveLinear(A, b);
  if (!h || h.some((v) => !Number.isFinite(v))) return null;
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

// Apply H to a point; null on/behind the vanishing line (w <= 0).
export function project(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (!(w > 1e-12)) return null;
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

export function invert3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-15) return null;
  const inv = [
    [A, -(b * i - c * h), b * f - c * e],
    [B, a * i - c * g, -(a * f - c * d)],
    [C, -(a * h - b * g), a * e - b * d],
  ];
  return inv.map((row) => row.map((v) => v / det));
}

// Pitch-metre corners matching the TL, TR, BR, BL click order.
export const PITCH_CORNERS_M = [[0, 0], [PITCH.L, 0], [PITCH.L, PITCH.W], [0, PITCH.W]];

export function isConvexQuad(points) {
  if (points?.length !== 4) return false;
  let sign = 0;
  for (let i = 0; i < 4; i += 1) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % 4];
    const [cx, cy] = points[(i + 2) % 4];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (Math.abs(cross) < 1e-9) return false;
    const s = Math.sign(cross);
    if (sign && s !== sign) return false;
    sign = s;
  }
  return true;
}

// Pitch markings as polylines in pitch metres (pitch.js convention).
export function pitchPolylines() {
  const { L, W } = PITCH;
  const rect = (x, y, w, h) => [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]];
  const arc = (cx, cy, r, a0, a1, n = 48) => Array.from({ length: n + 1 }, (_, k) => {
    const a = a0 + ((a1 - a0) * k) / n;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  });
  // Penalty arcs: the part of the 9.15 m circle around the spot outside the box.
  const da = Math.acos((16.5 - 11) / 9.15);
  return {
    outline: [rect(0, 0, L, W)],
    lines: [
      [[L / 2, 0], [L / 2, W]],
      rect(0, (W - 40.32) / 2, 16.5, 40.32), rect(L - 16.5, (W - 40.32) / 2, 16.5, 40.32),
      rect(0, (W - 18.32) / 2, 5.5, 18.32), rect(L - 5.5, (W - 18.32) / 2, 5.5, 18.32),
      arc(L / 2, W / 2, 9.15, 0, Math.PI * 2, 72),
      arc(11, W / 2, 9.15, -da, da, 24), arc(L - 11, W / 2, 9.15, Math.PI - da, Math.PI + da, 24),
    ],
    spots: [[L / 2, W / 2], [11, W / 2], [L - 11, W / 2]],
  };
}

// Goal mouths the pipeline derives from a corner calibration
// (game_tracking.estimate_field_geometry._calibrated_goal): project the posts
// at y = -3.66 - 2.5 m and +3.66 m (centred metres) on each goal line, take
// the mean x as the goal line and pad outward by max(24 px, 5 % of the field
// width). Inputs/outputs in source pixels.
export function deriveGoalBoxes(cornersPx, frameW) {
  const H = solveHomography(PITCH_CORNERS_M, cornersPx);
  if (!H) return { left: null, right: null };
  const xs = cornersPx.map((p) => p[0]);
  const depth = Math.max(24, (Math.max(...xs) - Math.min(...xs)) * 0.05);
  const yTop = PITCH.W / 2 - 3.66 - 2.5;
  const yBottom = PITCH.W / 2 + 3.66;
  const box = (side) => {
    const x = side === 'left' ? 0 : PITCH.L;
    const a = project(H, x, yTop);
    const b = project(H, x, yBottom);
    if (!a || !b) return null;
    const gx = (a[0] + b[0]) / 2;
    const y1 = Math.min(a[1], b[1]);
    const y2 = Math.max(a[1], b[1]);
    if (y2 - y1 < 4) return null;
    return side === 'left'
      ? { x1: Math.max(0, gx - depth), y1, x2: gx, y2 }
      : { x1: gx, y1, x2: Math.min(frameW, gx + depth), y2 };
  };
  return { left: box('left'), right: box('right') };
}

// Image pixels -> pitch metres (pitch.js convention) for event maps. Prefers
// the run's saved manual calibration (calibration.json, normalized corners),
// then the analysis calibration (analysis_player_stats.pitch_calibration,
// whose homography is centred: TL corner -> (-52.5, -34)).
export function calibrationMapper(model) {
  const manual = model.manualCalibration;
  const fw = +manual?.frame_width || model.frame?.w;
  const fh = +manual?.frame_height || model.frame?.h;
  if (manual?.pitch_corners?.length === 4 && fw && fh) {
    const H = solveHomography(manual.pitch_corners.map(([x, y]) => [x * fw, y * fh]), PITCH_CORNERS_M);
    if (H) return { mode: 'manual', toPitch: (x, y) => project(H, x, y) };
  }
  const cal = model.calibration;
  if (Array.isArray(cal?.homography) && cal.homography.length === 3) {
    const H = cal.homography;
    let off = [PITCH.L / 2, PITCH.W / 2];
    const tl = cal.image_corners_px?.[0];
    if (tl) {
      const p = project(H, +tl[0], +tl[1]);
      if (p && Math.hypot(p[0], p[1]) < 5) off = [0, 0]; // already top-left based
    }
    return {
      mode: 'calibrated',
      toPitch: (x, y) => { const p = project(H, x, y); return p && [p[0] + off[0], p[1] + off[1]]; },
    };
  }
  return null;
}
