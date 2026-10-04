// DOM, formatting and component helpers shared by all views.

import { icon } from './icons.js';

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export const esc = (value) =>
  String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ---------------- formatting ---------------- */

export const fmtSeconds = (t) => {
  t = Math.max(0, +t || 0);
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
};

// Clock with hours when needed: 1:02:03 / 02:03.
export const fmtClock = (t) => {
  t = Math.max(0, +t || 0);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
};

export const fmtMs = (ms) => fmtSeconds((+ms || 0) / 1000);

export const fmtBytes = (bytes) => {
  bytes = +bytes || 0;
  if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(2)} TB`;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

export const fmtDate = (iso) => String(iso || '').slice(0, 16).replace('T', ' ');

export const fmtDay = (iso) => {
  if (!iso) return '';
  const date = new Date(iso.length <= 10 ? `${iso}T12:00:00` : iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
};

export const fmtHours = (ms) => {
  const hours = ms / 3600000;
  if (hours < 1) return `${Math.round(hours * 60)} min`;
  return `${hours.toFixed(1)} h`;
};

export const fmtDuration = (seconds) => {
  seconds = Math.max(0, Math.round(+seconds || 0));
  if (seconds < 60) return `${seconds}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h) return `${h}h ${m}m`;
  return s ? `${m}m ${s}s` : `${m}m`;
};

export const fmtRel = (iso) => {
  if (!iso) return '';
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(delta)) return '';
  if (delta < 45) return 'just now';
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  if (delta < 86400 * 14) return `${Math.round(delta / 86400)} d ago`;
  return fmtDay(iso);
};

// Number with fixed digits, or an en dash when unknown.
export const num = (value, digits = 0, suffix = '') => {
  if (value == null || value === '' || !Number.isFinite(+value)) return '–';
  return `${(+value).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}${suffix}`;
};

export const titleCase = (value) =>
  String(value || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

export const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

export function debounce(fn, wait = 200) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
}

/* ---------------- route lifecycle ---------------- */
// Views register teardown (timers, listeners, media) with onLeave(); the
// router calls leaveRoute() before rendering the next page. routeToken()
// lets async loaders detect that the user navigated away mid-request.

let leaveFns = [];
let token = 0;

export function onLeave(fn) { leaveFns.push(fn); }

export function leaveRoute() {
  const fns = leaveFns;
  leaveFns = [];
  token += 1;
  for (const fn of fns) {
    try { fn(); } catch (error) { console.warn('cleanup failed', error); }
  }
  $$('.drawer, .scrim').forEach((node) => node.remove());
}

export const routeToken = () => token;
export const isCurrent = (value) => value === token;

// Interval that is cleared automatically on navigation.
export function every(ms, fn) {
  const id = setInterval(fn, ms);
  onLeave(() => clearInterval(id));
  return () => clearInterval(id);
}

// Event listener removed automatically on navigation.
export function listen(target, type, handler, options) {
  target.addEventListener(type, handler, options);
  onLeave(() => target.removeEventListener(type, handler, options));
}

/* ---------------- components ---------------- */

export function toast(message, kind = 'ok') {
  const box = $('#toasts');
  if (!box) return;
  const node = document.createElement('div');
  node.className = `toast ${kind === 'ok' ? '' : kind}`.trim();
  const glyph = kind === 'err' ? 'alert' : kind === 'warn' ? 'info' : 'ok';
  node.innerHTML = `${icon(glyph)}<div>${esc(message)}</div>`;
  box.appendChild(node);
  while (box.children.length > 3) box.firstElementChild.remove();
  setTimeout(() => node.remove(), kind === 'err' ? 9000 : 4500);
}

export function setMain(html) {
  const main = $('#main');
  main.innerHTML = html;
  return main;
}

export function emptyState(glyph, title, text = '', action = '') {
  return `<div class="empty"><div class="ico">${icon(glyph, 'lg')}</div><h3>${esc(title)}</h3>
    ${text ? `<p>${text}</p>` : ''}${action}</div>`;
}

export const skLines = (count = 3) =>
  Array.from({ length: count }, (_, i) => `<div class="sk sk-line" style="width:${90 - (i % 3) * 18}%"></div>`).join('');

export const skCards = (count = 6) => Array.from({ length: count }, () => `
  <div class="mcard" aria-hidden="true"><div class="thumb sk" style="border-radius:0"></div>
    <div class="body">${skLines(3)}</div></div>`).join('');

export const skBlock = (height = 200) => `<div class="sk" style="height:${height}px"></div>`;

export function confirmDialog({ title, body = '', confirmLabel = 'Confirm', danger = false }) {
  return new Promise((resolve) => {
    const scrim = document.createElement('div');
    scrim.className = 'scrim';
    scrim.innerHTML = `<div class="modal" role="dialog" aria-modal="true" aria-labelledby="dlg_t">
      <h3 id="dlg_t">${esc(title)}</h3><div class="muted small">${body}</div>
      <div class="btnrow"><button class="btn2" data-a="no">Cancel</button>
      <button class="btn ${danger ? 'danger' : ''}" data-a="yes">${esc(confirmLabel)}</button></div></div>`;
    const done = (value) => { scrim.remove(); document.removeEventListener('keydown', onKey, true); resolve(value); };
    const onKey = (event) => { if (event.key === 'Escape') { event.stopPropagation(); done(false); } };
    scrim.addEventListener('click', (event) => {
      if (event.target === scrim) done(false);
      const action = event.target.closest('[data-a]')?.dataset.a;
      if (action) done(action === 'yes');
    });
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(scrim);
    $('[data-a=yes]', scrim).focus();
  });
}

export function promptDialog({ title, label, value = '', confirmLabel = 'Save' }) {
  return new Promise((resolve) => {
    const scrim = document.createElement('div');
    scrim.className = 'scrim';
    scrim.innerHTML = `<div class="modal" role="dialog" aria-modal="true">
      <h3>${esc(title)}</h3><label for="dlg_in">${esc(label)}</label><input type="text" id="dlg_in" value="${esc(value)}">
      <div class="btnrow"><button class="btn2" data-a="no">Cancel</button><button class="btn" data-a="yes">${esc(confirmLabel)}</button></div></div>`;
    const input = $('#dlg_in', scrim);
    const done = (ok) => { scrim.remove(); resolve(ok ? input.value.trim() : null); };
    scrim.addEventListener('click', (event) => {
      if (event.target === scrim) done(false);
      const action = event.target.closest('[data-a]')?.dataset.a;
      if (action) done(action === 'yes');
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') done(true);
      if (event.key === 'Escape') done(false);
      event.stopPropagation();
    });
    document.body.appendChild(scrim);
    input.focus();
  });
}

// Side drawer. Returns {el, body, close}. Only one drawer at a time.
export function openDrawer({ title, subtitle = '', html = '', footer = '', onClose }) {
  $$('.drawer').forEach((node) => node.remove());
  const el = document.createElement('aside');
  el.className = 'drawer';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', title.replace(/<[^>]+>/g, ''));
  el.innerHTML = `<header><div>${title}${subtitle ? `<div class="faint xs">${subtitle}</div>` : ''}</div>
    <button class="iconbtn" data-close aria-label="Close">${icon('x')}</button></header>
    <div class="body">${html}</div>${footer ? `<footer>${footer}</footer>` : ''}`;
  const onKey = (event) => { if (event.key === 'Escape' && !$('.scrim')) close(); };
  function close() {
    el.remove();
    document.removeEventListener('keydown', onKey);
    if (onClose) onClose();
  }
  $('[data-close]', el).onclick = close;
  document.addEventListener('keydown', onKey);
  document.body.appendChild(el);
  return { el, body: $('.body', el), close };
}

export async function copyText(text, label = 'Copied to clipboard') {
  try {
    await navigator.clipboard.writeText(text);
    toast(label);
  } catch {
    toast('Copy failed — select the text and copy it manually', 'warn');
  }
}

export function statusBadge(status) {
  const value = String(status || 'unknown');
  return `<span class="status ${esc(value)}">${esc(titleCase(value === 'cancel_requested' ? 'cancelling' : value))}</span>`;
}

export function toggleHtml(id, label, checked = false, extra = '') {
  return `<label class="toggle" for="${id}"><input type="checkbox" id="${id}" ${checked ? 'checked' : ''} ${extra}>
    <span class="track"></span><span>${label}</span></label>`;
}
