// Small DOM/format helpers shared by all views.

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export const esc = (value) =>
  String(value ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export const fmtSeconds = (t) => {
  t = Math.max(0, +t || 0);
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
};

export const fmtMs = (ms) => fmtSeconds((+ms || 0) / 1000);

export const fmtBytes = (bytes) => {
  bytes = +bytes || 0;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

export const fmtDate = (iso) => String(iso || '').slice(0, 16).replace('T', ' ');

export const fmtHours = (ms) => {
  const hours = ms / 3600000;
  if (hours < 1) return `${Math.round(hours * 60)} min`;
  return `${hours.toFixed(1)} h`;
};

export function toast(message, kind = 'ok') {
  const box = $('#toasts');
  const node = document.createElement('div');
  node.className = `toast ${kind === 'ok' ? '' : kind}`.trim();
  node.textContent = message;
  box.appendChild(node);
  setTimeout(() => node.remove(), 6000);
}

export function setMain(html) {
  $('#main').innerHTML = html;
  return $('#main');
}
