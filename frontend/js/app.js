// App shell: boot, auth gate, hash router, responsive nav.

import { whoami } from './api.js';
import { clearSession, getSession, setSession } from './session.js';
import { $, $$, esc } from './ui.js';
import { renderLogin } from './views/login.js';
import { renderMatches, renderMatchDetail } from './views/matches.js';
import { renderCreate } from './views/create.js';
import { renderJobs } from './views/jobs.js';
import { renderRuns, renderRunDetail } from './views/runs.js';
import { renderShare } from './views/share.js';

let currentUser = null;

export function go(page, arg) {
  location.hash = arg ? `${page}/${encodeURIComponent(arg)}` : page;
}

function markNav(page) {
  $$('#navlinks button[data-page]').forEach((button) =>
    button.classList.toggle('active', button.dataset.page === page));
}

function renderUserChip() {
  const box = $('#navuser');
  if (!currentUser) { box.innerHTML = ''; return; }
  const tenant = getSession().tenantId || '';
  box.innerHTML = `
    <div class="who">${esc(currentUser.user_id)}</div>
    <div>${esc(currentUser.role)}${currentUser.is_global_admin ? ' · global admin' : ''}</div>
    ${tenant ? `<div title="tenant">${esc(tenant)}</div>` : ''}
    <button id="signout">Sign out</button>`;
  // Explicit sign-out sticks even when the server allows anonymous access,
  // so the login screen is reachable in development mode too.
  $('#signout').onclick = () => { clearSession(); setSession({ signedOut: true }); currentUser = null; render(); };
}

async function render() {
  const [routePage, routeArg] = (location.hash.slice(1) || 'matches').split('/');
  // Share links are public: never gate them behind sign-in, and hide the
  // app navigation — a recruiter opening a link is not a Studio user.
  if (routePage === 'share' && routeArg) {
    $('#navlinks').classList.remove('open');
    document.body.classList.add('public');
    markNav('');
    return renderShare(decodeURIComponent(routeArg));
  }
  document.body.classList.remove('public');

  if (!currentUser) {
    const signIn = () => renderLogin(async () => { currentUser = await whoami(); renderUserChip(); render(); });
    if (getSession().signedOut) { renderUserChip(); signIn(); return; }
    try {
      currentUser = await whoami();
    } catch {
      renderUserChip();
      signIn();
      return;
    }
    renderUserChip();
  }
  const [page, rawArg] = (location.hash.slice(1) || 'matches').split('/');
  const arg = rawArg ? decodeURIComponent(rawArg) : undefined;
  markNav(page);
  $('#navlinks').classList.remove('open');
  if (page === 'matches' && arg) return renderMatchDetail(arg);
  if (page === 'matches') return renderMatches();
  if (page === 'create') return renderCreate();
  if (page === 'jobs') return renderJobs(arg);
  if (page === 'runs' && arg) return renderRunDetail(arg);
  if (page === 'runs') return renderRuns();
  return renderMatches();
}

function bindNav() {
  $$('#navlinks button[data-page]').forEach((button) => {
    button.onclick = () => {
      const page = button.dataset.page;
      if (page === '_docs') { window.open('/docs', '_blank'); return; }
      go(page);
    };
  });
  $('#navtoggle').onclick = () => $('#navlinks').classList.toggle('open');
}

window.addEventListener('hashchange', render);
bindNav();
render();
