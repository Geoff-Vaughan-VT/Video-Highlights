// App shell: boot, auth gate, hash router, responsive nav, theme.
//
// Routes (hash):
//   #matches                      library (default; #library alias)
//   #matches/<match_id>?run=&tab= match review
//   #runs                         library, unlinked-runs section
//   #runs/<run_id>                run review (redirects to its match when linked)
//   #create                       new-run wizard (#new alias)
//   #jobs[/<job_id>]              jobs + live progress
//   #settings                     hardware & settings
//   #share/<token>                public share page (no auth)

import { whoami } from './api.js';
import { go, parseRoute } from './router.js';
import { icon } from './icons.js';
import { clearSession, getSession, setSession } from './session.js';
import { $, $$, esc, leaveRoute } from './ui.js';
import { renderLogin } from './views/login.js';
import { renderMatches, renderMatchDetail } from './views/matches.js';
import { renderCreate } from './views/create.js';
import { renderJobs } from './views/jobs.js';
import { renderRuns, renderRunDetail } from './views/runs.js';
import { renderSettings } from './views/settings.js';
import { renderShare } from './views/share.js';

let currentUser = null;

const NAV_ALIASES = { library: 'matches', runs: 'matches', new: 'create' };

function markNav(page) {
  const active = NAV_ALIASES[page] || page;
  $$('#navlinks button[data-page]').forEach((button) =>
    button.classList.toggle('active', button.dataset.page === active));
}

/* ---------------- theme ---------------- */

function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function toggleTheme() {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('vh_theme', next); } catch { /* private mode */ }
  renderUserChip();
  window.dispatchEvent(new CustomEvent('vh-theme'));
}

function renderUserChip() {
  const box = $('#navuser');
  const themeBtn = `<button class="iconbtn" id="themebtn" aria-label="Switch to ${currentTheme() === 'dark' ? 'light' : 'dark'} theme"
    title="Theme">${icon(currentTheme() === 'dark' ? 'sun' : 'moon')}</button>`;
  if (!currentUser) {
    box.innerHTML = `<div class="footbtns">${themeBtn}</div>`;
  } else {
    const tenant = getSession().tenantId || currentUser.tenant_id || '';
    box.innerHTML = `
      <div class="who">${esc(currentUser.user_id)}</div>
      <div>${esc(currentUser.role)}${currentUser.is_global_admin ? ' · global admin' : ''}</div>
      ${tenant ? `<div title="tenant">${esc(tenant)}</div>` : ''}
      <div class="footbtns">${themeBtn}
        <button class="btn-ghost sm" id="signout">${icon('logout', 'sm')} Sign out</button></div>`;
    // Explicit sign-out sticks even when the server allows anonymous access,
    // so the login screen is reachable in development mode too.
    $('#signout').onclick = () => { clearSession(); setSession({ signedOut: true }); currentUser = null; render(); };
  }
  $('#themebtn').onclick = toggleTheme;
}

/* ---------------- router ---------------- */

async function render() {
  leaveRoute();
  const { page, arg, query } = parseRoute();
  $('#navlinks').classList.remove('open');
  // Share links are public: never gate them behind sign-in, and hide the
  // app navigation — a recruiter opening a link is not a Studio user.
  if (page === 'share' && arg) {
    document.body.classList.add('public');
    markNav('');
    return renderShare(arg);
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
  markNav(page);
  window.scrollTo(0, 0);
  if ((page === 'matches' || page === 'library') && arg) return renderMatchDetail(arg, query);
  if (page === 'matches' || page === 'library') return renderMatches(query);
  if (page === 'create' || page === 'new') return renderCreate();
  if (page === 'jobs') return renderJobs(arg);
  if (page === 'runs' && arg) return renderRunDetail(arg, query);
  if (page === 'runs') return renderRuns();
  if (page === 'settings') return renderSettings();
  return renderMatches(query);
}

function bindNav() {
  $('#logomark').innerHTML = icon('logo');
  $('#navtoggle').innerHTML = icon('menu');
  $$('#navlinks button[data-page]').forEach((button) => {
    button.innerHTML = `${icon(button.dataset.icon)}<span>${button.innerHTML}</span>`;
    button.onclick = () => {
      const page = button.dataset.page;
      if (page === '_docs') { window.open('/docs', '_blank', 'noopener'); return; }
      if (location.hash === `#${page}`) render(); else go(page);
    };
  });
  $('#navtoggle').onclick = () => {
    const open = $('#navlinks').classList.toggle('open');
    $('#navtoggle').setAttribute('aria-expanded', String(open));
  };
}

window.addEventListener('hashchange', render);
bindNav();
renderUserChip();
render();
