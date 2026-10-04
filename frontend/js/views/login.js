// Sign-in view. Two paths:
//  - Access token: paste an env token or JWT issued by /v1/auth/token.
//  - Developer sign-in: identity headers, only valid when the server does
//    not require auth (VH_AUTH_REQUIRED=false).

import { whoami } from '../api.js';
import { icon } from '../icons.js';
import { setSession } from '../session.js';
import { $, esc, setMain } from '../ui.js';

export function renderLogin(onSignedIn) {
  setMain(`
    <div class="loginwrap">
      <div class="panel" style="padding:32px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
          <span class="mark" style="width:36px;height:36px;border-radius:10px;background:var(--accent);display:grid;place-items:center;color:var(--accent-ink)">${icon('logo')}</span>
          <div><h1 style="font-size:20px">Sign in</h1><div class="faint xs">Video Highlights Studio</div></div></div>
        <div class="logintabs" role="tablist">
          <button id="tab_token" class="active" role="tab">Access token</button>
          <button id="tab_dev" role="tab">Developer</button>
        </div>
        <div id="pane_token">
          <label for="f_token">API token or JWT</label>
          <input type="password" id="f_token" placeholder="Bearer token" autocomplete="off">
          <label for="f_token_tenant">Tenant (optional — id or slug)</label>
          <input type="text" id="f_token_tenant" placeholder="leave blank to auto-resolve">
          <button class="btn" id="go_token">Sign in</button>
          <div class="errnote" id="err_token" role="alert"></div>
        </div>
        <div id="pane_dev" hidden>
          <label for="f_dev_user">User id</label>
          <input type="text" id="f_dev_user" value="dev_user">
          <label for="f_dev_role">Role</label>
          <select id="f_dev_role">
            <option value="admin" selected>admin</option><option value="coach">coach</option>
            <option value="analyst">analyst</option><option value="parent">parent</option>
          </select>
          <label for="f_dev_tenant">Tenant (optional — id or slug)</label>
          <input type="text" id="f_dev_tenant" placeholder="leave blank to auto-resolve">
          <button class="btn" id="go_dev">Sign in</button>
          <div class="note">Works only when the API runs without required auth (development mode).</div>
          <div class="errnote" id="err_dev" role="alert"></div>
        </div>
      </div>
    </div>`);

  const showPane = (token) => {
    $('#pane_token').hidden = !token;
    $('#pane_dev').hidden = token;
    $('#tab_token').classList.toggle('active', token);
    $('#tab_dev').classList.toggle('active', !token);
  };
  $('#tab_token').onclick = () => showPane(true);
  $('#tab_dev').onclick = () => showPane(false);

  async function attempt(session, errorBox) {
    setSession(session);
    try {
      await whoami();
      onSignedIn();
    } catch (error) {
      errorBox.textContent = error.status === 401
        ? 'Sign-in failed: the server rejected these credentials.'
        : `Sign-in failed: ${esc(error.message)}`;
    }
  }

  $('#go_token').onclick = () => attempt(
    { token: $('#f_token').value.trim(), tenantId: $('#f_token_tenant').value.trim() || undefined },
    $('#err_token'),
  );
  $('#f_token').onkeydown = (event) => { if (event.key === 'Enter') $('#go_token').click(); };
  $('#go_dev').onclick = () => attempt(
    {
      devUserId: $('#f_dev_user').value.trim() || 'dev_user',
      devRole: $('#f_dev_role').value,
      tenantId: $('#f_dev_tenant').value.trim() || undefined,
    },
    $('#err_dev'),
  );
}
