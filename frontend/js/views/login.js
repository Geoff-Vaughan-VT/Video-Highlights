// Sign-in view. Two paths:
//  - Access token: paste an env token or JWT issued by /v1/auth/token.
//  - Developer sign-in: identity headers, only valid when the server does
//    not require auth (VH_AUTH_REQUIRED=false).

import { whoami } from '../api.js';
import { setSession } from '../session.js';
import { $, esc, setMain } from '../ui.js';

export function renderLogin(onSignedIn) {
  setMain(`
    <div class="loginwrap">
      <h1>Sign in</h1>
      <div class="sub">Video Highlights Studio</div>
      <div class="logintabs">
        <button id="tab_token" class="active">Access token</button>
        <button id="tab_dev">Developer</button>
      </div>
      <div class="panel" id="pane_token">
        <label>API token or JWT</label>
        <input type="password" id="f_token" placeholder="Bearer token" autocomplete="off">
        <label>Tenant (optional — id or slug)</label>
        <input type="text" id="f_token_tenant" placeholder="leave blank to auto-resolve">
        <button class="btn" id="go_token">Sign in</button>
        <div class="errnote" id="err_token"></div>
      </div>
      <div class="panel" id="pane_dev" style="display:none">
        <label>User id</label>
        <input type="text" id="f_dev_user" value="dev_user">
        <label>Role</label>
        <select id="f_dev_role">
          <option value="admin" selected>admin</option>
          <option value="coach">coach</option>
          <option value="analyst">analyst</option>
          <option value="parent">parent</option>
        </select>
        <label>Tenant (optional — id or slug)</label>
        <input type="text" id="f_dev_tenant" placeholder="leave blank to auto-resolve">
        <button class="btn" id="go_dev">Sign in</button>
        <div class="note">Works only when the API runs without required auth (development mode).</div>
        <div class="errnote" id="err_dev"></div>
      </div>
    </div>`);

  const showPane = (token) => {
    $('#pane_token').style.display = token ? '' : 'none';
    $('#pane_dev').style.display = token ? 'none' : '';
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
  $('#go_dev').onclick = () => attempt(
    {
      devUserId: $('#f_dev_user').value.trim() || 'dev_user',
      devRole: $('#f_dev_role').value,
      tenantId: $('#f_dev_tenant').value.trim() || undefined,
    },
    $('#err_dev'),
  );
}
