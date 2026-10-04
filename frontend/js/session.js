// Session store: auth token or dev identity, plus the resolved tenant.
// Persisted in localStorage so refreshes keep the signed-in state.

const KEY = 'vh_session_v1';

let session = load();

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {};
  } catch {
    return {};
  }
}

function persist() {
  localStorage.setItem(KEY, JSON.stringify(session));
}

export function getSession() {
  return { ...session };
}

export function setSession(next) {
  session = { ...next };
  persist();
}

export function updateSession(patch) {
  session = { ...session, ...patch };
  persist();
}

export function clearSession() {
  session = {};
  localStorage.removeItem(KEY);
}

// Headers for every API call. Three shapes:
//  - token auth:  Authorization: Bearer <token>
//  - dev auth:    x-user-id / x-user-role (server must allow anonymous access)
//  - either mode: X-Tenant-Id when a tenant has been chosen/resolved
export function authHeaders() {
  const headers = {};
  if (session.token) headers['Authorization'] = `Bearer ${session.token}`;
  if (session.devUserId) headers['x-user-id'] = session.devUserId;
  if (session.devRole) headers['x-user-role'] = session.devRole;
  if (session.tenantId) headers['X-Tenant-Id'] = session.tenantId;
  return headers;
}
