// Thin API client. All requests carry the session's auth + tenant headers.

import { authHeaders, updateSession } from './session.js';

export const API = '/v1';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function parseError(response) {
  let message = `${response.status}`;
  try {
    const body = await response.json();
    message = body?.error?.message || JSON.stringify(body);
  } catch {
    try { message = await response.text(); } catch { /* keep status */ }
  }
  return new ApiError(response.status, message);
}

export async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  if (!response.ok) throw await parseError(response);
  const type = response.headers.get('content-type') || '';
  return type.includes('application/json') ? response.json() : response.text();
}

export const get = (path) => api(path);
export const post = (path, body) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
export const patch = (path, body) =>
  api(path, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
export const del = (path) => api(path, { method: 'DELETE' });

// Resolve the signed-in identity and remember the tenant the server picked,
// so later requests are explicit even when the user has several memberships.
export async function whoami() {
  const me = await get('/auth/me');
  if (me?.tenant_id) updateSession({ tenantId: me.tenant_id });
  return me;
}

// Multipart upload with progress callback (fetch has no upload progress).
export function uploadFile(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}${path}`);
    const headers = authHeaders();
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); } catch { resolve({}); }
      } else {
        let message = xhr.responseText;
        try { message = JSON.parse(xhr.responseText)?.error?.message || message; } catch { /* raw */ }
        reject(new ApiError(xhr.status, message));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, 'Upload failed: network error'));
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

// Authenticated file download (an <a href> would drop the auth headers).
export async function downloadFile(path, filename) {
  const response = await fetch(`${API}${path}`, { headers: authHeaders() });
  if (!response.ok) throw await parseError(response);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
