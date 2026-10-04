// Hash-route helpers shared by the shell and views.

export function go(page, arg, query) {
  const qs = query ? `?${new URLSearchParams(query)}` : '';
  location.hash = arg ? `${page}/${encodeURIComponent(arg)}${qs}` : `${page}${qs}`;
}

export function parseRoute() {
  const raw = location.hash.slice(1) || 'matches';
  const [path, qs = ''] = raw.split('?');
  const [page, rawArg] = path.split('/');
  return { page: page || 'matches', arg: rawArg ? decodeURIComponent(rawArg) : undefined, query: new URLSearchParams(qs) };
}

// Update the query string without re-rendering the page.
export function setQuery(patch) {
  const { page, arg, query } = parseRoute();
  for (const [key, value] of Object.entries(patch)) {
    if (value == null || value === '') query.delete(key); else query.set(key, value);
  }
  const qs = query.toString();
  const next = `#${page}${arg ? `/${encodeURIComponent(arg)}` : ''}${qs ? `?${qs}` : ''}`;
  history.replaceState(null, '', next);
}
