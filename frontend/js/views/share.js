// Public share view. Rendered from a token in the URL hash with no
// authentication: this is what a parent, recruiter, or agent sees when a
// customer sends them a link.

import { API } from '../api.js';
import { esc, fmtMs, setMain } from '../ui.js';

export async function renderShare(token) {
  setMain('<div class="empty">Loading shared match…</div>');
  let payload;
  try {
    // Deliberately a bare fetch: no auth or tenant headers on a public link.
    const response = await fetch(`${API}/public/shares/${encodeURIComponent(token)}`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body?.error?.message || 'This share link is invalid, expired, or revoked.');
    }
    payload = await response.json();
  } catch (error) {
    setMain(`<div class="empty">${esc(error.message)}</div>`);
    return;
  }

  const match = payload.match || {};
  const header = `
    <h1>${esc(match.name || 'Shared match')}</h1>
    <div class="sub">${esc(match.home_team_name || 'Home')} vs ${esc(match.away_team_name || 'Away')}
      ${match.match_date ? ' · ' + esc(match.match_date) : ''}
      ${payload.label ? ' · ' + esc(payload.label) : ''}</div>`;

  if (payload.scope === 'highlight') return setMain(header + highlightPanel(payload.highlight));
  if (payload.scope === 'player_card') return setMain(header + playerCardPanel(payload.player_card));

  setMain(`${header}
    <div class="panel"><h3>Team stats</h3>
      <div class="statgrid">${(payload.stats || []).map(shareStatTile).join('')}</div>
      <div class="note">Stats shown as “–” could not be measured from this footage${
        payload.analysis?.source_label ? ` (source: ${esc(payload.analysis.source_label)})` : ''}.</div>
    </div>
    <div class="panel"><h3>Highlights</h3>
      ${(payload.highlights || []).length ? `<div class="tablewrap"><table>
        <thead><tr><th>at</th><th>type</th><th>player</th></tr></thead>
        <tbody>${payload.highlights.map((item) => `
          <tr><td>${fmtMs(item.occurred_at_ms)}</td><td>${esc(item.event_type)}</td>
          <td>${item.player_name ? `#${esc(item.jersey_number || '')} ${esc(item.player_name)}` : '—'}</td></tr>`).join('')}
        </tbody></table></div>` : '<div class="note">No highlights yet.</div>'}
    </div>
    <div class="note">Shared from Video Highlights Studio · <a href="#matches">Sign in</a></div>`);
}

function shareStatTile(stat) {
  const fmt = (value) => value == null ? '–' : (stat.unit === 'percent' ? `${value}%` : `${value}`);
  if (!stat.available) {
    return `<div class="stat na"><div class="k">${esc(stat.label)}</div>
      <div class="vals"><span>–</span><span class="mid">|</span><span>–</span></div></div>`;
  }
  return `<div class="stat"><div class="k">${esc(stat.label)}</div>
    <div class="vals"><span>${fmt(stat.home)}</span><span class="mid">home · away</span><span>${fmt(stat.away)}</span></div></div>`;
}

function highlightPanel(highlight) {
  if (!highlight) return '<div class="empty">Highlight unavailable.</div>';
  return `<div class="panel"><h3>Highlight</h3>
    <div class="metrics">
      <div class="metric"><div class="v">${esc(highlight.event_type)}</div><div class="l">Type</div></div>
      <div class="metric"><div class="v">${fmtMs(highlight.occurred_at_ms)}</div><div class="l">Match time</div></div>
      <div class="metric"><div class="v">${(+highlight.confidence || 0).toFixed(2)}</div><div class="l">Confidence</div></div>
    </div>
    ${highlight.player_name ? `<div class="note">Attributed to #${esc(highlight.jersey_number || '')} ${esc(highlight.player_name)}</div>` : ''}
  </div>`;
}

function playerCardPanel(card) {
  if (!card) return '<div class="empty">Player card unavailable.</div>';
  return `<div class="panel"><h3>Player card</h3>
    <div style="font-size:18px;font-weight:700;margin-bottom:4px">#${esc(card.jersey_number)} ${esc(card.player_name)}</div>
    <div class="note" style="margin:0 0 12px">${esc(card.position || '')}${card.team_name ? ' · ' + esc(card.team_name) : ''}</div>
    <div class="metrics">
      <div class="metric"><div class="v">${card.highlight_count}</div><div class="l">Highlights</div></div>
      ${(card.stats || []).slice(0, 2).map((stat) =>
        `<div class="metric"><div class="v">${stat.count}</div><div class="l">${esc(stat.label)}</div></div>`).join('')}
    </div>
    ${(card.highlights || []).length ? `<div class="tablewrap"><table>
      <thead><tr><th>at</th><th>type</th><th>conf</th></tr></thead>
      <tbody>${card.highlights.map((item) => `
        <tr><td>${fmtMs(item.occurred_at_ms)}</td><td>${esc(item.event_type)}</td>
        <td>${(+item.confidence || 0).toFixed(2)}</td></tr>`).join('')}
      </tbody></table></div>` : '<div class="note">No highlights attributed yet.</div>'}
  </div>`;
}
