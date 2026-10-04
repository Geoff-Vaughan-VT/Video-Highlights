// Public share view. Rendered from a token in the URL hash with no
// authentication: this is what a parent, recruiter, or agent sees when a
// customer sends them a link.

import { API } from '../api.js';
import { icon } from '../icons.js';
import { emptyState, esc, fmtDay, fmtMs, setMain, skLines } from '../ui.js';

export async function renderShare(token) {
  setMain(`<div class="page" style="padding-top:24px"><div class="panel">${skLines(5)}</div></div>`);
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
    setMain(`<div class="page" style="padding-top:48px">${emptyState('link', 'Link unavailable', esc(error.message))}</div>`);
    return;
  }

  const match = payload.match || {};
  const header = `<div class="pagehead" style="margin-top:16px"><div><div class="eyebrow">Shared match</div>
      <h1>${esc(match.name || 'Shared match')}</h1>
      <div class="sub">${esc(match.home_team_name || 'Home')} vs ${esc(match.away_team_name || 'Away')}${match.match_date ? ` · ${esc(fmtDay(match.match_date))}` : ''}${payload.label ? ` · ${esc(payload.label)}` : ''}</div></div></div>`;
  const footer = '<div class="note" style="text-align:center;margin-top:32px">Shared from Video Highlights Studio · <a href="#matches">Sign in</a></div>';

  if (payload.scope === 'highlight') return setMain(`<div class="page">${header}${highlightPanel(payload.highlight)}${footer}</div>`);
  if (payload.scope === 'player_card') return setMain(`<div class="page">${header}${playerCardPanel(payload.player_card)}${footer}</div>`);

  setMain(`<div class="page">${header}
    <div class="panel"><div class="panel-title">Team stats</div>
      <div class="statgrid">${(payload.stats || []).map(shareStatTile).join('')}</div>
      <div class="note">Stats shown as “–” could not be measured from this footage${payload.analysis?.source_label ? ` (source: ${esc(payload.analysis.source_label)})` : ''}.</div></div>
    <div class="panel"><div class="panel-title">Highlights</div>
      ${(payload.highlights || []).length ? `<div class="tablewrap"><table>
        <thead><tr><th>Time</th><th>Type</th><th>Player</th></tr></thead>
        <tbody>${payload.highlights.map((item) => `<tr><td class="num">${fmtMs(item.occurred_at_ms)}</td><td>${esc(item.event_type.replace(/_/g, ' '))}</td>
          <td>${item.player_name ? `#${esc(item.jersey_number || '')} ${esc(item.player_name)}` : '—'}</td></tr>`).join('')}
        </tbody></table></div>` : '<div class="note">No highlights yet.</div>'}</div>${footer}</div>`);
}

function shareStatTile(stat) {
  const fmt = (value) => (value == null ? '–' : (stat.unit === 'percent' ? `${value}%` : `${value}`));
  if (!stat.available) {
    return `<div class="stat na"><div class="k">${esc(stat.label)}</div><div class="vals"><span>–</span><span class="mid">|</span><span>–</span></div></div>`;
  }
  return `<div class="stat"><div class="k">${esc(stat.label)}</div>
    <div class="vals"><span>${fmt(stat.home)}</span><span class="mid">home · away</span><span>${fmt(stat.away)}</span></div></div>`;
}

function highlightPanel(highlight) {
  if (!highlight) return emptyState('film', 'Highlight unavailable');
  return `<div class="panel"><div class="panel-title">${icon('film', 'sm')} Highlight</div>
    <div class="metrics">
      <div class="metric"><div class="v">${esc(highlight.event_type.replace(/_/g, ' '))}</div><div class="l">Type</div></div>
      <div class="metric"><div class="v">${fmtMs(highlight.occurred_at_ms)}</div><div class="l">Match time</div></div>
      <div class="metric"><div class="v">${Math.round((+highlight.confidence || 0) * 100)}%</div><div class="l">Confidence</div></div>
    </div>
    ${highlight.player_name ? `<div class="note">Attributed to #${esc(highlight.jersey_number || '')} ${esc(highlight.player_name)}</div>` : ''}</div>`;
}

function playerCardPanel(card) {
  if (!card) return emptyState('card', 'Player card unavailable');
  return `<div class="panel"><div class="pcard-head" style="margin-bottom:16px"><div class="jersey" style="--c:var(--accent);color:var(--accent-ink)">${esc(card.jersey_number)}</div>
      <div><h3>${esc(card.player_name)}</h3><div class="faint xs">${esc(card.position || '')}${card.team_name ? ` · ${esc(card.team_name)}` : ''}</div></div></div>
    <div class="metrics">
      <div class="metric"><div class="v">${card.highlight_count}</div><div class="l">Highlights</div></div>
      ${(card.stats || []).slice(0, 2).map((stat) => `<div class="metric"><div class="v">${stat.count}</div><div class="l">${esc(stat.label)}</div></div>`).join('')}
    </div>
    ${(card.highlights || []).length ? `<div class="tablewrap"><table>
      <thead><tr><th>Time</th><th>Type</th><th class="r">Confidence</th></tr></thead>
      <tbody>${card.highlights.map((item) => `<tr><td class="num">${fmtMs(item.occurred_at_ms)}</td><td>${esc(item.event_type.replace(/_/g, ' '))}</td>
        <td class="r">${Math.round((+item.confidence || 0) * 100)}%</td></tr>`).join('')}
      </tbody></table></div>` : '<div class="note">No highlights attributed yet.</div>'}</div>`;
}
