// Runs: film-review workspace over completed processing runs — the ported
// Studio review UI (video views, bookmarks, team panel, AI match report).

import { API, get } from '../api.js';
import { $, $$, esc, fmtSeconds, setMain } from '../ui.js';

let currentRun = null;

export async function renderRuns() {
  const main = setMain(`
    <h1>Runs</h1>
    <div class="sub">Every processing run — click to review the film</div>
    <div class="grid" id="grid"><div class="empty">Loading…</div></div>`);
  try {
    const { runs } = await get('/studio/runs');
    const grid = $('#grid', main);
    grid.innerHTML = runs.length ? runs.map((run) => {
      const stats = run.stats || {};
      return `<div class="card" data-id="${esc(run.run_id)}">
        <h3>${esc(run.run_id)}</h3>
        <div class="meta">${esc((run.generated_at || '').slice(0, 16).replace('T', ' '))}</div>
        <div class="chips">
          <span class="chip goal">⚽ ${(run.goal_events || []).length} goals</span>
          <span class="chip">${stats.bookmark_count ?? 0} bookmarks</span>
          <span class="chip">${(run.videos?.clips || []).length} clips</span>
          ${(run.card_events || []).map((card) =>
            `<span class="chip ${card.kind === 'red_card' ? 'card-r' : 'card-y'}">${card.kind === 'red_card' ? '🟥' : '🟨'} ${fmtSeconds(card.t)}</span>`).join('')}
        </div></div>`;
    }).join('') : '<div class="empty">No runs yet — process a match from <b>Create</b>.</div>';
    $$('.card', grid).forEach((card) => {
      card.onclick = () => { location.hash = `runs/${encodeURIComponent(card.dataset.id)}`; };
    });
  } catch (error) {
    $('#grid', main).innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}

export async function renderRunDetail(runId) {
  const main = setMain('<div class="empty">Loading run…</div>');
  let run;
  try {
    run = currentRun = await get(`/studio/runs/${runId}`);
  } catch (error) {
    main.innerHTML = `<div class="empty">${esc(error.message)}</div>`;
    return;
  }
  const videos = run.videos || {};
  const views = [];
  if (videos.debug) views.push(['Debug', 'debug', videos.debug]);
  if (videos.zoom) views.push(['Zoom', 'zoom', videos.zoom]);
  if (videos.reel) views.push(['Broadcast reel', 'reel', videos.reel]);
  if (videos.montage) views.push(['Montage', 'montage', videos.montage]);
  if (videos.original) views.push(['Original (window)', 'original', videos.original]);
  (videos.clips || []).forEach((clip, index) => views.push([`Clip ${index + 1}`, `clip${index}`, clip]));
  const first = videos.reel ? 'reel' : (views[0]?.[1] || '');

  main.innerHTML = `
    <h1>${esc(runId)}</h1>
    <div class="sub"><a href="#runs">← Runs</a> &nbsp; ${esc(run.video_path || '')}</div>
    <div class="viewbar" id="vb">${views.map(([label, key]) =>
      `<button data-v="${key}">${esc(label)}</button>`).join('')}</div>
    <div class="player">
      <div>
        <video id="vid" controls preload="metadata"></video>
        <div class="panel" style="margin-top:14px"><h3>Bookmarks</h3>
          <div class="tablewrap"><table><thead><tr><th>at</th><th>type</th><th>conf</th><th>state</th><th>sources</th></tr></thead>
          <tbody>${(run.bookmarks || []).map((bookmark) =>
            `<tr class="clickable" data-seek="${+bookmark.occurred_at_s || 0}">
              <td>${fmtSeconds(bookmark.occurred_at_s)}</td>
              <td>${esc(bookmark.event_type)}${String(bookmark.label || '').endsWith('_detected') ? ' ✓' : ''}</td>
              <td>${(+bookmark.confidence || 0).toFixed(2)}</td>
              <td>${esc(bookmark.game_state || '')}</td>
              <td>${esc((bookmark.sources || []).join(','))}</td></tr>`).join('') || '<tr><td colspan=5>none</td></tr>'}
          </tbody></table></div>
          <div class="note">Clicking a bookmark seeks the Original/Zoom/Debug views (clip/reel timebases differ).</div>
        </div>
      </div>
      <div>
        <div class="metrics">
          <div class="metric"><div class="v">${(run.goal_events || []).length}</div><div class="l">Goals</div></div>
          <div class="metric"><div class="v">${(run.card_events || []).length}</div><div class="l">Cards</div></div>
          <div class="metric"><div class="v">${run.stats?.bookmark_count ?? 0}</div><div class="l">Bookmarks</div></div>
        </div>
        ${teamPanel(run.team_stats)}
        ${reportPanel(run.match_report)}
        ${panelList('Goals', (run.goal_events || []).map((goal) =>
          `${fmtSeconds(goal.t)}${goal.team ? ' - <b>' + esc(goal.team) + '</b>' : ''} - ${esc(goal.side)} goal (${(+goal.confidence).toFixed(2)})<br><small style="color:var(--dim)">${esc(goal.reason)}</small>`))}
        ${panelList('Set pieces', (run.set_piece_events || []).map((piece) =>
          `${fmtSeconds(piece.t_kick)} - ${esc(piece.kind)}${piece.side ? ' (' + esc(piece.side) + ')' : ''}`))}
        ${panelList('Cards', (run.card_events || []).map((card) =>
          `${fmtSeconds(card.t)} - ${esc(card.kind)} (${(+card.confidence).toFixed(2)})`))}
        <div class="panel"><h3>Game states (s)</h3><table><tbody>${Object.entries(run.state_summary_s || {}).map(([key, value]) =>
          `<tr><td>${esc(key)}</td><td>${value}</td></tr>`).join('')}</tbody></table></div>
        ${(run.card_crops || []).length ? `<div class="panel"><h3>Card review crops</h3><div class="crops">${run.card_crops.map((crop) =>
          `<img src="${API}/studio/runs/${esc(runId)}/file/${esc(crop)}" title="${esc(crop)}">`).join('')}</div></div>` : ''}
      </div>
    </div>`;

  const fileByView = Object.fromEntries(views.map(([, key, file]) => [key, file]));
  const setView = (key) => {
    $$('#vb button').forEach((button) => button.classList.toggle('active', button.dataset.v === key));
    $('#vid').src = `${API}/studio/runs/${runId}/file/${fileByView[key]}`;
  };
  $$('#vb button').forEach((button) => { button.onclick = () => setView(button.dataset.v); });
  $$('tr[data-seek]').forEach((row) => {
    row.onclick = () => {
      const video = $('#vid');
      const offset = +(currentRun?.trim_offset_seconds || 0);
      video.currentTime = Math.max(0, +row.dataset.seek - offset - 4); // 4s pre-roll
      video.play();
    };
  });
  if (first) setView(first);
}

function teamPanel(teamStats) {
  if (!teamStats || !teamStats.teams || !teamStats.teams.length) return '';
  const [teamA, teamB] = teamStats.teams;
  const possessionA = teamStats.possession_pct?.[teamA.team] ?? 0;
  const possessionB = teamStats.possession_pct?.[teamB.team] ?? 0;
  const goalsA = teamStats.goals?.[teamA.team] ?? 0;
  const goalsB = teamStats.goals?.[teamB.team] ?? 0;
  return `<div class="panel"><h3>Teams</h3>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <span><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${esc(teamA.color)}"></i> <b>${esc(teamA.team)}</b></span>
      <span style="font-size:20px;font-weight:700">${goalsA} - ${goalsB}</span>
      <span><b>${esc(teamB.team)}</b> <i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${esc(teamB.color)}"></i></span>
    </div>
    <div style="color:var(--dim);font-size:12px;margin-bottom:4px">Possession ${possessionA}% / ${possessionB}%</div>
    <div class="bar" style="height:10px"><i style="width:${possessionA}%;background:${esc(teamA.color)}"></i></div>
    ${teamStats.defending_side && Object.keys(teamStats.defending_side).length
      ? `<div class="note">1st half: ${Object.entries(teamStats.defending_side).map(([team, side]) => `${esc(team)} defends ${side}`).join(', ')}</div>` : ''}
  </div>`;
}

function panelList(title, items) {
  return items.length
    ? `<div class="panel"><h3>${title}</h3>${items.map((item) => `<div style="margin-bottom:8px">${item}</div>`).join('')}</div>`
    : '';
}

function reportPanel(markdown) {
  if (!markdown) return '';
  const html = esc(markdown).split(/\r?\n/).map((line) => {
    if (/^###\s/.test(line)) return `<div style="font-weight:700;margin:10px 0 4px">${line.slice(4)}</div>`;
    if (/^##\s/.test(line)) return `<div style="font-weight:700;font-size:14px;margin:12px 0 4px">${line.slice(3)}</div>`;
    if (/^#\s/.test(line)) return `<div style="font-weight:700;font-size:15px;margin:12px 0 4px">${line.slice(2)}</div>`;
    if (/^[-*]\s/.test(line)) return `<div style="padding-left:14px">• ${line.slice(2)}</div>`;
    if (!line.trim()) return '<div style="height:6px"></div>';
    return `<div>${line}</div>`;
  }).join('').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  return `<div class="panel"><h3>AI match report</h3><div style="font-size:12.5px;max-height:420px;overflow:auto">${html}</div></div>`;
}
