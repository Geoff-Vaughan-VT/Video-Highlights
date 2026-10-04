// Matches: the customer-facing library. The detail view is the match
// dashboard: baseline stat catalog with evidence drill-down, roster
// management with CSV import, and highlight (event) assignment.

import { del, downloadFile, get, post } from '../api.js';
import { $, $$, esc, fmtDate, fmtMs, setMain, toast } from '../ui.js';

export async function renderMatches() {
  const main = setMain(`
    <h1>Matches</h1>
    <div class="sub">Your uploaded games — open one for stats, roster, and highlights</div>
    <div class="grid" id="grid"><div class="empty">Loading…</div></div>`);
  try {
    const { items } = await get('/matches?limit=200');
    const grid = $('#grid', main);
    if (!items.length) {
      grid.innerHTML = '<div class="empty">No matches yet — upload one from <b>Create</b>.</div>';
      return;
    }
    grid.innerHTML = items.map((match) => `
      <div class="card" data-id="${esc(match.match_id)}">
        <h3>${esc(match.name || match.match_id)}</h3>
        <div class="meta">${esc(match.home_team_name || 'Home')} vs ${esc(match.away_team_name || 'Away')}
          ${match.match_date ? ' · ' + esc(match.match_date) : ''}</div>
        <div class="chips"><span class="chip">${esc(fmtDate(match.created_at))}</span></div>
      </div>`).join('');
    $$('.card', grid).forEach((card) => {
      card.onclick = () => { location.hash = `matches/${encodeURIComponent(card.dataset.id)}`; };
    });
  } catch (error) {
    $('#grid', main).innerHTML = `<div class="empty">${esc(error.message)}</div>`;
  }
}

export async function renderMatchDetail(matchId) {
  const main = setMain('<div class="empty">Loading match…</div>');
  let match;
  try {
    match = await get(`/matches/${matchId}`);
  } catch (error) {
    main.innerHTML = `<div class="empty">${esc(error.message)}</div>`;
    return;
  }
  main.innerHTML = `
    <h1>${esc(match.name || matchId)}</h1>
    <div class="sub"><a href="#matches">← Matches</a> &nbsp;
      ${esc(match.home_team_name || 'Home')} vs ${esc(match.away_team_name || 'Away')}
      ${match.match_date ? ' · ' + esc(match.match_date) : ''}</div>
    <div class="viewbar" style="margin:0 0 14px">
      <button class="btn2" id="share_match">⬆️ Share match</button>
      <button class="btn2" id="share_list">Share links</button>
    </div>
    <div id="sharebox"></div>
    <div id="statsbox" class="panel"><h3>Team stats</h3><div class="note">Loading…</div></div>
    <div class="player">
      <div id="eventsbox" class="panel"><h3>Highlights</h3><div class="note">Loading…</div></div>
      <div>
        <div id="rosterbox" class="panel"><h3>Roster</h3><div class="note">Loading…</div></div>
        <div id="jobsbox" class="panel"><h3>Processing runs</h3><div class="note">Loading…</div></div>
      </div>
    </div>`;

  const state = { matchId, roster: [], events: [], templates: [], filter: 'all', highlightIds: new Set() };
  $('#share_match', main).onclick = () => createShare(state, { scope: 'match' }, 'Match share link');
  $('#share_list', main).onclick = () => showShareList(state);
  await Promise.all([
    loadStats(state),
    loadRoster(state).then(() => loadEvents(state)),
    loadJobs(state),
  ]);
}

/* ---------------- sharing ---------------- */

function shareUrl(urlPath) {
  return `${location.origin}${urlPath}`;
}

async function createShare(state, body, title) {
  try {
    const link = await post(`/matches/${state.matchId}/shares`, body);
    const url = shareUrl(link.url_path);
    try {
      await navigator.clipboard.writeText(url);
      toast('Share link copied to clipboard');
    } catch {
      toast('Share link ready');
    }
    const box = $('#sharebox');
    if (box) {
      box.innerHTML = `<div class="panel"><h3>${esc(title)}</h3>
        <input type="text" readonly value="${esc(url)}" id="shareurl">
        <div class="note">Anyone with this link can view it — no account needed.
          <a href="${esc(link.url_path)}" target="_blank" rel="noopener">Open ↗</a></div></div>`;
      $('#shareurl').onclick = (event) => event.target.select();
    }
  } catch (error) { toast(error.message, 'err'); }
}

async function showShareList(state) {
  const box = $('#sharebox');
  if (!box) return;
  try {
    const { items } = await get(`/matches/${state.matchId}/shares`);
    box.innerHTML = `<div class="panel"><h3>Share links</h3>
      ${items.length ? `<div class="tablewrap"><table>
        <thead><tr><th>scope</th><th>label</th><th>views</th><th>link</th><th></th></tr></thead>
        <tbody>${items.map((item) => `
          <tr class="${item.revoked ? 'na' : ''}">
            <td>${esc(item.scope)}</td><td>${esc(item.label || '')}</td><td>${item.view_count}</td>
            <td>${item.revoked ? '<span style="color:var(--dim)">revoked</span>'
              : `<a href="${esc(item.url_path)}" target="_blank" rel="noopener">open ↗</a>`}</td>
            <td>${item.revoked ? '' : `<button class="btn2" data-revoke="${esc(item.share_id)}">Revoke</button>`}</td>
          </tr>`).join('')}
        </tbody></table></div>` : '<div class="note">No share links yet.</div>'}</div>`;
    $$('button[data-revoke]', box).forEach((button) => {
      button.onclick = async () => {
        try {
          await del(`/shares/${button.dataset.revoke}`);
          toast('Share link revoked');
          showShareList(state);
        } catch (error) { toast(error.message, 'err'); }
      };
    });
  } catch (error) {
    box.innerHTML = `<div class="panel"><div class="errnote">${esc(error.message)}</div></div>`;
  }
}

/* ---------------- stats ---------------- */

async function loadStats(state) {
  let box = $('#statsbox');
  if (!box) return;
  try {
    const data = await get(`/matches/${state.matchId}/stats`);
    box = $('#statsbox');
    if (!box) return; // view changed while loading
    const home = data.teams.home || 'Home';
    const away = data.teams.away || 'Away';
    const note = data.analysis.has_completed_job
      ? ''
      : '<div class="warnnote">No completed analysis yet — stats appear after the first processing run finishes.</div>';
    const sourceNote = data.analysis.source_supported_stat_count < 15
      ? `<div class="warnnote">Source: ${esc(data.analysis.source_label)} — only
          ${data.analysis.source_supported_stat_count} of 15 stats are computable from this kind of link.
          Upload the raw file to get the full catalog.</div>`
      : '';
    box.innerHTML = `
      <h3>Team stats</h3>
      <div class="teamsline"><b>${esc(home)}</b><span>vs</span><b>${esc(away)}</b></div>
      <div class="statgrid">${data.stats.map(statTile).join('')}</div>
      <div class="note">Greyed stats can’t be computed from this footage or aren’t detected yet — never shown as a fake 0.
        Click a stat to highlight its evidence in the Highlights list.</div>
      ${sourceNote}${note}`;
    $$('.stat.linked', box).forEach((tile) => {
      tile.onclick = () => {
        const key = tile.dataset.key;
        const stat = data.stats.find((item) => item.key === key);
        const selected = tile.classList.toggle('selected');
        $$('.stat.selected', box).forEach((other) => { if (other !== tile) other.classList.remove('selected'); });
        state.highlightIds = new Set(selected ? (stat?.event_ids || []) : []);
        renderEventsTable(state);
      };
    });
  } catch (error) {
    box.innerHTML = `<h3>Team stats</h3><div class="errnote">${esc(error.message)}</div>`;
  }
}

function statTile(stat) {
  const fmt = (value) => value == null ? '–' : (stat.unit === 'percent' ? `${value}%` : `${value}`);
  if (!stat.available) {
    const why = {
      no_completed_analysis: 'awaiting analysis',
      not_detected_by_pipeline: 'coming soon',
      team_stats_artifact_missing: 'needs a full run',
      not_available_for_source: `not available from ${stat.raw?.source_label || 'this source'}`,
    }[stat.reason] || stat.reason || 'unavailable';
    return `<div class="stat na"><div class="k">${esc(stat.label)}</div>
      <div class="vals"><span>–</span><span class="mid">|</span><span>–</span></div>
      <div class="why">${esc(why)}</div></div>`;
  }
  const linked = (stat.event_ids || []).length ? ' linked' : '';
  const unattributed = stat.unattributed ? `<div class="why">+${stat.unattributed} unattributed</div>` : '';
  return `<div class="stat${linked}" data-key="${esc(stat.key)}" title="${esc(stat.method || '')}">
    <div class="k">${esc(stat.label)}</div>
    <div class="vals"><span>${fmt(stat.home)}</span><span class="mid">home · away</span><span>${fmt(stat.away)}</span></div>
    ${unattributed}</div>`;
}

/* ---------------- roster ---------------- */

async function loadRoster(state) {
  let box = $('#rosterbox');
  if (!box) return;
  try {
    const { items } = await get(`/matches/${state.matchId}/roster`);
    state.roster = items;
    box = $('#rosterbox');
    if (!box) return; // view changed while loading
    let templates = [];
    try { templates = (await get('/roster-templates')).items; } catch { templates = []; }
    state.templates = templates;
    box = $('#rosterbox');
    if (!box) return;

    box.innerHTML = `
      <h3>Roster</h3>
      ${items.length ? `<div class="tablewrap"><table><thead><tr><th>#</th><th>player</th><th>pos</th><th>side</th><th></th></tr></thead>
        <tbody>${items.map((entry) => `
          <tr><td>${esc(entry.jersey_number)}</td><td>${esc(entry.player_name)}</td>
          <td>${esc(entry.position || '')}</td><td>${esc(entry.team_side)}</td>
          <td style="white-space:nowrap">
            <button class="btn2" data-card="${esc(entry.roster_entry_id)}" title="Player card">🪪</button>
            <button class="btn2" data-del="${esc(entry.roster_entry_id)}" title="Remove">✕</button></td></tr>`).join('')}
        </tbody></table></div>` : '<div class="note">No roster yet. Add players so highlights can be routed to them.</div>'}
      <div class="row" style="margin-top:10px">
        <div><label>Player name</label><input type="text" id="r_name" placeholder="Alex Morgan"></div>
        <div><label>Jersey #</label><input type="text" id="r_jersey" placeholder="13"></div>
      </div>
      <div class="row">
        <div><label>Position</label><input type="text" id="r_pos" placeholder="Forward"></div>
        <div><label>Email</label><input type="email" id="r_email" placeholder="player@example.com"></div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px">
        <button class="btn2" id="r_add">Add player</button>
        <button class="btn2" id="r_import">Import CSV…</button>
        <button class="btn2" id="r_template">Template</button>
        <input type="file" id="r_file" accept=".csv,text/csv" style="display:none">
      </div>
      ${items.length ? `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
        <button class="btn2" id="r_route">Route highlights</button>
        <button class="btn2" id="r_cards">Email player cards</button>
        <button class="btn2" id="r_savetpl">Save as team…</button>
      </div>` : ''}
      ${templates.length ? `<div style="margin-top:10px">
        <label>Load a saved team</label>
        <select id="r_tplpick">
          <option value="">— choose a saved roster —</option>
          ${templates.map((tpl) =>
            `<option value="${esc(tpl.template_id)}">${esc(tpl.name)} (${tpl.entry_count})</option>`).join('')}
        </select>
      </div>` : ''}
      <div id="routenote"></div>`;

    $('#r_add', box).onclick = async () => {
      try {
        await post(`/matches/${state.matchId}/roster`, {
          player_name: $('#r_name', box).value.trim(),
          jersey_number: $('#r_jersey', box).value.trim(),
          position: $('#r_pos', box).value.trim() || null,
          email: $('#r_email', box).value.trim() || null,
        });
        toast('Player added');
        await loadRoster(state);
        renderEventsTable(state);
      } catch (error) { toast(error.message, 'err'); }
    };
    $('#r_import', box).onclick = () => $('#r_file', box).click();
    $('#r_file', box).onchange = async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      try {
        const csvText = await file.text();
        const result = await post(`/matches/${state.matchId}/roster/import`, { csv_text: csvText });
        const problems = result.errors.length ? `, ${result.errors.length} rows skipped (${result.errors.map((e) => `line ${e.line}: ${e.issue}`).join('; ')})` : '';
        toast(`Roster import: ${result.created} added, ${result.updated} updated${problems}`, result.errors.length ? 'warn' : 'ok');
        await loadRoster(state);
        renderEventsTable(state);
      } catch (error) { toast(error.message, 'err'); }
    };
    $('#r_template', box).onclick = () =>
      downloadFile('/matches/roster-template.csv', 'roster_template.csv').catch((error) => toast(error.message, 'err'));
    $$('button[data-del]', box).forEach((button) => {
      button.onclick = async () => {
        try {
          const result = await del(`/matches/${state.matchId}/roster/${button.dataset.del}`);
          toast(result.unassigned_events ? `Player removed; ${result.unassigned_events} highlights unassigned` : 'Player removed');
          await loadRoster(state);
          await loadEvents(state);
        } catch (error) { toast(error.message, 'err'); }
      };
    });
    $$('button[data-card]', box).forEach((button) => {
      button.onclick = () => showPlayerCard(state, button.dataset.card);
    });

    const routeButton = $('#r_route', box);
    if (routeButton) {
      routeButton.onclick = async () => {
        try {
          const result = await post(`/matches/${state.matchId}/roster/route`, {});
          const note = $('#routenote');
          if (note) {
            note.innerHTML = result.routed
              ? `<div class="note">Routed ${result.routed} highlight(s) by jersey number.
                  ${result.unassigned_remaining} still unassigned.</div>`
              : `<div class="warnnote">Nothing to route automatically: no highlight carries a recognized jersey
                  number yet${result.unmatched_jersey_numbers.length
                    ? ` (unmatched: ${result.unmatched_jersey_numbers.map(esc).join(', ')})`
                    : ''}. Assign highlights manually in the list on the left.</div>`;
          }
          toast(result.routed ? `Routed ${result.routed} highlights` : 'No jersey numbers to route', result.routed ? 'ok' : 'warn');
          await loadEvents(state);
          loadStats(state);
        } catch (error) { toast(error.message, 'err'); }
      };
    }

    const cardsButton = $('#r_cards', box);
    if (cardsButton) {
      cardsButton.onclick = async () => {
        try {
          const result = await post(`/matches/${state.matchId}/roster/cards/send`, {});
          toast(`Player cards: ${result.sent} sent, ${result.skipped} skipped`, result.sent ? 'ok' : 'warn');
          const note = $('#routenote');
          if (note && result.skipped) {
            const missing = result.details.filter((item) => item.status !== 'sent').map((item) => item.player_name);
            note.innerHTML = `<div class="warnnote">No email on file for: ${missing.map(esc).join(', ')}</div>`;
          }
        } catch (error) { toast(error.message, 'err'); }
      };
    }

    const saveButton = $('#r_savetpl', box);
    if (saveButton) {
      saveButton.onclick = async () => {
        const name = prompt('Save this roster as a reusable team. Name:');
        if (!name) return;
        try {
          await post(`/matches/${state.matchId}/roster/save-template`, { name });
          toast(`Saved "${name}" — reuse it on your next match`);
          await loadRoster(state);
        } catch (error) { toast(error.message, 'err'); }
      };
    }

    const picker = $('#r_tplpick', box);
    if (picker) {
      picker.onchange = async () => {
        if (!picker.value) return;
        try {
          const result = await post(`/matches/${state.matchId}/roster/apply-template/${picker.value}`, {});
          toast(`Loaded roster: ${result.created} added, ${result.skipped} already present`);
          await loadRoster(state);
          renderEventsTable(state);
        } catch (error) { toast(error.message, 'err'); }
      };
    }
  } catch (error) {
    const target = $('#rosterbox');
    if (target) target.innerHTML = `<h3>Roster</h3><div class="errnote">${esc(error.message)}</div>`;
  }
}

async function showPlayerCard(state, entryId) {
  const box = $('#sharebox');
  if (!box) return;
  try {
    const card = await get(`/matches/${state.matchId}/roster/${entryId}/card`);
    box.innerHTML = `<div class="panel">
      <h3>Player card — #${esc(card.jersey_number)} ${esc(card.player_name)}</h3>
      <div class="metrics">
        <div class="metric"><div class="v">${card.highlight_count}</div><div class="l">Highlights</div></div>
        ${(card.stats || []).slice(0, 2).map((stat) =>
          `<div class="metric"><div class="v">${stat.count}</div><div class="l">${esc(stat.label)}</div></div>`).join('')}
      </div>
      <button class="btn2" id="share_card">⬆️ Share this card</button>
      ${card.highlight_count ? '' : '<div class="note">No highlights attributed yet — route or assign highlights first.</div>'}
    </div>`;
    $('#share_card').onclick = () =>
      createShare(state, { scope: 'player_card', roster_entry_id: entryId, label: `Player card: ${card.player_name}` },
        `Player card link — ${card.player_name}`);
  } catch (error) {
    box.innerHTML = `<div class="panel"><div class="errnote">${esc(error.message)}</div></div>`;
  }
}

/* ---------------- events / highlights ---------------- */

async function loadEvents(state) {
  try {
    const { items } = await get(`/matches/${state.matchId}/events?limit=500`);
    state.events = items.sort((a, b) => a.occurred_at_ms - b.occurred_at_ms);
  } catch (error) {
    const box = $('#eventsbox');
    if (box) box.innerHTML = `<h3>Highlights</h3><div class="errnote">${esc(error.message)}</div>`;
    return;
  }
  renderEventsTable(state);
}

function renderEventsTable(state) {
  const box = $('#eventsbox');
  if (!box) return;
  const filtered = state.events.filter((event) => {
    if (state.filter === 'unassigned') return !event.player_id;
    if (state.filter === 'assigned') return !!event.player_id;
    return true;
  });
  const playerOptions = (selected) => ['<option value="">— unassigned —</option>']
    .concat(state.roster.map((entry) =>
      `<option value="${esc(entry.roster_entry_id)}" ${entry.roster_entry_id === selected ? 'selected' : ''}>#${esc(entry.jersey_number)} ${esc(entry.player_name)}</option>`))
    .join('');

  box.innerHTML = `
    <h3>Highlights</h3>
    <div class="viewbar" style="margin:4px 0 10px">
      ${['all', 'unassigned', 'assigned'].map((key) =>
        `<button data-f="${key}" class="${state.filter === key ? 'active' : ''}">${key[0].toUpperCase()}${key.slice(1)}</button>`).join('')}
    </div>
    ${filtered.length ? `<div class="tablewrap"><table>
      <thead><tr><th>at</th><th>type</th><th>conf</th><th>player</th><th></th></tr></thead>
      <tbody>${filtered.map((event) => `
        <tr class="${state.highlightIds.has(event.event_id) ? 'hl' : ''}">
          <td>${fmtMs(event.occurred_at_ms)}</td>
          <td>${esc(event.event_type)}</td>
          <td>${(+event.confidence || 0).toFixed(2)}</td>
          <td><select data-assign="${esc(event.event_id)}" ${state.roster.length ? '' : 'disabled'}>${playerOptions(event.player_id)}</select></td>
          <td><button class="btn2" data-share="${esc(event.event_id)}" title="Share this highlight">⬆️</button></td>
        </tr>`).join('')}
      </tbody></table></div>` : '<div class="note">No highlights match this filter yet.</div>'}
    <div class="note">Unassigned highlights stay shareable — assign them to route stats to the right player.</div>`;

  $$('button[data-f]', box).forEach((button) => {
    button.onclick = () => { state.filter = button.dataset.f; renderEventsTable(state); };
  });
  $$('button[data-share]', box).forEach((button) => {
    button.onclick = () => {
      const event = state.events.find((item) => item.event_id === button.dataset.share);
      createShare(
        state,
        { scope: 'highlight', event_id: button.dataset.share, label: event ? `${event.event_type} highlight` : 'Highlight' },
        'Highlight share link',
      );
    };
  });
  $$('select[data-assign]', box).forEach((select) => {
    select.onchange = async () => {
      const eventId = select.dataset.assign;
      try {
        const updated = await post(`/matches/${state.matchId}/events/${eventId}/assign`, {
          roster_entry_id: select.value || null,
        });
        const index = state.events.findIndex((event) => event.event_id === eventId);
        if (index >= 0) state.events[index] = updated;
        toast(select.value ? 'Highlight assigned' : 'Assignment cleared');
        loadStats(state);
      } catch (error) {
        toast(error.message, 'err');
        renderEventsTable(state);
      }
    };
  });
}

/* ---------------- jobs ---------------- */

async function loadJobs(state) {
  let box = $('#jobsbox');
  if (!box) return;
  try {
    const { items } = await get(`/matches/${state.matchId}/jobs`);
    box = $('#jobsbox');
    if (!box) return; // view changed while loading
    box.innerHTML = `
      <h3>Processing runs</h3>
      ${items.length ? `<div class="tablewrap"><table><thead><tr><th>job</th><th>status</th><th></th></tr></thead>
        <tbody>${items.map((job) => `
          <tr><td style="font-family:monospace;font-size:11px">${esc(job.job_id)}</td>
          <td><span class="status ${esc(job.status)}">${esc(job.status)}</span></td>
          <td>${job.status === 'completed' ? `<a href="#runs/${encodeURIComponent(job.job_id)}">Review film →</a>` : `<a href="#jobs">Queue →</a>`}</td></tr>`).join('')}
        </tbody></table></div>` : '<div class="note">No runs yet — start one from <b>Create</b>.</div>'}`;
  } catch (error) {
    box.innerHTML = `<h3>Processing runs</h3><div class="errnote">${esc(error.message)}</div>`;
  }
}
