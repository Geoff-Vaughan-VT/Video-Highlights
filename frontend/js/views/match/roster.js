// Roster & sharing tab: roster CRUD + CSV import + saved teams, highlight
// routing and player cards, event-to-player assignment, and share links.

import { del, downloadFile, get, post } from '../../api.js';
import { icon } from '../../icons.js';
import { $, $$, confirmDialog, copyText, esc, fmtDate, fmtMs, promptDialog, toast } from '../../ui.js';

function shareUrl(urlPath) { return `${location.origin}${urlPath}`; }

async function createShare(ctx, body, title) {
  try {
    const link = await post(`/matches/${ctx.matchId}/shares`, body);
    const url = shareUrl(link.url_path);
    await copyText(url, 'Share link copied to clipboard');
    const box = $('#sharebox');
    if (box) {
      box.innerHTML = `<div class="panel" style="border-color:var(--accent)"><div class="panel-title">${esc(title)}</div>
        <div class="btnrow"><input type="text" readonly value="${esc(url)}" id="shareurl" style="flex:1">
        <button class="btn2" id="sharecopy">${icon('copy', 'sm')} Copy</button>
        <a class="btn-ghost" href="${esc(link.url_path)}" target="_blank" rel="noopener">${icon('external', 'sm')} Open</a></div>
        <div class="note">Anyone with this link can view it — no account needed.</div></div>`;
      $('#shareurl').onclick = (event) => event.target.select();
      $('#sharecopy').onclick = () => copyText(url);
    }
    showShareList(ctx);
  } catch (error) { toast(error.message, 'err'); }
}

async function showShareList(ctx) {
  const box = $('#sharelist');
  if (!box) return;
  try {
    const { items } = await get(`/matches/${ctx.matchId}/shares`);
    box.innerHTML = items.length ? `<div class="tablewrap"><table>
      <thead><tr><th>Scope</th><th>Label</th><th class="r">Views</th><th>Created</th><th></th></tr></thead>
      <tbody>${items.map((item) => `<tr class="${item.revoked ? 'na' : ''}">
        <td>${esc(item.scope.replace('_', ' '))}</td><td>${esc(item.label || '—')}</td><td class="r">${item.view_count}</td>
        <td class="faint">${esc(fmtDate(item.created_at))}</td>
        <td class="r">${item.revoked ? '<span class="badge">revoked</span>'
          : `<a class="btn-ghost sm" href="${esc(item.url_path)}" target="_blank" rel="noopener">${icon('external', 'sm')} Open</a>
             <button class="btn-ghost sm" data-copy="${esc(item.url_path)}">${icon('copy', 'sm')}</button>
             <button class="btn-ghost sm danger" data-revoke="${esc(item.share_id)}">Revoke</button>`}</td></tr>`).join('')}
      </tbody></table></div>` : '<div class="note">No share links yet.</div>';
    $$('[data-copy]', box).forEach((b) => { b.onclick = () => copyText(shareUrl(b.dataset.copy)); });
    $$('[data-revoke]', box).forEach((button) => {
      button.onclick = async () => {
        if (!await confirmDialog({ title: 'Revoke this link?', body: 'Anyone holding it will lose access immediately.', confirmLabel: 'Revoke', danger: true })) return;
        try { await del(`/shares/${button.dataset.revoke}`); toast('Share link revoked'); showShareList(ctx); } catch (error) { toast(error.message, 'err'); }
      };
    });
  } catch (error) {
    box.innerHTML = `<div class="errnote">${esc(error.message)}</div>`;
  }
}

async function loadRoster(ctx) {
  const box = $('#rosterbox');
  if (!box) return;
  try {
    const [{ items }, templates] = await Promise.all([
      get(`/matches/${ctx.matchId}/roster`),
      get('/roster-templates').then((r) => r.items).catch(() => []),
    ]);
    ctx.roster = items;
    if (!box.isConnected) return;
    box.innerHTML = `<div class="charthead"><div class="panel-title" style="margin:0">Roster <span class="faint xs">${items.length} players</span></div>
      <div class="btnrow">
        <button class="btn-ghost sm" id="r_import">${icon('upload', 'sm')} Import CSV</button>
        <button class="btn-ghost sm" id="r_template">${icon('download', 'sm')} Template</button>
        ${items.length ? `<button class="btn-ghost sm" id="r_savetpl">${icon('tag', 'sm')} Save as team</button>` : ''}
        <input type="file" id="r_file" accept=".csv,text/csv" hidden></div></div>
      ${items.length ? `<div class="tablewrap"><table><thead><tr><th>#</th><th>Player</th><th>Position</th><th>Side</th><th></th></tr></thead>
        <tbody>${items.map((entry) => `<tr><td class="num"><b>${esc(entry.jersey_number)}</b></td><td>${esc(entry.player_name)}</td>
          <td class="muted">${esc(entry.position || '—')}</td><td><span class="badge">${esc(entry.team_side)}</span></td>
          <td class="r" style="white-space:nowrap">
            <button class="iconbtn" data-card="${esc(entry.roster_entry_id)}" title="Player card" aria-label="Player card for ${esc(entry.player_name)}">${icon('card', 'sm')}</button>
            <button class="iconbtn" data-del="${esc(entry.roster_entry_id)}" title="Remove" aria-label="Remove ${esc(entry.player_name)}">${icon('trash', 'sm')}</button></td></tr>`).join('')}
        </tbody></table></div>` : '<div class="note">No roster yet. Add players so highlights can be routed to them and player cards shared.</div>'}
      <div class="row3" style="margin-top:8px;grid-template-columns:2fr 1fr 1.4fr">
        <div><label for="r_name">Player name</label><input type="text" id="r_name" placeholder="Alex Morgan"></div>
        <div><label for="r_jersey">Shirt #</label><input type="text" id="r_jersey" placeholder="13"></div>
        <div><label for="r_pos">Position</label><input type="text" id="r_pos" placeholder="Forward"></div></div>
      <div class="row" style="grid-template-columns:2fr 1fr;align-items:end">
        <div><label for="r_email">Email (for player cards)</label><input type="email" id="r_email" placeholder="player@example.com"></div>
        <button class="btn2" id="r_add">${icon('plus', 'sm')} Add player</button></div>
      ${templates.length ? `<label for="r_tplpick">Load a saved team</label><select id="r_tplpick"><option value="">Choose a saved roster…</option>
        ${templates.map((tpl) => `<option value="${esc(tpl.template_id)}">${esc(tpl.name)} (${tpl.entry_count})</option>`).join('')}</select>` : ''}
      ${items.length ? `<div class="divider"></div><div class="btnrow">
        <button class="btn2 sm" id="r_route">${icon('users', 'sm')} Route highlights by shirt number</button>
        <button class="btn2 sm" id="r_cards">${icon('mail', 'sm')} Email player cards</button></div>` : ''}
      <div id="routenote"></div>`;
    bindRoster(ctx, box);
  } catch (error) {
    box.innerHTML = `<div class="panel-title">Roster</div><div class="errnote">${esc(error.message)}</div>`;
  }
}

function bindRoster(ctx, box) {
  const refresh = async () => { await loadRoster(ctx); await loadEvents(ctx); };
  $('#r_add', box).onclick = async () => {
    try {
      await post(`/matches/${ctx.matchId}/roster`, {
        player_name: $('#r_name', box).value.trim(), jersey_number: $('#r_jersey', box).value.trim(),
        position: $('#r_pos', box).value.trim() || null, email: $('#r_email', box).value.trim() || null,
      });
      toast('Player added');
      await refresh();
    } catch (error) { toast(error.message, 'err'); }
  };
  $('#r_import', box).onclick = () => $('#r_file', box).click();
  $('#r_file', box).onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const result = await post(`/matches/${ctx.matchId}/roster/import`, { csv_text: await file.text() });
      const problems = result.errors.length ? `, ${result.errors.length} rows skipped (${result.errors.map((e) => `line ${e.line}: ${e.issue}`).join('; ')})` : '';
      toast(`Roster import: ${result.created} added, ${result.updated} updated${problems}`, result.errors.length ? 'warn' : 'ok');
      await refresh();
    } catch (error) { toast(error.message, 'err'); }
  };
  $('#r_template', box).onclick = () => downloadFile('/matches/roster-template.csv', 'roster_template.csv').catch((error) => toast(error.message, 'err'));
  $$('[data-del]', box).forEach((button) => {
    button.onclick = async () => {
      if (!await confirmDialog({ title: 'Remove this player?', body: 'Highlights assigned to them become unassigned.', confirmLabel: 'Remove', danger: true })) return;
      try {
        const result = await del(`/matches/${ctx.matchId}/roster/${button.dataset.del}`);
        toast(result.unassigned_events ? `Player removed; ${result.unassigned_events} highlights unassigned` : 'Player removed');
        await refresh();
      } catch (error) { toast(error.message, 'err'); }
    };
  });
  $$('[data-card]', box).forEach((button) => { button.onclick = () => showPlayerCard(ctx, button.dataset.card); });
  const route = $('#r_route', box);
  if (route) {
    route.onclick = async () => {
      try {
        const result = await post(`/matches/${ctx.matchId}/roster/route`, {});
        $('#routenote').innerHTML = result.routed
          ? `<div class="note">Routed ${result.routed} highlight(s) by shirt number. ${result.unassigned_remaining} still unassigned.</div>`
          : `<div class="warnnote">${icon('info', 'sm')}<span>Nothing to route automatically: no highlight carries a recognised shirt number yet${result.unmatched_jersey_numbers.length
            ? ` (unmatched: ${result.unmatched_jersey_numbers.map(esc).join(', ')})` : ''}. Assign highlights manually below.</span></div>`;
        toast(result.routed ? `Routed ${result.routed} highlights` : 'No shirt numbers to route', result.routed ? 'ok' : 'warn');
        await loadEvents(ctx);
      } catch (error) { toast(error.message, 'err'); }
    };
  }
  const cards = $('#r_cards', box);
  if (cards) {
    cards.onclick = async () => {
      try {
        const result = await post(`/matches/${ctx.matchId}/roster/cards/send`, {});
        toast(`Player cards: ${result.sent} sent, ${result.skipped} skipped`, result.sent ? 'ok' : 'warn');
        if (result.skipped) {
          const missing = result.details.filter((item) => item.status !== 'sent').map((item) => item.player_name);
          $('#routenote').innerHTML = `<div class="warnnote">${icon('mail', 'sm')}<span>No email on file for: ${missing.map(esc).join(', ')}</span></div>`;
        }
      } catch (error) { toast(error.message, 'err'); }
    };
  }
  const save = $('#r_savetpl', box);
  if (save) {
    save.onclick = async () => {
      const name = await promptDialog({ title: 'Save roster as a team', label: 'Team name', confirmLabel: 'Save team' });
      if (!name) return;
      try { await post(`/matches/${ctx.matchId}/roster/save-template`, { name }); toast(`Saved "${name}" — reuse it on your next match`); await loadRoster(ctx); } catch (error) { toast(error.message, 'err'); }
    };
  }
  const picker = $('#r_tplpick', box);
  if (picker) {
    picker.onchange = async () => {
      if (!picker.value) return;
      try {
        const result = await post(`/matches/${ctx.matchId}/roster/apply-template/${picker.value}`, {});
        toast(`Loaded roster: ${result.created} added, ${result.skipped} already present`);
        await refresh();
      } catch (error) { toast(error.message, 'err'); }
    };
  }
}

async function showPlayerCard(ctx, entryId) {
  const box = $('#sharebox');
  if (!box) return;
  try {
    const card = await get(`/matches/${ctx.matchId}/roster/${entryId}/card`);
    box.innerHTML = `<div class="panel"><div class="pcard-head" style="margin-bottom:12px"><div class="jersey" style="--c:var(--surface-3)">${esc(card.jersey_number)}</div>
      <div><h3>${esc(card.player_name)}</h3><div class="faint xs">${esc(card.position || '')}${card.team_name ? ` · ${esc(card.team_name)}` : ''}</div></div></div>
      <div class="metrics"><div class="metric"><div class="v">${card.highlight_count}</div><div class="l">Highlights</div></div>
        ${(card.stats || []).slice(0, 2).map((stat) => `<div class="metric"><div class="v">${stat.count}</div><div class="l">${esc(stat.label)}</div></div>`).join('')}</div>
      <button class="btn2" id="share_card">${icon('share', 'sm')} Share this card</button>
      ${card.highlight_count ? '' : '<div class="note">No highlights attributed yet — route or assign highlights first.</div>'}</div>`;
    $('#share_card').onclick = () => createShare(ctx, { scope: 'player_card', roster_entry_id: entryId, label: `Player card: ${card.player_name}` }, `Player card link — ${card.player_name}`);
  } catch (error) {
    box.innerHTML = `<div class="panel"><div class="errnote">${esc(error.message)}</div></div>`;
  }
}

async function loadEvents(ctx) {
  const box = $('#assignbox');
  if (!box) return;
  try {
    const { items } = await get(`/matches/${ctx.matchId}/events?limit=500`);
    ctx.dbEvents = items;
    renderAssign(ctx);
  } catch (error) {
    box.innerHTML = `<div class="errnote">${esc(error.message)}</div>`;
  }
}

function renderAssign(ctx) {
  const box = $('#assignbox');
  if (!box) return;
  const filter = ctx.assignFilter || 'all';
  const roster = ctx.roster || [];
  const events = [...(ctx.dbEvents || [])].sort((a, b) => a.occurred_at_ms - b.occurred_at_ms)
    .filter((e) => (filter === 'unassigned' ? !e.player_id : filter === 'assigned' ? !!e.player_id : true));
  const options = (selected) => ['<option value="">Unassigned</option>'].concat(roster.map((entry) =>
    `<option value="${esc(entry.roster_entry_id)}" ${entry.roster_entry_id === selected ? 'selected' : ''}>#${esc(entry.jersey_number)} ${esc(entry.player_name)}</option>`)).join('');
  box.innerHTML = `<div class="charthead"><div class="panel-title" style="margin:0">Highlight attribution</div>
    <div class="seg">${['all', 'unassigned', 'assigned'].map((key) => `<button data-f="${key}" class="${filter === key ? 'active' : ''}">${key[0].toUpperCase()}${key.slice(1)}</button>`).join('')}</div></div>
    ${events.length ? `<div class="tablewrap"><table><thead><tr><th>Source time</th><th>Type</th><th>Status</th><th>Player</th><th></th></tr></thead>
      <tbody>${events.map((event) => `<tr><td class="num">${fmtMs(event.occurred_at_ms)}</td><td>${esc(event.event_type.replace(/_/g, ' '))}</td>
        <td><span class="badge ${event.status === 'confirmed' ? 'ok' : event.status === 'rejected' ? 'danger' : ''}">${esc(event.status.replace('_', ' '))}</span></td>
        <td><select data-assign="${esc(event.event_id)}" ${roster.length ? '' : 'disabled'} aria-label="Assign player" style="padding:5px 8px;font-size:13px">${options(event.player_id)}</select></td>
        <td class="r"><button class="iconbtn" data-share="${esc(event.event_id)}" title="Share this highlight" aria-label="Share highlight">${icon('share', 'sm')}</button></td></tr>`).join('')}
      </tbody></table></div>` : '<div class="note">No highlights in the database for this filter yet.</div>'}
    <div class="note">Unassigned highlights stay shareable — assign them to route stats to the right player.</div>`;
  $$('[data-f]', box).forEach((button) => { button.onclick = () => { ctx.assignFilter = button.dataset.f; renderAssign(ctx); }; });
  $$('[data-share]', box).forEach((button) => {
    button.onclick = () => {
      const event = ctx.dbEvents.find((item) => item.event_id === button.dataset.share);
      createShare(ctx, { scope: 'highlight', event_id: button.dataset.share, label: event ? `${event.event_type} highlight` : 'Highlight' }, 'Highlight share link');
    };
  });
  $$('select[data-assign]', box).forEach((select) => {
    select.onchange = async () => {
      try {
        const updated = await post(`/matches/${ctx.matchId}/events/${select.dataset.assign}/assign`, { roster_entry_id: select.value || null });
        const index = ctx.dbEvents.findIndex((e) => e.event_id === select.dataset.assign);
        if (index >= 0) ctx.dbEvents[index] = updated;
        toast(select.value ? 'Highlight assigned' : 'Assignment cleared');
      } catch (error) { toast(error.message, 'err'); renderAssign(ctx); }
    };
  });
}

export function renderRosterTab(body, ctx, options = {}) {
  body.innerHTML = `<div class="cols-2">
    <div><div class="panel" id="rosterbox"><div class="sk sk-line"></div><div class="sk sk-line"></div></div>
      <div class="panel" id="assignbox"><div class="sk sk-line"></div></div></div>
    <div><div class="panel"><div class="charthead"><div class="panel-title" style="margin:0">Sharing</div>
        <button class="btn" id="share_match">${icon('share', 'sm')} Share match</button></div>
        <div class="note" style="margin:0 0 12px">Public, read-only links for parents, recruiters and scouts. Revoke any time.</div>
        <div id="sharelist"><div class="sk sk-line"></div></div></div>
      <div id="sharebox"></div></div></div>`;
  $('#share_match', body).onclick = () => createShare(ctx, { scope: 'match' }, 'Match share link');
  showShareList(ctx);
  loadRoster(ctx).then(() => loadEvents(ctx));
  if (options.share) createShare(ctx, { scope: 'match' }, 'Match share link');
}
