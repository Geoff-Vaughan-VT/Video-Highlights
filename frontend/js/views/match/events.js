// Event list for the Watch tab: filters (type / team / player / reel),
// excitement + confidence, play, add-to-reel, confirm/reject, render clip.

import { icon } from '../../icons.js';
import { playerName, teamName } from '../../runmodel.js';
import { $, $$, esc, fmtClock, toast } from '../../ui.js';
import { confirmEvent, rejectEvent, renderEventClip } from './actions.js';
import { reelIds, toggleReel } from './reelstate.js';

export function createEventList(root, ctx, { onPlay, onFilter }) {
  const model = ctx.model;
  const f = ctx.eventFilter || (ctx.eventFilter = { type: '', team: '', player: '', reel: false });
  const types = [...new Set(model.events.map((e) => e.type))];
  const playersWithEvents = [...new Set(model.events.map((e) => e.player_track_id).filter((v) => v != null))];
  const state = { current: null, visible: [] };

  root.innerHTML = `<div class="panel flush evpanel">
    <div class="panel-title"><span>Events <span class="faint" id="evcount"></span></span>
      <span class="faint xs" title="Source of the list">${model.run.events_source === 'bookmarks' ? 'from bookmarks (legacy)' : ''}</span></div>
    <div class="evfilters">
      <div class="row" style="gap:8px">
        <select id="f_type" aria-label="Event type"><option value="">All types</option>
          ${types.map((t) => `<option value="${esc(t)}">${esc(model.events.find((e) => e.type === t).info.label)}</option>`).join('')}</select>
        <select id="f_team" aria-label="Team"><option value="">Both teams</option>
          ${model.teams.map((t) => `<option value="${t.idx}">${esc(t.name)}</option>`).join('')}</select>
      </div>
      <div class="row" style="gap:8px;align-items:center">
        <select id="f_player" aria-label="Player"><option value="">All players</option>
          ${playersWithEvents.map((id) => `<option value="${id}">${esc(playerName(model, id))}</option>`).join('')}</select>
        <label class="check"><input type="checkbox" id="f_reel"> In reel only</label>
      </div>
    </div>
    <div class="evlist" id="evlist" role="list"></div>
  </div>`;

  $('#f_type', root).value = f.type;
  $('#f_team', root).value = f.team;
  $('#f_player', root).value = f.player;
  $('#f_reel', root).checked = f.reel;

  const matches = (e) => (!f.type || e.type === f.type)
    && (f.team === '' || String(e.team) === f.team)
    && (f.player === '' || String(e.player_track_id) === f.player)
    && (!f.reel || reelIds(ctx).includes(e.id));

  function statusBadge(e) {
    if (e.db_status === 'confirmed') return '<span class="badge ok">Confirmed</span>';
    if (e.db_status === 'rejected') return '<span class="badge danger">Rejected</span>';
    return '';
  }

  function itemHtml(e) {
    const inReel = reelIds(ctx).includes(e.id);
    const conf = Math.round((+e.confidence || 0) * 100);
    const exc = Math.round((+e.excitement || 0) * 100);
    return `<div class="ev ${e.db_status === 'rejected' ? 'rejected' : ''}" role="listitem" data-id="${esc(e.id)}" style="--c:${e.info.color}">
      <button class="play" data-play aria-label="Play ${esc(e.info.label)} at ${fmtClock(e.t)}">${icon('play', 'sm')}</button>
      <div style="min-width:0">
        <div class="ttl"><span class="tdot"></span>${esc(e.info.label)}
          ${e.team === 0 || e.team === 1 ? `<span class="badge"><span class="swatch" style="background:${esc(model.teams[e.team].color)}"></span>${esc(teamName(model, e.team))}</span>` : ''}
          ${statusBadge(e)}</div>
        ${e.reason ? `<div class="why">${esc(e.reason)}</div>` : ''}
        <div class="meta"><span class="tc" title="Source timecode ${fmtClock(e.t + model.trim)}">${fmtClock(e.t)}</span>
          ${e.player_track_id != null ? `<span>${icon('user', 'sm')} ${esc(playerName(model, e.player_track_id))}</span>` : ''}
          <span title="Excitement ${exc}%"><span class="meter"><i style="width:${exc}%"></i></span></span>
          <span title="Detector confidence ${conf}%">${conf}%</span></div>
      </div>
      <div class="acts">
        <label class="reelbox" title="Include in the highlights reel"><input type="checkbox" data-reel ${inReel ? 'checked' : ''}> Reel</label>
        <div class="row-btns">
          <button class="iconbtn" data-ok title="Confirm" aria-label="Confirm event">${icon('check', 'sm')}</button>
          <button class="iconbtn" data-no title="Reject (false positive)" aria-label="Reject event">${icon('x', 'sm')}</button>
          <button class="iconbtn" data-clip title="Render clip" aria-label="Render clip">${icon('scissors', 'sm')}</button>
        </div>
      </div></div>`;
  }

  function render() {
    state.visible = model.events.filter(matches);
    $('#evcount', root).textContent = `${state.visible.length}${state.visible.length !== model.events.length ? ` of ${model.events.length}` : ''}`;
    $('#evlist', root).innerHTML = state.visible.length ? state.visible.map(itemHtml).join('')
      : `<div class="empty" style="margin:16px;padding:24px">${model.events.length ? 'No events match these filters.' : 'No events were detected in this run.'}</div>`;
    onFilter?.(new Set(state.visible.map((e) => e.id)));
    markCurrent(state.current, false);
  }

  function markCurrent(id, scroll = true) {
    state.current = id;
    $$('.ev', root).forEach((node) => node.classList.toggle('cur', node.dataset.id === id));
    if (scroll && id) $(`.ev[data-id="${CSS.escape(id)}"]`, root)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  const busy = async (button, fn) => {
    button.disabled = true;
    try { await fn(); } catch (error) { toast(error.message, 'err'); } finally { button.disabled = false; }
  };

  $('#evlist', root).addEventListener('click', (event) => {
    const row = event.target.closest('.ev');
    if (!row) return;
    const e = model.events.find((x) => x.id === row.dataset.id);
    if (event.target.closest('[data-reel]')) { toggleReel(ctx, e.id, event.target.checked); return; }
    if (event.target.closest('.reelbox')) return;
    const ok = event.target.closest('[data-ok]');
    const no = event.target.closest('[data-no]');
    const clip = event.target.closest('[data-clip]');
    if (ok) return busy(ok, async () => { await confirmEvent(ctx, e); render(); });
    if (no) return busy(no, async () => { await rejectEvent(ctx, e); render(); });
    if (clip) return busy(clip, () => renderEventClip(ctx, e));
    markCurrent(e.id, false);
    onPlay(e);
  });
  for (const [id, key] of [['#f_type', 'type'], ['#f_team', 'team'], ['#f_player', 'player']]) {
    $(id, root).onchange = (event) => { f[key] = event.target.value; render(); };
  }
  $('#f_reel', root).onchange = (event) => { f.reel = event.target.checked; render(); };

  render();
  return {
    get visible() { return state.visible; },
    // Highlight the event whose window contains t (or the last one before it).
    update(t) {
      if (t == null) return;
      let cur = null;
      for (const e of state.visible) {
        const start = e.t_start ?? e.t - 3;
        if (start <= t + 0.25) cur = e; else break;
      }
      if (cur && t > (cur.t_end ?? cur.t + 5) + 2) cur = null;
      const id = cur?.id || null;
      if (id !== state.current) markCurrent(id);
    },
    setPlayerFilter(trackId) {
      f.player = trackId == null ? '' : String(trackId);
      const sel = $('#f_player', root);
      if (sel && ![...sel.options].some((o) => o.value === f.player)) {
        sel.insertAdjacentHTML('beforeend', `<option value="${esc(f.player)}">${esc(playerName(model, trackId))}</option>`);
      }
      if (sel) sel.value = f.player;
      render();
    },
    refresh: render,
  };
}
