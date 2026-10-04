// Reel builder: target length presets, auto-pick by excitement, toggle and
// drag-reorder events, export via /matches/{id}/exports/highlights, and the
// engine's rendered reel + previous exports.

import { authedBlobUrl, post, runFileUrl } from '../../api.js';
import { icon } from '../../icons.js';
import { teamName } from '../../runmodel.js';
import { $, $$, esc, fmtClock, fmtDuration, fmtRel, toast } from '../../ui.js';
import { ensureDbEvent } from './actions.js';
import { reelIds, resetReel, setReel } from './reelstate.js';

const PRESETS = [[60, '1 min'], [180, '3 min'], [300, '5 min'], [600, '10 min']];

function clipLength(e, pre, post) {
  const start = e.t_start ?? e.t - 2;
  const end = e.t_end ?? e.t + 4;
  return Math.max(1, end - start) + pre + post;
}

export function renderReelTab(body, ctx) {
  const model = ctx.model;
  const byId = new Map(model.events.map((e) => [e.id, e]));
  const st = ctx.reelOpts || (ctx.reelOpts = { target: +model.reelPlan.target_duration_s || 180, pre: 1.5, post: 2, title: '' });
  const exports = (ctx.match?.metadata?.highlight_exports || []).slice(-5).reverse();
  const reelVideo = model.run.videos?.reel;

  body.innerHTML = `<div class="reel">
    <div>
      <div class="panel">
        <div class="charthead"><div class="panel-title" style="margin:0">Your reel</div>
          <div class="btnrow"><button class="btn-ghost sm" id="rl_reset">${icon('rerun', 'sm')} Engine pick</button>
          <button class="btn-ghost sm" id="rl_clear">${icon('x', 'sm')} Clear</button></div></div>
        <div class="charthead">
          <div class="seg" role="group" aria-label="Target length" id="rl_presets">
            ${PRESETS.map(([s, l]) => `<button data-s="${s}">${l}</button>`).join('')}
            <button data-s="custom">Custom</button></div>
          <div class="btnrow"><input type="number" id="rl_custom" min="10" max="3600" step="10" style="width:96px" aria-label="Custom length in seconds" value="${st.target}">
            <button class="btn2 sm" id="rl_auto">${icon('sparkle', 'sm')} Auto-pick to length</button></div>
        </div>
        <div id="rl_sum" style="margin:12px 0"></div>
        <div id="rl_list"></div>
      </div>
    </div>
    <div>
      <div class="panel">
        <div class="panel-title">Export</div>
        <label for="rl_title">Title</label><input type="text" id="rl_title" placeholder="${esc(ctx.match?.name || 'Match')} highlights" value="${esc(st.title)}">
        <div class="row"><div><label for="rl_pre">Lead-in (s)</label><input type="number" id="rl_pre" min="0" max="20" step="0.5" value="${st.pre}"></div>
        <div><label for="rl_post">Tail (s)</label><input type="number" id="rl_post" min="0" max="20" step="0.5" value="${st.post}"></div></div>
        <button class="btn" id="rl_export" style="width:100%;margin-top:16px" ${ctx.matchId ? '' : 'disabled title="Needs a run linked to a match"'}>${icon('film', 'sm')} Export reel</button>
        <div id="rl_out"></div>
      </div>
      ${reelVideo ? `<div class="panel"><div class="panel-title">Engine reel <span class="faint xs">from this run</span></div>
        <video src="${esc(runFileUrl(model.runId, reelVideo))}" controls preload="metadata" style="width:100%;border-radius:8px;background:#000"></video></div>` : ''}
      ${exports.length ? `<div class="panel"><div class="panel-title">Previous exports</div>${exports.map((x) => `
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)">
          <div><div class="small" style="font-weight:600">${esc(x.title || 'Highlights')}</div><div class="faint xs">${x.clip_count} clips · ${fmtDuration((x.duration_ms || 0) / 1000)} · ${esc(fmtRel(x.created_at))}</div></div>
          <button class="iconbtn" data-asset="${esc(x.asset_id)}" aria-label="Play export">${icon('play', 'sm')}</button></div>`).join('')}</div>` : ''}
      <div class="panel flush"><div class="panel-title">Add events <span class="faint xs">by excitement</span></div><div class="evlist reelpool" id="rl_pool" style="max-height:520px"></div></div>
    </div></div>`;

  const ids = () => reelIds(ctx).filter((id) => byId.has(id));
  const pre = () => Math.max(0, +$('#rl_pre', body).value || 0);
  const postS = () => Math.max(0, +$('#rl_post', body).value || 0);

  function summary() {
    const list = ids().map((id) => byId.get(id));
    const total = list.reduce((s, e) => s + clipLength(e, pre(), postS()), 0);
    const over = total > st.target * 1.1;
    $('#rl_sum', body).innerHTML = `<div class="charthead"><div><b class="num" style="font-size:20px">${fmtDuration(total)}</b>
      <span class="faint small">of ${fmtDuration(st.target)} target · ${list.length} clips</span></div>
      ${over ? `<span class="badge warn">${fmtDuration(total - st.target)} over</span>` : list.length ? '<span class="badge ok">Fits</span>' : ''}</div>
      <div class="durbar" aria-hidden="true">${list.map((e) => `<i style="width:${(100 * clipLength(e, pre(), postS())) / Math.max(total, st.target)}%;background:${e.info.color}" title="${esc(e.info.label)} ${fmtClock(e.t)}"></i>`).join('')}</div>`;
    $$('#rl_presets [data-s]', body).forEach((b) => b.classList.toggle('active', b.dataset.s === String(st.target) || (b.dataset.s === 'custom' && !PRESETS.some(([s]) => s === st.target))));
  }

  function renderList() {
    const list = ids();
    $('#rl_list', body).innerHTML = list.length ? list.map((id, i) => {
      const e = byId.get(id);
      return `<div class="reelitem" draggable="true" data-id="${esc(id)}">
        <span class="grip" title="Drag to reorder">${icon('grip', 'sm')}</span><span class="idx">${i + 1}</span>
        <div style="min-width:0"><div class="small" style="font-weight:600;display:flex;gap:8px;align-items:center"><span class="swatch" style="background:${e.info.color}"></span>${esc(e.info.label)}
          <span class="faint xs">${esc(teamName(model, e.team))}</span></div>
          <div class="faint xs">${fmtClock(e.t)} · ${fmtDuration(clipLength(e, pre(), postS()))} · excitement ${Math.round((+e.excitement || 0) * 100)}%</div></div>
        <div class="btnrow" style="gap:0"><button class="iconbtn" data-mv="-1" aria-label="Move up" ${i ? '' : 'disabled'}>${icon('up', 'sm')}</button>
          <button class="iconbtn" data-mv="1" aria-label="Move down" ${i < list.length - 1 ? '' : 'disabled'}>${icon('down', 'sm')}</button></div>
        <button class="iconbtn" data-rm aria-label="Remove from reel">${icon('x', 'sm')}</button></div>`;
    }).join('') : '<div class="empty" style="padding:32px">No clips yet. Add events from the list, or auto-pick to a target length.</div>';
    const chosen = new Set(list);
    const pool = model.events.filter((e) => !chosen.has(e.id)).sort((a, b) => (b.excitement || 0) - (a.excitement || 0));
    $('#rl_pool', body).innerHTML = pool.length ? pool.map((e) => `<div class="ev" data-id="${esc(e.id)}" style="--c:${e.info.color}">
      <button class="play" data-add aria-label="Add ${esc(e.info.label)} to reel">${icon('plus', 'sm')}</button>
      <div><div class="ttl"><span class="tdot"></span>${esc(e.info.label)} <span class="faint xs">${esc(teamName(model, e.team))}</span></div>
        <div class="meta"><span class="tc">${fmtClock(e.t)}</span><span class="meter"><i style="width:${Math.round((+e.excitement || 0) * 100)}%"></i></span></div></div>
      <div></div></div>`).join('') : '<div class="note" style="padding:16px">Every event is in the reel.</div>';
    summary();
    bindDrag();
  }

  function bindDrag() {
    let dragId = null;
    $$('.reelitem', body).forEach((item) => {
      item.ondragstart = (event) => { dragId = item.dataset.id; item.classList.add('dragging'); event.dataTransfer.effectAllowed = 'move'; };
      item.ondragend = () => { item.classList.remove('dragging'); $$('.reelitem.over', body).forEach((n) => n.classList.remove('over')); };
      item.ondragover = (event) => { event.preventDefault(); item.classList.add('over'); };
      item.ondragleave = () => item.classList.remove('over');
      item.ondrop = (event) => {
        event.preventDefault();
        const list = ids().filter((x) => x !== dragId);
        const at = list.indexOf(item.dataset.id);
        const rect = item.getBoundingClientRect();
        list.splice(event.clientY > rect.top + rect.height / 2 ? at + 1 : at, 0, dragId);
        setReel(ctx, list);
        renderList();
      };
    });
  }

  function autoPick() {
    const pool = [...model.events].filter((e) => e.db_status !== 'rejected').sort((a, b) => (b.excitement || 0) - (a.excitement || 0));
    const chosen = [];
    let total = 0;
    for (const e of pool) {
      const len = clipLength(e, pre(), postS());
      if (total + len > st.target && chosen.length) continue;
      chosen.push(e);
      total += len;
      if (total >= st.target * 0.95) break;
    }
    chosen.sort((a, b) => a.t - b.t);
    setReel(ctx, chosen.map((e) => e.id));
    renderList();
    toast(`Picked ${chosen.length} clips · ${fmtDuration(total)}`);
  }

  body.addEventListener('click', (event) => {
    const item = event.target.closest('.reelitem');
    if (item && event.target.closest('[data-rm]')) { setReel(ctx, ids().filter((x) => x !== item.dataset.id)); renderList(); return; }
    const mv = event.target.closest('[data-mv]');
    if (item && mv) {
      const list = ids();
      const i = list.indexOf(item.dataset.id);
      const j = i + +mv.dataset.mv;
      [list[i], list[j]] = [list[j], list[i]];
      setReel(ctx, list);
      renderList();
      return;
    }
    const add = event.target.closest('[data-add]');
    if (add) { setReel(ctx, [...ids(), add.closest('.ev').dataset.id]); renderList(); return; }
    const preset = event.target.closest('#rl_presets [data-s]');
    if (preset) {
      if (preset.dataset.s === 'custom') { $('#rl_custom', body).focus(); return; }
      st.target = +preset.dataset.s;
      $('#rl_custom', body).value = st.target;
      summary();
    }
    const asset = event.target.closest('[data-asset]');
    if (asset) playAsset(asset.dataset.asset);
  });
  $('#rl_custom', body).onchange = (event) => { st.target = Math.max(10, +event.target.value || 60); summary(); };
  $('#rl_pre', body).oninput = () => { st.pre = pre(); renderList(); };
  $('#rl_post', body).oninput = () => { st.post = postS(); renderList(); };
  $('#rl_title', body).oninput = (event) => { st.title = event.target.value; };
  $('#rl_auto', body).onclick = autoPick;
  $('#rl_reset', body).onclick = () => { resetReel(ctx); renderList(); };
  $('#rl_clear', body).onclick = () => { setReel(ctx, []); renderList(); };

  async function playAsset(assetId) {
    const out = $('#rl_out', body);
    out.innerHTML = '<div class="bar indeterminate" style="margin-top:16px"><i></i></div>';
    try {
      const url = await authedBlobUrl(`/studio/matches/${ctx.matchId}/assets/${assetId}/file`);
      out.innerHTML = `<video src="${url}" controls autoplay style="width:100%;border-radius:8px;margin-top:16px;background:#000"></video>
        <a class="btn2 sm" style="margin-top:8px" href="${url}" download="highlights.mp4">${icon('download', 'sm')} Download</a>`;
    } catch (error) { out.innerHTML = `<div class="errnote">${esc(error.message)}</div>`; }
  }

  $('#rl_export', body).onclick = async (event) => {
    const button = event.currentTarget;
    const list = ids();
    if (!list.length) { toast('Add at least one clip to the reel first', 'warn'); return; }
    button.disabled = true;
    const out = $('#rl_out', body);
    out.innerHTML = `<div style="margin-top:16px"><div class="small" id="rl_stage">Preparing ${list.length} clips…</div><div class="bar indeterminate" style="margin-top:8px"><i></i></div></div>`;
    try {
      const eventIds = [];
      for (const id of list) eventIds.push(await ensureDbEvent(ctx, byId.get(id)));
      $('#rl_stage', body).textContent = `Rendering ${list.length} clips from the source and joining them…`;
      const result = await post(`/matches/${ctx.matchId}/exports/highlights`, {
        event_ids: eventIds, pre_seconds: pre(), post_seconds: postS(), anchor: 'event_window', include_audio: true,
        title: $('#rl_title', body).value.trim() || `${ctx.match?.name || 'Match'} highlights`,
      });
      toast(`Reel exported · ${result.clip_count} clips`);
      const url = await authedBlobUrl(`/studio/matches/${ctx.matchId}/assets/${result.asset_id}/file`);
      out.innerHTML = `<div class="panel" style="margin:16px 0 0;padding:12px"><div class="small" style="font-weight:600">${result.clip_count} clips · ${fmtDuration(result.duration_ms / 1000)}</div>
        <div class="faint xs mono" style="margin:4px 0 8px;word-break:break-all">${esc(result.path)}</div>
        <video src="${url}" controls style="width:100%;border-radius:8px;background:#000"></video>
        <a class="btn2 sm" style="margin-top:8px" href="${url}" download="${esc(result.export_id)}.mp4">${icon('download', 'sm')} Download</a></div>`;
    } catch (error) {
      out.innerHTML = `<div class="errnote">${icon('alert', 'sm')}<span>${esc(error.message)}</span></div>`;
    } finally {
      button.disabled = false;
    }
  };

  renderList();
}
