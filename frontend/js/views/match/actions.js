// Actions shared by the match tabs: bridging analysis events to DB events,
// review feedback, on-demand clips, player labels and follow-player reruns.

import { authedBlobUrl, patch, post, put } from '../../api.js';
import { icon } from '../../icons.js';
import { go } from '../../router.js';
import { esc, fmtClock, openDrawer, toast } from '../../ui.js';
import { playerName, typeLabel } from '../../runmodel.js';

// Analysis events live in analysis_events.json; feedback, clips and exports
// work on DB Event rows. Resolve (materializing on demand) the DB id.
export async function ensureDbEvent(ctx, event) {
  if (event.db_event_id) return event.db_event_id;
  if (!ctx.jobId) throw new Error('This run is not linked to a match job, so events cannot be reviewed or clipped.');
  const result = await post(`/studio/runs/${encodeURIComponent(ctx.runId)}/events/${encodeURIComponent(event.id)}/db-event`, {});
  event.db_event_id = result.db_event_id;
  event.db_status = event.db_status || 'auto_detected';
  return result.db_event_id;
}

export async function confirmEvent(ctx, event) {
  const id = await ensureDbEvent(ctx, event);
  await patch(`/matches/${ctx.matchId}/events/${id}`, { status: 'confirmed' });
  event.db_status = 'confirmed';
  toast(`${typeLabel(event.type)} at ${fmtClock(event.t)} confirmed`);
}

export async function rejectEvent(ctx, event) {
  const id = await ensureDbEvent(ctx, event);
  await post(`/matches/${ctx.matchId}/events/${id}/feedback`, {
    feedback_type: 'false_positive',
    severity: 'medium',
    comment: `Rejected in Studio review (${event.type} at ${fmtClock(event.t)})`,
  });
  await patch(`/matches/${ctx.matchId}/events/${id}`, { status: 'rejected' });
  event.db_status = 'rejected';
  toast(`${typeLabel(event.type)} marked as a false positive — thanks, this trains the detector`, 'warn');
}

export async function renderEventClip(ctx, event) {
  const id = await ensureDbEvent(ctx, event);
  const drawer = openDrawer({
    title: `<b>${esc(typeLabel(event.type))}</b> clip`,
    subtitle: `${fmtClock(event.t)} · rendered from the source video`,
    html: `<div class="bar indeterminate"><i></i></div><div class="note">Rendering… this takes a few seconds per clip.</div>`,
  });
  try {
    const clip = await post(`/matches/${ctx.matchId}/events/${id}/clip-on-demand`, {
      pre_seconds: 2, post_seconds: 4, anchor: 'event_window', include_audio: true,
    });
    const url = await authedBlobUrl(`/studio/matches/${ctx.matchId}/assets/${clip.asset_id}/file`);
    drawer.body.innerHTML = `<video src="${url}" controls autoplay playsinline style="width:100%;border-radius:10px;background:#000"></video>
      <dl class="kv" style="margin-top:16px"><dt>Length</dt><dd>${(clip.duration_ms / 1000).toFixed(1)} s</dd>
      <dt>Source window</dt><dd>${fmtClock(clip.start_ms / 1000)} – ${fmtClock(clip.end_ms / 1000)}</dd>
      <dt>Cached</dt><dd>${clip.reused_existing ? 'yes (reused)' : 'new render'}</dd></dl>
      <div class="btnrow" style="margin-top:16px"><a class="btn2" href="${url}" download="${esc(event.type)}_${Math.round(event.t)}s.mp4">${icon('download', 'sm')} Download</a></div>`;
  } catch (error) {
    drawer.body.innerHTML = `<div class="errnote">${icon('alert', 'sm')}<span>${esc(error.message)}</span></div>`;
  }
}

export async function saveLabel(ctx, trackId, label) {
  const next = { ...ctx.model.labels };
  const clean = { name: (label.name || '').trim() || undefined, number: (label.number || '').trim() || undefined };
  if (clean.name || clean.number) next[String(trackId)] = { ...(next[String(trackId)] || {}), ...clean };
  else delete next[String(trackId)];
  const result = await put(`/studio/runs/${encodeURIComponent(ctx.runId)}/player-labels`, { labels: next });
  ctx.model.labels = result.labels || {};
  toast(`Saved ${playerName(ctx.model, trackId)}`);
}

// Config keys the integration layer implements for re-rendering around one
// player without re-detecting (see the Studio hand-off notes).
export function followPlayerOverrides(ctx, trackId, mode) {
  const base = {
    camera_mode: 'follow_player',
    focus_track_id: Number(trackId),
    reuse_tracking_from_job: ctx.jobId,
    // Render into a fresh run folder (job_runner falls back to
    // <output_root>/<new job id>) so the source run stays intact.
    output_dir: null,
  };
  if (mode === 'spotlight') return { ...base, render_full_follow_cam: false, player_spotlight_reel: true, broadcast_reel: true };
  return { ...base, render_full_follow_cam: true };
}

export async function rerunForPlayer(ctx, trackId, mode = 'track') {
  if (!ctx.jobId) throw new Error('This run is not linked to a job; open it from its match to re-render.');
  const job = await post(`/jobs/${ctx.jobId}/rerun`, {
    config_overrides: followPlayerOverrides(ctx, trackId, mode),
    reason: `${mode === 'spotlight' ? 'player_spotlight' : 'track_player'}:${trackId}`,
  });
  toast(mode === 'spotlight'
    ? `Spotlight reel for ${playerName(ctx.model, trackId)} queued — tracking is reused, no re-detection`
    : `Follow-cam for ${playerName(ctx.model, trackId)} queued — tracking is reused, no re-detection`);
  return job;
}

export function openJob(jobId) { go('jobs', jobId); }
