// The reel selection (ordered event ids) shared by Watch and Reel tabs.
// Starts from the engine's reel plan; user edits persist per run locally.

const key = (ctx) => `vh_reel_${ctx.runId}`;

export function reelIds(ctx) {
  if (!ctx.reel) {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(ctx)) || 'null'); } catch { saved = null; }
    const known = new Set(ctx.model.events.map((e) => e.id));
    const plan = ctx.model.reelPlan.selected_event_ids || [];
    const base = Array.isArray(saved) ? saved : plan.length ? plan : ctx.model.run.events_source === 'bookmarks' ? [...known] : [];
    ctx.reel = base.filter((id) => known.has(id));
  }
  return ctx.reel;
}

export function setReel(ctx, ids) {
  ctx.reel = [...ids];
  try { localStorage.setItem(key(ctx), JSON.stringify(ctx.reel)); } catch { /* private mode */ }
}

export function toggleReel(ctx, id, on) {
  const ids = reelIds(ctx).filter((x) => x !== id);
  if (on) {
    // keep chronological order when adding from the list
    const t = ctx.model.events.find((e) => e.id === id)?.t ?? 0;
    const at = ids.findIndex((x) => (ctx.model.events.find((e) => e.id === x)?.t ?? 0) > t);
    if (at < 0) ids.push(id); else ids.splice(at, 0, id);
  }
  setReel(ctx, ids);
}

export function resetReel(ctx) {
  try { localStorage.removeItem(key(ctx)); } catch { /* ignore */ }
  ctx.reel = null;
  return reelIds(ctx);
}
