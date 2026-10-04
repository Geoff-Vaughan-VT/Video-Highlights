// Runs: raw run folders are browsed from the library ("Runs without a
// match"); a run opens in the full match review. Linked runs redirect to
// their match so there is one review page, not two.

import { renderMatches, renderRunReview } from './matches.js';

export function renderRuns() {
  return renderMatches();
}

export function renderRunDetail(runId, query) {
  return renderRunReview(runId, query);
}
