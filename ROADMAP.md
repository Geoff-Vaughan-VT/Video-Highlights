# Roadmap

This roadmap aligns with `PRD.md` and targets a web/cloud platform with Dockerized CPU/GPU workers.

## Phase 0: Planning and Architecture (1-2 weeks)

Deliverables:

1. Final PRD sign-off and requirement IDs lock
2. Architecture decision records (API, queue, storage, worker model)
3. Baseline labeled dataset and evaluation metrics

Exit criteria:

1. MVP scope approved
2. Build/deploy strategy approved for local and cloud

## Phase 1: Pipeline Refactor and Container Baseline (2-3 weeks)

Deliverables:

1. Extract processing pipeline into worker-friendly modules
2. Introduce job schema for input, config, status, and artifacts
3. Build container strategy for `cpu` and `gpu` images

Exit criteria:

1. Pipeline can run independently from UI
2. Repeatable Docker builds for CPU/GPU targets

## Phase 2: Cloud MVP Platform (4-6 weeks)

Scope focus:

- `FR-INGEST-*`
- `FR-CAM-01` to `FR-CAM-05`
- `FR-EVENT-01` to `FR-EVENT-07`
- `FR-EVENT-09`
- `FR-UI-01` to `FR-UI-04`
- `FR-UI-12`
- `FR-NOTIFY-01`
- `FR-GUIDE-03`
- `NFR-DEPLOY-*`

Deliverables:

1. API for uploads, jobs, and output retrieval
2. Queue + worker orchestration with retries
3. Object storage and relational database integration
4. Production web app foundation replacing the single-file Studio page: auth UX, responsive layout, guided upload (drag-and-drop, size/length validation), and stage-level job progress
5. Completion email notifications

Exit criteria:

1. Full-match upload and async processing works end-to-end
2. Follow-cam, timeline, and base highlight outputs are produced
3. A customer can complete upload -> status -> review without developer tooling or hardcoded tenant context

## Phase 3: Editor and Analytics (3-5 weeks)

Scope focus:

- `FR-EDITOR-*`
- `FR-TEAM-*`
- `FR-PLAYER-01`
- `FR-PLAYER-04`
- `FR-STATS-01` to `FR-STATS-05`
- `FR-ROSTER-01`
- `FR-ROSTER-04`
- `FR-ROSTER-06`
- `FR-SHARE-01`
- `FR-SHARE-03`
- `FR-UI-05` to `FR-UI-06`
- `FR-UI-08` to `FR-UI-11`
- `FR-GUIDE-01` to `FR-GUIDE-02`
- `FR-SUPPORT-02`
- `FR-LEARN-01` to `FR-LEARN-03`
- `FR-AGENT-01`

Deliverables:

1. Timeline editor with event navigation and clip adjustments
2. Annotation tools and clip tagging
3. Momentum graph, heatmaps, and position summaries
4. Baseline per-team stat catalog (15 stats) with availability flags and evidence drilldown dashboard
5. Roster template import with manual assignment and single-player roster support
6. Public share links, durable downloadable reports, and share center UI
7. Filming guidance content, upload-time quality warnings, and match notes area
8. Onboarding, empty states, accessibility (WCAG 2.1 AA), and design system pass
9. Export presets (clip, montage, player package)
10. Structured reviewer feedback capture in editor workflows
11. Basic natural-language event query assistant

Exit criteria:

1. Coach review workflow is complete in web UI
2. Team analytics and editable highlights are shippable
3. Every completed match presents the baseline stat catalog with per-stat drilldown and share links

## Phase 4: Advanced Player Intelligence and Live (4-8 weeks)

Scope focus:

- `FR-EVENT-08`
- `FR-PLAYER-02`
- `FR-PLAYER-03`
- `FR-SOURCE-01` to `FR-SOURCE-04`
- `FR-ROSTER-02`
- `FR-ROSTER-03`
- `FR-ROSTER-05`
- `FR-PREDICT-01` to `FR-PREDICT-03`
- `FR-NOTIFY-02`
- `FR-LIVE-*`
- `FR-QA-01` to `FR-QA-04`
- `FR-LONG-01` to `FR-LONG-03`
- `FR-SCOUT-01` to `FR-SCOUT-03`
- `FR-AGENT-02` to `FR-AGENT-03`
- `FR-LEARN-04`

Deliverables:

1. Jersey-assisted player identification pipeline with manual correction UX
2. Automatic highlight routing to rostered players, player card emails, and saved roster library
3. Link-based ingest (YouTube, Vimeo, VEO, Hudl, Pixellot, XbotGo, NBC Sports Engine) with per-source capability matrix and coverage disclosure
4. Predictive analysis (xG, xGA, xS, xG + xA, win-draw-loss) unlocked at three analyzed matches
5. Foul candidate detection with confidence markers
6. Live stream ingestion and near-real-time marker pipeline
7. Instant replay bookmarks
8. Confidence calibration and human-review queue workflow
9. Season trend dashboards and opponent scouting report generation
10. Agent explainability summaries and review-priority suggestions
11. Automated error trend reporting

Exit criteria:

1. Advanced features reach target quality thresholds
2. Live feature set validated for supported deployment profiles

## Phase 5: Hardening and Scale (ongoing)

Scope focus:

- `NFR-OPS-01`
- `NFR-SEC-01` to `NFR-SEC-03`
- `NFR-DATA-01` to `NFR-DATA-02`
- `NFR-PLAT-01`
- `NFR-COMP-01`
- `NFR-COST-01`
- `FR-INT-01` to `FR-INT-04`
- `FR-FLEET-01` to `FR-FLEET-04`
- `FR-SUPPORT-01`
- `FR-SUPPORT-03`
- `FR-NOTIFY-03`
- `FR-AGENT-04`
- `FR-LEARN-05` to `FR-LEARN-06`
- `NFR-AGENT-01` to `NFR-AGENT-03`
- `NFR-SLA-01` to `NFR-SLA-02`
- `NFR-COMP-02` to `NFR-COMP-03`

Deliverables:

1. Autoscaling and queue priority controls
2. Production observability and alerting
3. Retention/deletion policy enforcement
4. Multi-tenant hardening and cost controls
5. Turnaround SLA tracking (4-6 hour team stats, 24-48 hour premium routing) with user-facing status
6. In-app chat/contact support channels and in-product FAQ surface
7. GDPR compliance controls, rights attestation capture, and email deliverability hardening
8. Open API and webhook ecosystem
9. Edge/camera fleet diagnostics and remote operations controls
10. LLM provider abstraction and governance controls
11. Feedback-driven retraining pipeline with staged rollout gates

Exit criteria:

1. Stable operations with measurable SLOs
2. Predictable cost and reliability under load

## Phase 6: Coaching, Recruiting, and Commercial Expansion (ongoing)

Scope focus:

- `FR-COACH-01` to `FR-COACH-03`
- `FR-RECRUIT-01` to `FR-RECRUIT-03`
- `FR-DIST-01` to `FR-DIST-04`
- `FR-PLAN-01` to `FR-PLAN-05`
- `FR-SHARE-02`
- `FR-UI-07`
- `FR-COLLAB-01` to `FR-COLLAB-04`
- `NFR-API-01`

Deliverables:

1. Drill recommendation and coaching action-plan engine
2. Recruiting profile, comparison, and verified sharing workflows
3. Self-service plans and billing: tiers, per-match pricing, oversize-upload add-on, receive-only player accounts, cancellation workflows
4. Direct social posting (Facebook, X, TikTok, Instagram) and distribution templates with sponsor overlays and rights controls
5. Collaboration threads, approval workflow, and audit history

Exit criteria:

1. Coaching and recruiting workflows produce measurable user retention gains
2. Distribution and monetization features are production-ready for target customer segments

## Milestones

### M1 (End of Phase 2)

1. Full-match upload with validation (size, minimum length, mobile-filmed video)
2. Async jobs with retries
3. Follow-cam + base event timeline + highlight export
4. Production web app foundation (auth UX, responsive, guided upload) with completion emails
5. Docker deployment with CPU/GPU worker support

### M2 (End of Phase 3)

1. Full editor workflow
2. Baseline 15-stat catalog with evidence drilldown and share links in UI
3. Team analytics in UI
4. Roster import, manual assignment, and player spotlight export improvements
5. Reviewer feedback capture and basic AI query assistant available

### M3 (End of Phase 4)

1. Jersey-assisted identity workflow with automatic highlight routing and player cards
2. Link-based ingest with per-source stat coverage disclosure
3. Predictive analysis (xG family and win-draw-loss)
4. Foul candidate markers
5. Live and instant replay capabilities for supported environments
6. Confidence review workflows and scouting reports live

### M4 (End of Phase 5)

1. Open API/webhook integrations
2. Fleet and edge operations tooling
3. Turnaround SLA tracking and in-app support channels live
4. Security/compliance (including GDPR and rights attestation) and data residency controls enforced
5. LLM provider abstraction and retraining governance operational

### M5 (End of Phase 6)

1. Coaching action engine and recruiting suite released
2. Plans, billing, and receive-only player accounts launched
3. Collaboration governance and commercial distribution toolkit (including social posting) launched

## Suggested Team Shape

1. Backend/platform engineer
2. CV/ML engineer
3. Frontend engineer
4. DevOps/platform engineer
5. QA + data labeling analyst
