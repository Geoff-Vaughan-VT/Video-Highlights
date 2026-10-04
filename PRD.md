# Product Requirements Document (PRD)

## 1. Purpose

Define the target product requirements for rebuilding Video Highlights as a web/cloud-enabled platform while preserving local and Docker execution options.

## 2. Product Vision

Enable coaches and families to upload full soccer match recordings and receive searchable events, player spotlights, analytics, and editable highlight reels with minimal manual effort.

## 3. Goals

1. Support full-match video ingest and asynchronous processing.
2. Generate follow-cam output from panoramic or wide-field recordings.
3. Detect key match events and produce highlight suggestions.
4. Provide player and team analytics for review workflows.
5. Deliver a Docker-deployable architecture with optional GPU acceleration on local hosts and cloud/edge nodes.
6. Build a decision-intelligence platform for coaching, recruiting, and club operations.
7. Add an AI copilot layer for natural-language analysis and structured feedback-driven model improvement.
8. Deliver a guaranteed baseline match-statistics catalog for every processed match, with premium stat tiers above it.
9. Replace the developer-oriented Studio page with a consumer-grade, responsive web application covering upload, review, sharing, rosters, plans, and notifications.

## 4. Scope and Constraints

### 4.1 In Scope

1. Web application and API platform
2. Background worker architecture (CPU and GPU)
3. Timeline editor and export workflows
4. Batch/offline processing and phase-gated live features
5. Link-based ingest from public video platforms with per-source stat coverage
6. Customer-facing plans, billing, player accounts, and email notifications

### 4.2 Out of Scope for MVP

1. Referee-grade foul adjudication accuracy guarantees
2. Fully autonomous player identity without manual correction path
3. Hard real-time SLA guarantees across all deployment profiles

## 5. Personas

1. Coach: reviews events, creates team clips, shares feedback.
2. Parent: generates player-specific reels and downloads highlights.
3. Club admin: manages teams, users, quotas, and reliability.
4. Player: holds a receive-only account, gets highlights routed by jersey number, and shares a player card with recruiters and agents.

## 6. Functional Requirements

### 6.1 Ingest and Job Management

- `FR-INGEST-01`: Upload full-match videos, including long-duration games.
- `FR-INGEST-02`: Support resumable uploads and durable storage.
- `FR-INGEST-03`: Run processing asynchronously with stage-level progress.
- `FR-INGEST-04`: Persist job metadata, artifacts, and retry history.
- `FR-INGEST-05`: Support drag-and-drop upload of MP4 match files up to a standard size limit (3 GB baseline).
- `FR-INGEST-06`: Support oversize uploads (up to 8 GB) as a paid add-on, with entitlement checks enforced at upload time.
- `FR-INGEST-07`: Enforce a minimum match length (30 minutes) with clear pre-upload validation messaging.
- `FR-INGEST-08`: Accept video filmed on mobile devices (iOS and Android), handling common codecs, frame rates, and orientations.

### 6.2 Camera/Tracking Intelligence

- `FR-CAM-01`: Ingest panoramic or wide-field video (including 180-degree style captures).
- `FR-CAM-02`: Generate automated follow-cam output with AI pan/zoom behavior from fixed-camera recordings.
- `FR-CAM-03`: Track ball and players across the full match timeline.
- `FR-CAM-04`: Support 4K-class source input pipelines, including dual-lens panoramic camera output formats.
- `FR-CAM-05`: Support unattended fixed-camera capture workflows (no dedicated camera operator needed for pan/zoom actions).

### 6.3 Event Detection

- `FR-EVENT-01`: Detect and timestamp goals.
- `FR-EVENT-02`: Detect and timestamp shots.
- `FR-EVENT-03`: Detect and timestamp corner kicks.
- `FR-EVENT-04`: Detect and timestamp penalty kicks.
- `FR-EVENT-05`: Detect and timestamp free kicks.
- `FR-EVENT-06`: Detect and timestamp goal kicks.
- `FR-EVENT-07`: Detect and timestamp kickoffs.
- `FR-EVENT-08`: Detect and timestamp likely fouls with confidence scoring.
- `FR-EVENT-09`: Provide confidence score per event.

### 6.4 Player-Centric Features

- `FR-PLAYER-01`: Generate player spotlight clips.
- `FR-PLAYER-02`: Support jersey-number assisted player identification.
- `FR-PLAYER-03`: Provide manual correction workflow for identity assignment.
- `FR-PLAYER-04`: Track player movement paths and per-player involvement metrics.

### 6.5 Team Analytics

- `FR-TEAM-01`: Provide match momentum visualization over timeline windows.
- `FR-TEAM-02`: Generate team heatmaps.
- `FR-TEAM-03`: Provide position and shape summaries.

### 6.6 Review and Editor

- `FR-EDITOR-01`: Allow switching between panoramic and follow-cam views.
- `FR-EDITOR-02`: Support zoom/pan inspection in review mode.
- `FR-EDITOR-03`: Provide timeline navigation with event jump points.
- `FR-EDITOR-04`: Provide drawing/annotation tools (arrows, shapes, notes).
- `FR-EDITOR-05`: Enable custom highlight creation and trimming.
- `FR-EDITOR-06`: Support tagging players and events on clips.
- `FR-EDITOR-07`: Support export and share flows for clips and montages.

### 6.7 Live and Instant Playback (Phase-Gated)

- `FR-LIVE-01`: Support live streaming input.
- `FR-LIVE-02`: Surface near-real-time event markers.
- `FR-LIVE-03`: Support instant replay bookmarks.

### 6.8 Data Trust and Quality Operations

- `FR-QA-01`: Calibrate event and tracking confidence by competition level, camera angle, and field conditions.
- `FR-QA-02`: Route low-confidence detections to human-review queues.
- `FR-QA-03`: Maintain benchmark datasets and recurring evaluation reports.
- `FR-QA-04`: Expose event evidence frames and rationale metadata for analyst review.

### 6.9 Longitudinal Player and Team Intelligence

- `FR-LONG-01`: Track season and career trends for players and teams.
- `FR-LONG-02`: Provide role-based benchmarks against peer cohorts.
- `FR-LONG-03`: Generate improvement and decline alerts from longitudinal metrics.

### 6.10 Opponent Scouting Automation

- `FR-SCOUT-01`: Generate opponent pattern reports (build-up, transition, and defensive shape).
- `FR-SCOUT-02`: Auto-detect and summarize set-piece tendencies.
- `FR-SCOUT-03`: Produce opponent threat maps and danger-zone summaries.

### 6.11 Coaching Action Engine

- `FR-COACH-01`: Recommend training drills from detected weaknesses with linked video evidence.
- `FR-COACH-02`: Generate weekly coaching plans with clip bundles and priorities.
- `FR-COACH-03`: Track completion and outcomes for coaching action items.

### 6.12 Recruiting and Exposure Workflows

- `FR-RECRUIT-01`: Build verified player profiles with tagged evidence clips.
- `FR-RECRUIT-02`: Provide recruiter-facing share links and controlled access packages.
- `FR-RECRUIT-03`: Support player-to-cohort comparison views for scouting decisions.

### 6.13 Distribution and Monetization

- `FR-DIST-01`: Provide social-first export profiles for major platforms.
- `FR-DIST-02`: Support sponsor overlays and branded templates.
- `FR-DIST-03`: Support fan distribution models (subscription or pay-per-view where applicable).
- `FR-DIST-04`: Apply watermarking and rights controls on distributed assets.

### 6.14 Integrations and Open Platform

- `FR-INT-01`: Provide public APIs and webhooks for downstream systems.
- `FR-INT-02`: Sync roster, schedule, and match metadata with external club tools.
- `FR-INT-03`: Support enterprise authentication integrations (SSO).
- `FR-INT-04`: Provide standards-based data export for analytics interoperability.

### 6.15 Camera Fleet and Edge Operations

- `FR-FLEET-01`: Monitor camera and edge node health status remotely.
- `FR-FLEET-02`: Provide calibration and setup validation diagnostics.
- `FR-FLEET-03`: Support offline capture with deferred upload/sync.
- `FR-FLEET-04`: Provide deployment diagnostics for site-level troubleshooting.

### 6.16 Collaboration and Governance

- `FR-COLLAB-01`: Enforce role-based permissions for coaches, analysts, players, and admins.
- `FR-COLLAB-02`: Support clip approval workflows before publication/sharing.
- `FR-COLLAB-03`: Provide threaded collaboration on annotations and clips.
- `FR-COLLAB-04`: Keep audit trails for edits, approvals, and shares.

### 6.17 AI Agent Copilot

- `FR-AGENT-01`: Provide natural-language match query workflows over detected events and tracked entities.
- `FR-AGENT-02`: Generate explainability summaries for detected events using timestamped evidence artifacts.
- `FR-AGENT-03`: Suggest candidate missed events for analyst review using low-confidence and temporal-context signals.
- `FR-AGENT-04`: Support configurable LLM API provider integration for agent capabilities.
- `FR-AGENT-05`: Keep CV/event models as authoritative event source while the agent acts as reasoning and workflow assistant.

### 6.18 Feedback Capture and Continuous Learning

- `FR-LEARN-01`: Capture structured reviewer feedback for false positives, missed events, wrong timestamps, and misattributions.
- `FR-LEARN-02`: Store feedback with event evidence, reviewer metadata, and decision status for curation.
- `FR-LEARN-03`: Route feedback into labeling/review queues with approval controls.
- `FR-LEARN-04`: Produce periodic quality reports (top misses, top false positives, class-specific error trends).
- `FR-LEARN-05`: Support dataset versioning and retraining triggers from approved feedback cohorts.
- `FR-LEARN-06`: Validate retrained models against benchmark datasets before staged rollout.

### 6.19 Multi-Tenant and Administration

- `FR-TENANT-01`: Support multi-tenant isolation where each tenant has independent matches, jobs, events, feedback, and training records.
- `FR-TENANT-02`: Provide a global admin control plane for tenant lifecycle management (create/update tenants, users, memberships).
- `FR-TENANT-03`: Provide a tenant-admin control plane for tenant-scoped user and role management.
- `FR-TENANT-04`: Enforce tenant membership checks for API access to tenant-scoped resources.
- `FR-TENANT-05`: Support separate admin portal experiences for global admins and tenant admins.

### 6.20 Link-Based Source Ingest

- `FR-SOURCE-01`: Ingest matches from pasted public links: YouTube, Vimeo, VEO, Hudl, Pixellot, XbotGo, and NBC Sports Engine.
- `FR-SOURCE-02`: Maintain a per-source capability matrix defining which statistics are computable from each source type.
- `FR-SOURCE-03`: Disclose expected stat coverage differences at submission time (raw file upload vs each link source), steering users toward raw files.
- `FR-SOURCE-04`: Degrade gracefully when a provider restricts access or changes terms: mark impacted stats as unavailable rather than failing the match.

### 6.21 Match Statistics Catalog

- `FR-STATS-01`: Compute a guaranteed baseline per-team stat catalog for every processed match: goals, assists, possession, total shots, shots on target, saves, offsides, total passes, pass accuracy, key passes, duels (50/50 balls), fouls, corners, free kicks, and penalties.
- `FR-STATS-02`: Time-stamp every counted event so each statistic drills down to video evidence.
- `FR-STATS-03`: Provide premium stat tiers above the baseline, including off-the-ball and individual player statistics.
- `FR-STATS-04`: Mark statistics that cannot be computed for a given source or film quality as unavailable instead of reporting misleading zeros.
- `FR-STATS-05`: Produce downloadable match reports that remain accessible to the customer after analysis (download, keep, and re-share).

### 6.22 Roster Management and Highlight Routing

- `FR-ROSTER-01`: Upload a match roster (player name, jersey number, position, email) after team-level stats complete, including spreadsheet template import.
- `FR-ROSTER-02`: Automatically route highlights and stats to rostered players via jersey-number recognition.
- `FR-ROSTER-03`: Email each rostered player a player card collecting the highlights attributed to them.
- `FR-ROSTER-04`: Keep unattributed highlights accessible on the match for manual assignment or sharing when automatic attribution fails.
- `FR-ROSTER-05`: Store reusable rosters in account profiles so teams do not re-enter players for every match.
- `FR-ROSTER-06`: Support single-player rosters for generating highlights for one player.

### 6.23 Plans, Billing, and Player Accounts

- `FR-PLAN-01`: Offer self-service plan tiers (basic and premium) with an in-app plans surface for upgrade and downgrade.
- `FR-PLAN-02`: Support per-match pricing plus paid add-ons (e.g., oversize upload fee) with billing integration.
- `FR-PLAN-03`: Offer receive-only player accounts (monthly subscription) that cannot upload matches but receive routed highlights and maintain a player card.
- `FR-PLAN-04`: Enforce plan entitlements across stat tiers, upload limits, predictive features, and routing features.
- `FR-PLAN-05`: Support cancellation and billing-policy workflows (commitment terms, outstanding-balance checks).

### 6.24 Sharing and Social Distribution

- `FR-SHARE-01`: Generate public share links for a full match and for individual stats or highlights, viewable without an account.
- `FR-SHARE-02`: Support direct posting of highlights to social platforms (Facebook, X, TikTok, Instagram).
- `FR-SHARE-03`: Persist analyzed results durably so customers can view, download, and re-share them indefinitely after processing.

### 6.25 Predictive Analysis

- `FR-PREDICT-01`: Unlock pre-match prediction models once a team has at least three analyzed matches in the system.
- `FR-PREDICT-02`: Produce xG (expected goals), xGA (expected goals against), xS (expected saves), xG + xA, and win-draw-loss odds.
- `FR-PREDICT-03`: Present predictions alongside the baseline stat catalog with clear model-confidence framing.

### 6.26 Guidance, Onboarding, and Support

- `FR-GUIDE-01`: Provide in-product filming guidance: midfield camera placement, sun/lighting positioning, 15-45 degree elevation (drones up to 90 degrees), obstruction-free framing, and 1080p/4K resolution recommendations.
- `FR-GUIDE-02`: Run upload-time quality checks (resolution, length, format) and warn users when film quality will reduce stat coverage or jersey-number readability.
- `FR-GUIDE-03`: Keep the core workflow zero-training: upload or paste a link, then fully automated processing with no required configuration.
- `FR-SUPPORT-01`: Provide in-app chat and contact channels with queued response handling.
- `FR-SUPPORT-02`: Provide a note-taking area alongside match video review.
- `FR-SUPPORT-03`: Surface FAQ/help content inside the product.

### 6.27 Notifications

- `FR-NOTIFY-01`: Email the customer when match processing completes, with a link to view results.
- `FR-NOTIFY-02`: Email the customer (and rostered players) when premium/individual stats and player cards are ready.
- `FR-NOTIFY-03`: Follow email deliverability best practices (SPF/DKIM/DMARC, sender reputation) to minimize spam-foldering.

### 6.28 Web Application UI Overhaul

The current UI is a single-file developer Studio (`frontend/index.html`) with a hardcoded tenant, no authentication UX, and desktop-only layout. It must be replaced with a production customer experience.

- `FR-UI-01`: Rebuild the frontend as a production-grade web application (componentized codebase, build pipeline, versioned releases) replacing the single-file Studio page.
- `FR-UI-02`: Provide full authentication UX: sign-up, login, password reset, and tenant/team context — no hardcoded tenant identifiers.
- `FR-UI-03`: Deliver responsive, mobile-first layouts across all core flows (upload, status, review, share, rosters).
- `FR-UI-04`: Provide a guided upload experience: drag-and-drop with progress, chunked/resumable uploads, pre-flight validation (size, length, format), and a paste-a-link flow with per-source coverage disclosure.
- `FR-UI-05`: Provide a match stats dashboard presenting the full baseline catalog with per-stat drilldown to time-stamped video evidence and availability flags.
- `FR-UI-06`: Provide roster management UI: template import, inline editing, routing status per player, and saved roster library.
- `FR-UI-07`: Provide a plans and billing surface: plan comparison, upgrade/downgrade, add-on purchases, and invoice history.
- `FR-UI-08`: Provide a share center: public links, social posting, downloadable reports, and player cards.
- `FR-UI-09`: Provide a notifications center and email-preference management.
- `FR-UI-10`: Meet WCAG 2.1 AA accessibility and apply a consistent design system with light/dark theming.
- `FR-UI-11`: Provide first-run onboarding and helpful empty states, including filming tips before the first upload.
- `FR-UI-12`: Present long-running job progress with stage-level status and expected turnaround (SLA) messaging.

## 7. Non-Functional Requirements

- `NFR-DEPLOY-01`: Must run in Docker.
- `NFR-DEPLOY-02`: Must support GPU acceleration for analysis workers.
- `NFR-DEPLOY-03`: Must support CPU-only fallback mode.
- `NFR-PLAT-01`: Deployment options include local PC host, NVIDIA-capable edge host, and cloud infrastructure.
- `NFR-OPS-01`: Provide logs, metrics, and traceable job stages.
- `NFR-SEC-01`: Multi-user access control and project-level isolation.
- `NFR-SEC-01A`: Tenant-level data isolation must prevent cross-tenant read/write access.
- `NFR-SEC-02`: Support MFA and enterprise SSO policy enforcement.
- `NFR-SEC-03`: Provide tamper-evident audit logging for sensitive actions.
- `NFR-DATA-01`: Configurable retention and deletion policy for video assets.
- `NFR-DATA-02`: Support data residency controls for regional deployments.
- `NFR-COMP-01`: Support youth privacy and consent management requirements.
- `NFR-COMP-02`: Comply with GDPR: uploads private by default, accessible only to the owner unless explicitly shared, with data subject rights supported.
- `NFR-COMP-03`: Require and record user attestation of legal rights to analyze uploaded or linked film.
- `NFR-API-01`: Provide versioned API contracts and backward-compatibility policy.
- `NFR-COST-01`: Provide quota and budget guardrails by organization/project.
- `NFR-SLA-01`: Target a 4-6 hour average processing turnaround for uploaded matches and surface actual turnaround to users.
- `NFR-SLA-02`: Deliver premium individual/routing statistics within 24-48 hours after team-level stats complete.
- `NFR-AGENT-01`: Maintain prompt/version traceability for agent outputs used in operational workflows.
- `NFR-AGENT-02`: Enforce guardrails to prevent the agent from mutating authoritative event labels without explicit reviewer approval.
- `NFR-AGENT-03`: Support provider abstraction to allow LLM API vendor switching without major product rewrites.

## 8. Acceptance Criteria (MVP)

1. A user can upload a full match and receive asynchronous job completion without blocking the UI.
2. Completed jobs produce follow-cam output, event timeline, and highlight clips.
3. Event timeline includes timestamp, type, and confidence fields.
4. Users can edit highlights in a timeline UI and export clips/montage.
5. Docker deployment can run CPU workers and GPU workers (`--gpus all`) from the same codebase.
6. Tenant A users cannot read or mutate Tenant B resources through API or portals.
7. Global admins and tenant admins have separate administration surfaces with role-appropriate access.
8. Completed matches display the baseline per-team stat catalog with per-stat availability flags and drilldown to timestamped evidence.
9. The web UI supports an authenticated, responsive upload -> status -> review -> share flow with no hardcoded tenant or developer tooling required.

## 9. Suggested Architecture

1. Web frontend
2. API backend
3. Queue/scheduler
4. GPU analysis workers
5. Rendering workers
6. Object storage
7. Relational database
8. Monitoring/alerting stack

## 10. Success Metrics

1. Time to first playable highlights per match
2. Event precision/recall on labeled validation set
3. Share/export completion rate
4. User edit rate on auto-generated clips
5. Compute cost per processed match
6. Baseline stat coverage rate by ingest source type (raw upload vs each link provider)
7. Jersey-number routing accuracy (highlights attributed to the correct player)
8. Processing turnaround vs the 4-6 hour SLA target

## 11. Risks and Mitigations

1. Camera angle and quality variance
   - Mitigation: calibration profiles, confidence gating, model tiers
2. Long-running job failures
   - Mitigation: checkpointing, retries, resumable stages
3. Player identity ambiguity
   - Mitigation: assisted jersey pipeline and manual assignment
4. Compute cost pressure
   - Mitigation: autoscaling, queue priority classes, fast mode profiles
5. Link-source access variability (provider terms and access change over time)
   - Mitigation: per-source capability matrix, submission-time coverage disclosure, graceful stat degradation
6. Email deliverability for completion notices and player cards
   - Mitigation: authenticated sending domain, deliverability monitoring, in-app notifications as fallback
