# Requirements Traceability

This file confirms coverage of requirements provided in the planning context.

## Context-Window Requirement Coverage

| Context requirement | Coverage in PRD | Roadmap phase | Status |
|---|---|---|---|
| Analyze full-game soccer recordings and generate highlights | `FR-INGEST-01`, `FR-INGEST-03`, `FR-EDITOR-05` | Phase 2, Phase 3 | Captured |
| Re-architecture from desktop/local to web/cloud | PRD Sections 2, 3, 9 | Phase 0, Phase 2 | Captured |
| Dockerized deployment | `NFR-DEPLOY-01` | Phase 1, Phase 2 | Captured |
| GPU acceleration on local host and cloud/edge | `NFR-DEPLOY-02`, `NFR-PLAT-01` | Phase 1, Phase 2, Phase 5 | Captured |
| Multi-client (multi-tenant) isolation with per-tenant users | `FR-TENANT-01`, `FR-TENANT-04`, `NFR-SEC-01A` | Phase 1, Phase 2 | Captured |
| Robust overall admin portal (platform/global admin) | `FR-TENANT-02`, `FR-TENANT-05` | Phase 1, Phase 2 | Captured |
| Separate tenant admin portal per tenant | `FR-TENANT-03`, `FR-TENANT-05` | Phase 1, Phase 2 | Captured |
| Automatic ball tracking | `FR-CAM-03` | Phase 2 | Captured |
| Broadcast-style AI follow-cam from static recording | `FR-CAM-02` | Phase 2 | Captured |
| Panoramic/wide input support (180-degree style) | `FR-CAM-01` | Phase 2 | Captured |
| Dual-lens/4K-class camera input support | `FR-CAM-04` | Phase 2 | Captured |
| Unattended fixed-camera workflow (no operator pan/zoom) | `FR-CAM-05` | Phase 2 | Captured |
| Event detection: goals | `FR-EVENT-01` | Phase 2 | Captured |
| Event detection: shots | `FR-EVENT-02` | Phase 2 | Captured |
| Event detection: corner kicks | `FR-EVENT-03` | Phase 2 | Captured |
| Event detection: penalty kicks | `FR-EVENT-04` | Phase 2 | Captured |
| Event detection: free kicks | `FR-EVENT-05` | Phase 2 | Captured |
| Event detection: goal kicks | `FR-EVENT-06` | Phase 2 | Captured |
| Event detection: kick-offs | `FR-EVENT-07` | Phase 2 | Captured |
| Event detection: fouls | `FR-EVENT-08` | Phase 4 | Captured (phase-gated) |
| Player spotlight clips | `FR-PLAYER-01` | Phase 3 | Captured |
| Player tracking by shirt number | `FR-PLAYER-02` | Phase 4 | Captured |
| Player movement and performance metrics | `FR-PLAYER-04` | Phase 3 | Captured |
| Match momentum graph | `FR-TEAM-01` | Phase 3 | Captured |
| Heatmaps and position analysis | `FR-TEAM-02`, `FR-TEAM-03` | Phase 3 | Captured |
| Interactive editor with panoramic/follow-cam switching | `FR-EDITOR-01` | Phase 3 | Captured |
| Zoom in editor | `FR-EDITOR-02` | Phase 3 | Captured |
| Drawing/annotation tools | `FR-EDITOR-04` | Phase 3 | Captured |
| Custom highlight creation and player tagging | `FR-EDITOR-05`, `FR-EDITOR-06` | Phase 3 | Captured |
| Highlight sharing/export | `FR-EDITOR-07` | Phase 3 | Captured |
| Live streaming support | `FR-LIVE-01` | Phase 4 | Captured (phase-gated) |
| Instant playback/replay bookmarks | `FR-LIVE-03` | Phase 4 | Captured (phase-gated) |
| AI agent API for natural-language analysis support | `FR-AGENT-01`, `FR-AGENT-04` | Phase 3, Phase 5 | Captured |
| Explainable event reasoning from AI assistant | `FR-AGENT-02` | Phase 4 | Captured |
| AI-assisted missed-event review suggestions | `FR-AGENT-03` | Phase 4 | Captured |
| Capture feedback on missed/incorrect events | `FR-LEARN-01` to `FR-LEARN-03` | Phase 3 | Captured |
| Use approved feedback to improve model training | `FR-LEARN-05`, `FR-LEARN-06` | Phase 5 | Captured |

## Notes

1. Fouls, jersey-number workflows, and live features are explicitly planned but phase-gated after MVP.
2. MVP focuses on stable asynchronous full-match processing, follow-cam, base event timeline, and editable exports.

## Market-Leadership Scope Coverage

| Added scope area | Coverage in PRD | Roadmap phase | Status |
|---|---|---|---|
| Data trust layer (calibration, confidence QA, human review) | `FR-QA-01` to `FR-QA-04` | Phase 4 | Captured |
| Season and career intelligence | `FR-LONG-01` to `FR-LONG-03` | Phase 4 | Captured |
| Opponent scouting automation | `FR-SCOUT-01` to `FR-SCOUT-03` | Phase 4 | Captured |
| Coach action engine (drills + plans) | `FR-COACH-01` to `FR-COACH-03` | Phase 6 | Captured |
| Recruiting workflows | `FR-RECRUIT-01` to `FR-RECRUIT-03` | Phase 6 | Captured |
| Distribution and monetization | `FR-DIST-01` to `FR-DIST-04` | Phase 6 | Captured |
| Open platform and integrations | `FR-INT-01` to `FR-INT-04` | Phase 5 | Captured |
| Camera fleet and edge operations | `FR-FLEET-01` to `FR-FLEET-04` | Phase 5 | Captured |
| Collaboration and governance | `FR-COLLAB-01` to `FR-COLLAB-04` | Phase 6 | Captured |
| Enterprise-grade security and compliance | `NFR-SEC-02`, `NFR-SEC-03`, `NFR-COMP-01`, `NFR-DATA-02` | Phase 5 | Captured |
| LLM governance and provider flexibility | `NFR-AGENT-01` to `NFR-AGENT-03` | Phase 5 | Captured |

## Customer FAQ Requirement Coverage (August 2026)

Requirements extracted from the customer-facing FAQ document.

| FAQ requirement | Coverage in PRD | Roadmap phase | Status |
|---|---|---|---|
| Baseline 15 per-team stats every match (goals, assists, possession, shots, shots on target, saves, offsides, passes, pass accuracy, key passes, duels, fouls, corners, free kicks, penalties) | `FR-STATS-01`, `FR-STATS-02` | Phase 3 | Captured |
| Premium off-the-ball and individual player stats | `FR-STATS-03`, `FR-ROSTER-02` | Phase 4 | Captured |
| Raw file uploads preferred; full stat coverage from raw files | `FR-INGEST-05`, `FR-SOURCE-03` | Phase 2, Phase 4 | Captured |
| Public link ingest (YouTube, Vimeo, VEO, Hudl, Pixellot, XbotGo, NBC Sports Engine) | `FR-SOURCE-01` | Phase 4 | Captured |
| Reduced/variable stat coverage from link sources, disclosed to users | `FR-SOURCE-02` to `FR-SOURCE-04`, `FR-STATS-04` | Phase 4 | Captured |
| Drag-and-drop MP4 upload up to 3 GB; paid tier up to 8 GB | `FR-INGEST-05`, `FR-INGEST-06`, `FR-PLAN-02` | Phase 2, Phase 6 | Captured |
| 30-minute minimum video length | `FR-INGEST-07` | Phase 2 | Captured |
| iPhone/iPad and Android filmed video supported | `FR-INGEST-08` | Phase 2 | Captured |
| 4-6 hour average processing turnaround | `NFR-SLA-01`, `FR-UI-12` | Phase 2, Phase 5 | Captured |
| Email notification when match is ready (spam-folder caveat) | `FR-NOTIFY-01`, `FR-NOTIFY-03` | Phase 2, Phase 5 | Captured |
| 100% automated analysis, nothing to learn | `FR-GUIDE-03`, `FR-UI-04` | Phase 2 | Captured |
| Roster upload after team stats (name, jersey number, position, email) via template | `FR-ROSTER-01` | Phase 3 | Captured |
| Automatic highlight routing to players by jersey number | `FR-ROSTER-02`, `FR-PLAYER-02` | Phase 4 | Captured |
| Player card emailed to each rostered player | `FR-ROSTER-03`, `FR-NOTIFY-02` | Phase 4 | Captured |
| Premium player stats ready 24-48 hours after team stats | `NFR-SLA-02` | Phase 5 | Captured |
| Saved roster library in profile (avoid re-entry per match) | `FR-ROSTER-05` | Phase 4 | Captured |
| Single-player roster for one-player highlights | `FR-ROSTER-06` | Phase 3 | Captured |
| Unassigned highlights remain shareable when routing misses | `FR-ROSTER-04`, `FR-SHARE-01` | Phase 3 | Captured |
| Public share link for full match and individual stats | `FR-SHARE-01` | Phase 3 | Captured |
| Download, keep, and re-share results forever after analysis | `FR-STATS-05`, `FR-SHARE-03` | Phase 3 | Captured |
| Direct social posting (Facebook, X, TikTok, Instagram) | `FR-SHARE-02`, `FR-DIST-01` | Phase 6 | Captured |
| Plans tab with self-service upgrade/downgrade and cancellation policy | `FR-PLAN-01`, `FR-PLAN-05` | Phase 6 | Captured |
| Receive-only player accounts (monthly subscription, no uploads) | `FR-PLAN-03` | Phase 6 | Captured |
| Per-match pricing and paid oversize-upload add-on | `FR-PLAN-02`, `FR-PLAN-04` | Phase 6 | Captured |
| Predictive analysis after 3+ matches (xG, xGA, xS, xG + xA, win-draw-loss) | `FR-PREDICT-01` to `FR-PREDICT-03` | Phase 4 | Captured |
| Filming guidance (midfield placement, lighting, 15-45 degree angle, drones to 90 degrees, obstructions) | `FR-GUIDE-01` | Phase 3 | Captured |
| Film quality drives stat quality; 1080p/4K needed for jersey reading | `FR-GUIDE-02`, `FR-QA-01` | Phase 3, Phase 4 | Captured |
| In-app chat bubble and contact-us support | `FR-SUPPORT-01`, `FR-SUPPORT-03` | Phase 5 | Captured |
| Note-taking area under the match viewer | `FR-SUPPORT-02` | Phase 3 | Captured |
| Secure, private-by-default storage; GDPR compliant | `NFR-COMP-02`, `NFR-SEC-01` | Phase 5 | Captured |
| Users must hold legal rights to analyze film | `NFR-COMP-03` | Phase 5 | Captured |

## UI Overhaul Coverage

The current UI is a single-file developer Studio (`frontend/index.html`) with a hardcoded tenant and no authentication UX. These requirements define the production replacement.

| UI work area | Coverage in PRD | Roadmap phase | Status |
|---|---|---|---|
| Production web app replacing single-file Studio | `FR-UI-01` | Phase 2 | Captured |
| Sign-up/login/password-reset, no hardcoded tenant | `FR-UI-02` | Phase 2 | Captured |
| Responsive, mobile-first layouts | `FR-UI-03` | Phase 2 | Captured |
| Guided upload UX (drag-and-drop, resumable, validation, link paste) | `FR-UI-04` | Phase 2 | Captured |
| Stats dashboard with per-stat evidence drilldown | `FR-UI-05` | Phase 3 | Captured |
| Roster management UI with routing status | `FR-UI-06` | Phase 3 | Captured |
| Plans and billing surface | `FR-UI-07` | Phase 6 | Captured |
| Share center (links, social, reports, player cards) | `FR-UI-08` | Phase 3 | Captured |
| Notifications center and email preferences | `FR-UI-09` | Phase 3 | Captured |
| Accessibility (WCAG 2.1 AA) and design system | `FR-UI-10` | Phase 3 | Captured |
| First-run onboarding, empty states, filming tips | `FR-UI-11` | Phase 3 | Captured |
| Stage-level progress and SLA messaging for long jobs | `FR-UI-12` | Phase 2 | Captured |
