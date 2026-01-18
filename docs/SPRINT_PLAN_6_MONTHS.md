# Sprint Plan - 6 Months to Product Readiness

**Duration:** 26 weeks (13 sprints)
**Sprint Length:** 2 weeks
**Team Size:** 3-4 developers
**Target:** Production Launch by Week 26

---

## Visual Timeline

```
Week:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
       ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
       │ S1    │ S2    │ S3    │ S4    │ S5    │ S6    │ S7    │ S8    │ S9    │ S10   │ S11   │ S12   │ S13   │
       └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
       │       │               │               │               │       │               │               │
       M1      M2              M3              M4              M5      M6              M7              GA
       Demo    Auth            Governance      Discovery       Agents  Jira            Beta            Launch

Phase: ├─FOUNDATION─────────┼─GOVERNANCE────────────────────┼─AGENTS────────────────┼─PROD───────────────────┤
```

---

## Sprint-by-Sprint Plan

### Sprint 1 (Weeks 1-2) - Demo Fix ✅
**Goal:** Make the Salesforce demo fully functional
**Capacity:** 14 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Register Flask Blueprints | 1 | Dev 1 | ✅ |
| Move Credentials to .env | 2 | Dev 1 | ✅ |
| Run Data Ingestion | 3 | Dev 2 | ✅ |
| Index Salesforce Docs | 5 | Dev 2 | ✅ |
| End-to-End Testing | 3 | Dev 3 | ✅ |

**Exit Criteria:**
- [x] All API endpoints return 200
- [x] RAG queries return real responses (170MB FAISS index exists)
- [x] Voice + text queries both work
- [x] No hardcoded credentials in code

**MILESTONE M1: Demo Working** ✅ (14/14 points complete)

---

### Sprint 2 (Weeks 3-4) - Authentication
**Goal:** Users can register and login
**Capacity:** 25 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| User Database Model | 3 | Dev 1 | ✅ |
| Password Hashing | 2 | Dev 1 | ✅ |
| JWT Token Generation | 3 | Dev 1 | ✅ |
| Auth Middleware | 3 | Dev 2 | ✅ |
| Login API | 3 | Dev 2 | ✅ |
| Register API | 3 | Dev 2 | ✅ |
| Connect Login UI | 3 | Dev 3 | ✅ |
| Google OAuth | 5 | Dev 3 | ⬜ |

**Exit Criteria:**
- [x] Users can register with email
- [x] Users can login and receive JWT
- [x] Protected routes require valid token
- [ ] Google OAuth flow works

**MILESTONE M2: Auth Live** (Partial - 20/25 points complete, OAuth pending)

---

### Sprint 3 (Weeks 5-6) - Multi-Tenancy
**Goal:** Organizations can be created and users isolated
**Capacity:** 21 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Organization Model | 5 | Dev 1 | ✅ |
| User-Org Relationship | 3 | Dev 1 | ✅ |
| Organization APIs | 5 | Dev 2 | ✅ |
| Data Isolation Middleware | 5 | Dev 2 | ✅ |
| Org Selector UI | 3 | Dev 3 | ✅ |

**Exit Criteria:**
- [x] Organizations can be created
- [x] Users belong to organizations
- [x] All data filtered by org_id
- [x] Users can switch organizations

**MILESTONE: Sprint 3 Complete** (21/21 points)

---

### Sprint 4 (Weeks 7-8) - Dashboard & Harbyx Setup
**Goal:** Complete dashboard + start Harbyx integration
**Capacity:** 21 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Dashboard Home | 5 | Dev 3 | ✅ |
| Session History | 5 | Dev 3 | ✅ |
| Usage Statistics API | 3 | Dev 2 | ✅ |
| Harbyx SDK Setup | 3 | Dev 1 | ⬜ |
| Policy Configuration | 5 | Dev 1 | ⬜ |

**Exit Criteria:**
- [x] Dashboard shows real data
- [x] Session history visible
- [ ] Harbyx SDK installed and configured
- [ ] Type A/B/C policies created

**Progress: 13/21 points complete**

---

### Sprint 5 (Weeks 9-10) - Harbyx Core
**Goal:** @governed_action decorator working
**Capacity:** 18 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| @governed_action Decorator | 8 | Dev 2 | ⬜ |
| Handle ALLOW/BLOCK/REQUIRE | 5 | Dev 2 | ⬜ |
| Settings Page | 5 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Decorator wraps functions
- [ ] Actions evaluated by Harbyx
- [ ] ALLOW → action proceeds
- [ ] BLOCK → exception raised
- [ ] REQUIRE_APPROVAL → wait for decision

**MILESTONE M3: Governance MVP**

---

### Sprint 6 (Weeks 11-12) - Approval UI + Knowledge Foundation
**Goal:** Approvers can view and act on pending approvals + Knowledge layer foundation
**Capacity:** 28 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Approval Inbox Page | 8 | Dev 3 | ⬜ |
| Approval Card Component | 5 | Dev 3 | ⬜ |
| Harbyx Webhook Handler | 5 | Dev 1 | ⬜ |
| Real-time Updates | 5 | Dev 2 | ⬜ |
| Knowledge Base Model | 5 | Dev 1 | ⬜ |

**Exit Criteria:**
- [ ] Pending approvals listed
- [ ] Approve/reject buttons work
- [ ] Webhooks received and processed
- [ ] UI updates in real-time
- [ ] Knowledge base schema created

---

### Sprint 7 (Weeks 13-14) - Audit + Workflow + Knowledge
**Goal:** Audit logs visible, workflow engine started, knowledge patterns
**Capacity:** 31 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Audit Log Viewer | 5 | Dev 3 | ⬜ |
| Audit Search & Filters | 3 | Dev 3 | ⬜ |
| Workflow State Machine | 8 | Dev 1 | ⬜ |
| Phase Gate UI | 5 | Dev 3 | ⬜ |
| Pattern Library | 5 | Dev 2 | ⬜ |
| Template Management | 5 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Audit logs searchable
- [ ] BUILD workflow phases defined
- [ ] Projects can move through phases
- [ ] Visual timeline in UI
- [ ] Pattern library operational
- [ ] Template management system ready

---

### Sprint 8 (Weeks 15-16) - Backlog + Agent Registry + RUN Workflow
**Goal:** Backlog management + agent infrastructure + RUN workflow foundation
**Capacity:** 34 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Backlog Item Model | 3 | Dev 1 | ⬜ |
| Backlog CRUD API | 5 | Dev 1 | ⬜ |
| Kanban Board UI | 8 | Dev 3 | ⬜ |
| Agent Registry | 5 | Dev 2 | ⬜ |
| RUN Workflow State Machine | 8 | Dev 1 | ⬜ |
| RUN Triage UI | 5 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Backlog items can be created
- [ ] Kanban board functional
- [ ] 4 agent types registered
- [ ] RUN workflow state machine operational

**MILESTONE M4: Platform Foundation Complete**

---

### Sprint 9 (Weeks 17-18) - Discovery Agent + Human Roles + Knowledge
**Goal:** First agent operational + role management + knowledge injection
**Capacity:** 40 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Agent Orchestrator | 8 | Dev 2 | ⬜ |
| Platform Scanner | 13 | Dev 1+2 | ⬜ |
| Discovery Report | 3 | Dev 3 | ⬜ |
| Agent Knowledge Injection | 8 | Dev 1 | ⬜ |
| Low-Risk Auto-Execute | 5 | Dev 2 | ⬜ |
| RUN Incident Reports | 3 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Discovery Agent scans Salesforce orgs
- [ ] As-Is snapshot generated
- [ ] Report PDF/HTML output
- [ ] Action governed by Harbyx Type A
- [ ] Agents access knowledge layer during execution

**MILESTONE M5: First Agent Live**

---

### Sprint 10 (Weeks 19-20) - Scoping + Designer + Roles + Templates
**Goal:** Requirements to design pipeline + roles + doc templates
**Capacity:** 41 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Requirement Analyzer | 8 | Dev 1 | ⬜ |
| Epic/Story Generator | 8 | Dev 2 | ⬜ |
| Design Document Generator | 10 | Dev 1+2 | ⬜ |
| Human Role Models | 5 | Dev 1 | ⬜ |
| Role Assignment per Project | 3 | Dev 1 | ⬜ |
| Role-Based Approval Routing | 5 | Dev 2 | ⬜ |
| Project Brief Template | 2 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Requirements parsed and analyzed
- [ ] Epics/Stories auto-generated
- [ ] Design documents created
- [ ] All governed by Type B policy
- [ ] 5 human roles defined and operational
- [ ] Role-based approval routing working

---

### Sprint 11 (Weeks 21-22) - Delivery Agent + Jira + Templates
**Goal:** Deployment agent + Jira integration + remaining templates
**Capacity:** 43 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Enhanced MCP Integration | 8 | Dev 1 | ⬜ |
| Deployment Orchestration | 5 | Dev 2 | ⬜ |
| Type C Governance | 3 | Dev 1 | ⬜ |
| Jira OAuth | 5 | Dev 3 | ⬜ |
| Jira Sync (basic) | 5 | Dev 3 | ⬜ |
| As-Is Snapshot Template | 3 | Dev 3 | ⬜ |
| Solution Design Template | 3 | Dev 3 | ⬜ |
| Test Plan Template | 2 | Dev 3 | ⬜ |
| Release Plan Template | 2 | Dev 3 | ⬜ |
| Go-Live Report Template | 2 | Dev 3 | ⬜ |
| Runbook Template | 2 | Dev 3 | ⬜ |
| Incident Report Template | 2 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Delivery Agent executes deployments
- [ ] Production requires Type C approval
- [ ] Jira connected
- [ ] Basic issue sync working
- [ ] All 8 documentation templates complete

**MILESTONE M6: Core Agents + Jira**

---

### Sprint 12 (Weeks 23-24) - Analytics + Security + GDPR
**Goal:** Metrics dashboard + security hardening + GDPR compliance
**Capacity:** 40 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Delivery Metrics Dashboard | 8 | Dev 3 | ⬜ |
| Usage Tracking | 5 | Dev 2 | ⬜ |
| Security Audit | 8 | Dev 1 | ⬜ |
| Rate Limiting | 3 | Dev 2 | ⬜ |
| Data Localization | 5 | DevOps | ⬜ |
| Right to Erasure | 5 | Dev 2 | ⬜ |
| Data Export | 3 | Dev 2 | ⬜ |
| Consent Management | 3 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Metrics dashboard live
- [ ] Usage tracked per org
- [ ] Security audit passed
- [ ] Rate limits enforced
- [ ] GDPR: EU data localization working
- [ ] GDPR: User data deletion API operational
- [ ] GDPR: Data export available

**MILESTONE M7: Beta Ready (GDPR Compliant)**

---

### Sprint 13 (Weeks 25-26) - Launch
**Goal:** Production deployment + launch
**Capacity:** 23 points

| Story | Points | Owner | Status |
|-------|--------|-------|--------|
| Kubernetes Deployment | 8 | DevOps | ⬜ |
| CI/CD Pipeline | 5 | DevOps | ⬜ |
| Landing Page | 5 | Dev 3 | ⬜ |
| Beta Testing | 3 | All | ⬜ |
| Documentation | 2 | Dev 3 | ⬜ |

**Exit Criteria:**
- [ ] Deployed to production K8s
- [ ] CI/CD automated
- [ ] Landing page live
- [ ] 5 beta customers validated
- [ ] User docs available

**MILESTONE GA: Production Launch**

---

## Summary by Phase (Rebalanced)

| Phase | Sprints | Points | Key Deliverables |
|-------|---------|--------|------------------|
| Foundation | 1-2 | 39 | Demo fix, Auth |
| Multi-Tenant | 3 | 21 | Organizations |
| Governance | 4-7 | 98 | Harbyx, Approvals, Workflow, Knowledge Foundation |
| Agents | 8-9 | 74 | Registry, Discovery, RUN Workflow, Knowledge Injection |
| Core Agents | 10-11 | 84 | Scoping, Designer, Delivery, Roles, Templates |
| Production | 12-13 | 63 | Analytics, Security, GDPR, Launch |
| **TOTAL** | **13** | **379** | **MVP Launch** |

---

## Velocity & Capacity

| Metric | Value |
|--------|-------|
| Average Sprint Capacity | 30 points |
| Total Story Points | 379 |
| Buffer (10%) | 38 points |
| Effective Capacity | 341 points |

### Team Velocity Targets (Rebalanced)

| Sprint | Target Points | Cumulative | New Items Added |
|--------|---------------|------------|-----------------|
| 1 | 14 | 14 | - |
| 2 | 25 | 39 | - |
| 3 | 21 | 60 | - |
| 4 | 21 | 81 | - |
| 5 | 18 | 99 | - |
| 6 | 28 | 127 | +5 (Knowledge Base) |
| 7 | 31 | 158 | +10 (Pattern, Template Mgmt) |
| 8 | 34 | 192 | +13 (RUN Workflow) |
| 9 | 40 | 232 | +16 (Knowledge Injection, RUN) |
| 10 | 41 | 273 | +15 (Roles, Doc Template) |
| 11 | 43 | 316 | +17 (Doc Templates) |
| 12 | 40 | 356 | +16 (GDPR) |
| 13 | 23 | 379 | - |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Auth complexity | Medium | High | Use battle-tested library (Flask-JWT-Extended) |
| Harbyx learning curve | Low | Medium | Start SDK setup in Sprint 4 |
| Agent complexity | High | High | Focus on 4 core agents, defer 3 |
| Jira sync issues | Medium | Medium | Start with basic sync, iterate |
| Performance issues | Low | Medium | Load test in Sprint 12 |

---

## What's Included vs. Deferred

### Included in 6-Month MVP
- ✅ Full authentication (email + Google)
- ✅ Multi-tenancy with data isolation
- ✅ Harbyx governance (Type A/B/C)
- ✅ Approval inbox and audit logs
- ✅ Workflow engine (BUILD + RUN phases)
- ✅ Human Role Management (5 roles: BO, PO, Tech Lead, Release Mgr, Support)
- ✅ Knowledge Layer (Client context, patterns, templates)
- ✅ Backlog management (Kanban)
- ✅ 4 Core Agents: Discovery, Scoping, Designer, Delivery
- ✅ Documentation Templates (8 standard templates)
- ✅ Jira integration (bi-directional)
- ✅ GDPR Compliance (localization, erasure, export, consent)
- ✅ Metrics dashboard
- ✅ Security hardening
- ✅ Kubernetes deployment

### Deferred to Post-Launch
- ⏳ Challenger Agent
- ⏳ QA Agent
- ⏳ RunOps Agent
- ⏳ GitHub integration
- ⏳ Teams integration
- ⏳ ServiceNow integration
- ⏳ CI/CD integration (Copado/Gearset)
- ⏳ Stripe billing (manual invoicing initially)
- ⏳ Microsoft/Adobe platform adapters
- ⏳ Full documentation site

---

## Key Dates

| Date | Week | Milestone |
|------|------|-----------|
| Week 2 | End Jan | M1: Demo Working |
| Week 4 | Mid Feb | M2: Auth Live |
| Week 10 | End Mar | M3: Governance MVP |
| Week 16 | Mid May | M4: Platform Foundation |
| Week 18 | End May | M5: First Agent |
| Week 22 | End Jun | M6: Core Agents + Jira |
| Week 24 | Mid Jul | M7: Beta Ready |
| Week 26 | End Jul | GA: Production Launch |

---

*Document Version: 1.0*
*Created: January 9, 2026*
