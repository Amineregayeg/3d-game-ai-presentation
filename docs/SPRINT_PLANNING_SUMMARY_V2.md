# Sprint Planning Summary V2 - With Harbyx Integration

**Version:** 2.0
**Sprint Duration:** 2 weeks
**Team Size:** 3-4 developers
**Total Sprints:** 18 (was 27)
**Total Duration:** ~36 weeks with buffer (was 54 weeks)
**Key Change:** Harbyx (app.harbyx.com) provides governance layer

---

## Timeline Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    V1 (BUILD EVERYTHING) vs V2 (HARBYX)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  V1: ████████████████████████████████████████████████████  54 weeks        │
│                                                                             │
│  V2: ████████████████████████████████████               36 weeks           │
│                                                                             │
│  SAVINGS: ██████████████████  18 weeks (33% faster)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Sprint-to-Phase Mapping

| Phase | Sprints | Duration | Key Deliverable |
|-------|---------|----------|-----------------|
| **Phase 1: Foundation** | 1-2 | 4 weeks | Portal + Auth + Database |
| **Phase 2: Harbyx + Workflow** | 3-4 | 4 weeks | **Governance via Harbyx API** |
| **Phase 3: Agent Runtime** | 5-11 | 14 weeks | 7 agents with Harbyx governance |
| **Phase 4: Integration Hub** | 12-14 | 6 weeks | Jira, GitHub, Teams |
| **Phase 5: Multi-Platform** | 15-16 | 4 weeks | Microsoft + Adobe |
| **Phase 6: Production** | 17-18 | 4 weeks | Analytics, Billing, Deploy |

---

## Sprint-by-Sprint Plan

### Sprint 1 (Weeks 1-2) - Foundation Start
**Goal:** Authentication and core database
**Capacity:** 18 story points

| Story | Points | Owner |
|-------|--------|-------|
| User Authentication System | 8 | Dev 1 |
| Core Database Schema | 5 | Dev 2 |
| RESTful API Framework | 5 | Dev 3 |

**Exit Criteria:**
- [ ] Users can register and login
- [ ] Database migrations working
- [ ] API documentation available

---

### Sprint 2 (Weeks 3-4) - Foundation Complete
**Goal:** Multi-tenancy and real-time
**Capacity:** 26 story points

| Story | Points | Owner |
|-------|--------|-------|
| Organization & Tenant Management | 8 | Dev 1 |
| Portal Dashboard UI | 8 | Dev 2 |
| Vector Database for RAG | 5 | Dev 3 |
| WebSocket Infrastructure | 5 | Dev 4 |

**Exit Criteria:**
- [ ] Organizations can be created
- [ ] Users see dashboard after login
- [ ] Real-time WebSocket working

**MILESTONE: MVP 0 - Portal Ready**

---

### Sprint 3 (Weeks 5-6) - Harbyx Integration
**Goal:** Integrate Harbyx governance layer
**Capacity:** 16 story points

| Story | Points | Owner |
|-------|--------|-------|
| Harbyx SDK Setup | 3 | Dev 1 |
| Policy Configuration (Type A/B/C) | 5 | Dev 1 |
| Agent Action Decorator | 8 | Dev 2 + Dev 3 |

**Exit Criteria:**
- [ ] Harbyx SDK integrated
- [ ] Type A/B/C policies created
- [ ] `@governed_action` decorator working
- [ ] Test action evaluated by Harbyx

**Key Harbyx Setup:**
```python
# .env
HARBYX_API_KEY=your_api_key
HARBYX_BASE_URL=https://app.harbyx.com/api/v1

# Test connection
from aasp_sdk import AASPClient
client = AASPClient()
assert client.health_check()
```

---

### Sprint 4 (Weeks 7-8) - Governance UI + Workflow
**Goal:** Approval UI and workflow engine
**Capacity:** 34 story points

| Story | Points | Owner |
|-------|--------|-------|
| Approval Inbox UI | 8 | Dev 1 |
| Audit Log Viewer | 5 | Dev 2 |
| Webhook Integration | 5 | Dev 3 |
| Workflow State Machine | 8 | Dev 4 |
| Phase Gate UI | 5 | Dev 2 |
| Backlog Item CRUD | 3 | Dev 3 |

**Exit Criteria:**
- [ ] Approvers can view pending approvals from Harbyx
- [ ] Audit logs visible from Harbyx
- [ ] Real-time webhook events received
- [ ] Projects can move through BUILD/RUN phases
- [ ] Kanban board functional

**MILESTONE: MVP 1 - Human-in-the-Loop Governance**

---

### Sprint 5 (Weeks 9-10) - Agent Orchestration Start
**Goal:** Agent registry and management
**Capacity:** 13 story points

| Story | Points | Owner |
|-------|--------|-------|
| Agent Registry & Lifecycle | 13 | Dev 1 + Dev 2 |

**Exit Criteria:**
- [ ] 7 agent types registered
- [ ] Agent health monitoring
- [ ] All agents configured with `@governed_action`

---

### Sprint 6 (Weeks 11-12) - Agent Orchestrator
**Goal:** Agent coordination system
**Capacity:** 13 story points

| Story | Points | Owner |
|-------|--------|-------|
| Agent Orchestrator | 13 | Dev 1 + Dev 2 |

**Exit Criteria:**
- [ ] Agents invoked by workflow phase
- [ ] Input/output passing works
- [ ] Harbyx approval integration complete

---

### Sprint 7 (Weeks 13-14) - Discovery Agent
**Goal:** Platform scanning and risk detection
**Capacity:** 21 story points

| Story | Points | Owner |
|-------|--------|-------|
| Platform Scanner | 13 | Dev 1 + Dev 2 |
| Risk Detection Engine | 8 | Dev 3 |

**Exit Criteria:**
- [ ] Salesforce org can be scanned
- [ ] As-Is snapshot generated
- [ ] Risks identified (governed by Type A policy)

---

### Sprint 8 (Weeks 15-16) - Scoping Agent
**Goal:** Requirement analysis and backlog generation
**Capacity:** 16 story points

| Story | Points | Owner |
|-------|--------|-------|
| Requirement Analyzer | 8 | Dev 1 |
| Epic/Story Generator | 8 | Dev 2 |

**Exit Criteria:**
- [ ] Requirements parsed from text
- [ ] Epics and Stories generated (governed by Type B policy)
- [ ] Acceptance criteria written

---

### Sprint 9 (Weeks 17-18) - Solution Designer Agent
**Goal:** Design document and ADR generation
**Capacity:** 18 story points

| Story | Points | Owner |
|-------|--------|-------|
| Design Document Generator | 13 | Dev 1 + Dev 2 |
| ADR Generator | 5 | Dev 3 |

**Exit Criteria:**
- [ ] Solution design documents generated (governed by Type B)
- [ ] ADRs created from decisions
- [ ] Diagrams included

---

### Sprint 10 (Weeks 19-20) - Challenger + Delivery Agents
**Goal:** Review and execution agents
**Capacity:** 21 story points

| Story | Points | Owner |
|-------|--------|-------|
| Challenger Agent | 8 | Dev 1 |
| Delivery Agent | 13 | Dev 2 + Dev 3 |

**Exit Criteria:**
- [ ] Designs can be reviewed/challenged
- [ ] Deployments executed via MCP
- [ ] **Type C policy enforced for production deployments**

---

### Sprint 11 (Weeks 21-22) - QA + RunOps Agents
**Goal:** Complete agent suite
**Capacity:** 16 story points

| Story | Points | Owner |
|-------|--------|-------|
| QA Agent | 8 | Dev 1 |
| RunOps Agent | 8 | Dev 2 |

**Exit Criteria:**
- [ ] Test cases generated and executed
- [ ] Incidents can be triaged
- [ ] All 7 agents operational with Harbyx governance

**MILESTONE: MVP 2 - Full Agent Runtime**

---

### Sprint 12 (Weeks 23-24) - Jira Integration
**Goal:** Full Jira sync
**Capacity:** 18 story points

| Story | Points | Owner |
|-------|--------|-------|
| Jira OAuth Connection | 5 | Dev 1 |
| Jira Bi-directional Sync | 13 | Dev 2 + Dev 3 |

**Exit Criteria:**
- [ ] Jira Cloud connected
- [ ] Issues synced both ways
- [ ] Webhooks working

---

### Sprint 13 (Weeks 25-26) - GitHub + Teams
**Goal:** GitHub and Teams integrations
**Capacity:** 21 story points

| Story | Points | Owner |
|-------|--------|-------|
| GitHub App Connection | 5 | Dev 1 |
| PR and Issue Sync | 8 | Dev 2 |
| Teams App | 8 | Dev 3 |

**Exit Criteria:**
- [ ] GitHub App installed
- [ ] PRs linked to backlog items
- [ ] Teams notifications for approvals (relayed from Harbyx)

---

### Sprint 14 (Weeks 27-28) - ServiceNow + CI/CD
**Goal:** ITSM and deployment integrations
**Capacity:** 16 story points

| Story | Points | Owner |
|-------|--------|-------|
| ServiceNow Connector | 8 | Dev 1 |
| Copado/Gearset Integration | 8 | Dev 2 |

**Exit Criteria:**
- [ ] Incidents synced from ServiceNow
- [ ] Deployments triggered via CI/CD (governed by Harbyx)

**MILESTONE: MVP 3 - Integration Hub Complete**

---

### Sprint 15 (Weeks 29-30) - Microsoft Adapter
**Goal:** Power Platform support
**Capacity:** 16 story points

| Story | Points | Owner |
|-------|--------|-------|
| Dataverse Connection | 8 | Dev 1 |
| Power Platform Scanner | 8 | Dev 2 |

**Exit Criteria:**
- [ ] Microsoft Dataverse connected
- [ ] Power Platform environment scanned

---

### Sprint 16 (Weeks 31-32) - Adobe + SDK
**Goal:** Adobe support and extensibility
**Capacity:** 24 story points

| Story | Points | Owner |
|-------|--------|-------|
| AEP Connection | 8 | Dev 1 |
| Journey Orchestration | 8 | Dev 2 |
| Adapter SDK | 8 | Dev 3 |

**Exit Criteria:**
- [ ] Adobe Experience Platform connected
- [ ] Adapter SDK documented

**MILESTONE: MVP 4 - Multi-Platform**

---

### Sprint 17 (Weeks 33-34) - Analytics & Billing
**Goal:** Metrics and revenue
**Capacity:** 21 story points

| Story | Points | Owner |
|-------|--------|-------|
| Delivery Metrics Dashboard | 8 | Dev 1 |
| Agent Metrics from Harbyx | 3 | Dev 2 |
| Report Builder | 5 | Dev 2 |
| Usage Tracking | 5 | Dev 3 |

**Exit Criteria:**
- [ ] Delivery KPIs visible
- [ ] Agent metrics pulled from Harbyx audit logs
- [ ] Usage tracked per organization

---

### Sprint 18 (Weeks 35-36) - Production Deployment
**Goal:** Go-live ready
**Capacity:** 21 story points

| Story | Points | Owner |
|-------|--------|-------|
| Stripe Integration | 8 | Dev 1 |
| Security Audit | 5 | Dev 2 + Security |
| Kubernetes Deployment | 8 | DevOps |

**Exit Criteria:**
- [ ] Billing via Stripe working
- [ ] Security audit passed (leveraging Harbyx SOC 2)
- [ ] Production deployment complete

**MILESTONE: GA - Production Launch**

---

## Key Milestones Summary

| Milestone | Sprint | Week | Deliverable |
|-----------|--------|------|-------------|
| **MVP 0** | 2 | 4 | Portal with auth |
| **MVP 1** | 4 | 8 | **Harbyx governance live** |
| **MVP 2** | 11 | 22 | All 7 agents operational |
| **MVP 3** | 14 | 28 | Jira/GitHub/Teams connected |
| **MVP 4** | 16 | 32 | Multi-platform support |
| **GA** | 18 | 36 | Production launch |

---

## Velocity Assumptions

- **Story Points per Sprint:** 16-21 (with 3-4 devs)
- **Story Point Definition:** 1 point = ~4 hours of work
- **Sprint Length:** 2 weeks
- **Buffer:** 15% for meetings, reviews, bugs

---

## Harbyx Cost During Development

| Phase | Duration | Harbyx Plan | Monthly Cost |
|-------|----------|-------------|--------------|
| Sprint 1-4 | 8 weeks | Free Trial (14 days) then Growth | $899 |
| Sprint 5-18 | 28 weeks | Growth | $899/month |
| **Production** | Ongoing | Fintech Control Plane | $2,500/month |

**Development Phase Total:** ~$6,300 (7 months × $899)
**Production:** $2,500/month or $30,000/year

---

## Comparison: V1 vs V2

| Metric | V1 (Build All) | V2 (Harbyx) | Savings |
|--------|----------------|-------------|---------|
| **Sprints** | 27 | 18 | 9 sprints |
| **Duration** | 54 weeks | 36 weeks | **18 weeks (33%)** |
| **Story Points** | 435 | 367 | 68 points |
| **Dev Cost** | ~$380,000 | ~$300,000 | $80,000 |
| **Harbyx Cost** | $0 | ~$36,000 (Y1) | - |
| **Net Savings** | - | - | **$44,000 + 18 weeks** |

---

## Team Composition

| Role | Count | Responsibility |
|------|-------|----------------|
| **Tech Lead** | 1 | Architecture, Harbyx integration |
| **Backend Developer** | 2 | API, agents, integrations |
| **Frontend Developer** | 1 | Portal UI, dashboards |
| **DevOps/SRE** | 1 (part-time) | Infrastructure, deployment |
| **Product Owner** | 1 | Backlog, stakeholder comm |

**Total:** 4-5 FTE

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Harbyx downtime | Circuit breaker pattern; queue actions |
| Harbyx API changes | Service abstraction layer |
| Harbyx cost increase | Budget contingency; alternative eval |
| Harbyx rate limits | Caching; batch operations |

---

## What Harbyx Handles (No Build Required)

| Component | Harbyx Feature | API |
|-----------|----------------|-----|
| Policy Engine | Real-time evaluation | `/ingest` |
| Approval Workflow | Human routing + escalation | `/approvals` |
| Immutable Audit | Action logging | Automatic |
| Slack Notifications | Webhook integration | `/webhooks` |
| SSO | Google, Okta, SAML, Azure AD | Built-in |
| RBAC | Role-based routing | Built-in |
| Compliance | SOC 2 ready | Built-in |

**Estimated Development Saved:** 87 story points, 10 weeks

---

*Document Version: 2.0*
*Created: January 9, 2026*
*Harbyx Integration: app.harbyx.com*
