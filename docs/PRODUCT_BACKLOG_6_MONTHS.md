# Consulting Delivery OS - 6-Month Product Backlog

**Document Version:** 1.0
**Created:** January 9, 2026
**Target:** Product Readiness in 6 Months (26 weeks)
**Team Size:** 3-4 developers

---

## Executive Summary

This backlog tracks the development of the Consulting Delivery OS from current state to production-ready SaaS. Based on codebase analysis, approximately **35% of the Salesforce Virtual Assistant demo is functional**, while **0% of the full Consulting Delivery OS** (governance, agents, integrations) exists.

### Timeline Overview

```
Month 1-2: Foundation & Integration Fixes
Month 3-4: Governance Layer (Harbyx) + First Agents
Month 5:   Agent Suite Completion + Integrations
Month 6:   Production Hardening + Launch
```

---

## Status Legend

| Status | Icon | Description |
|--------|------|-------------|
| **DONE** | ✅ | Fully implemented and functional |
| **IN_PROGRESS** | 🔄 | Partially implemented, needs completion |
| **BLOCKED** | 🚫 | Cannot proceed until dependency resolved |
| **TODO** | ⬜ | Not started |

---

## PHASE 0: Current State (DONE)

### Epic 0.1: Salesforce Virtual Assistant Demo ✅

| ID | Story | Status | Points | Notes |
|----|-------|--------|--------|-------|
| S-001 | Salesforce MCP API | ✅ DONE | 8 | 11 endpoints: query, describe, CRUD, limits |
| S-002 | Salesforce Service (OAuth) | ✅ DONE | 5 | Client credentials flow working |
| S-003 | 7-Layer RAG Pipeline | ✅ DONE | 13 | Query analysis → generation → validation |
| S-004 | Consultant Avatar UI | ✅ DONE | 5 | ConsultantAvatar.tsx with status indicators |
| S-005 | Conversation Panel | ✅ DONE | 5 | Message history with speaker ID |
| S-006 | RAG Context Visualization | ✅ DONE | 5 | Real-time pipeline stage display |
| S-007 | MCP Operations Panel | ✅ DONE | 3 | Operations history and execution |
| S-008 | Settings Panel | ✅ DONE | 3 | Voice, connection, debug settings |
| S-009 | Greeting Modal | ✅ DONE | 2 | Onboarding persona introduction |
| S-010 | Salesforce Demo Page | ✅ DONE | 8 | Full page integration of all components |

**Subtotal Phase 0:** 57 points ✅

### Epic 0.2: Platform Infrastructure ✅

| ID | Story | Status | Points | Notes |
|----|-------|--------|--------|-------|
| S-011 | Flask App Framework | ✅ DONE | 5 | SQLAlchemy ORM, CORS, routing |
| S-012 | Database Models | ✅ DONE | 5 | Task, Team, Activity, Milestone, Decision, Resource, Changelog |
| S-013 | Next.js API Proxy Routes | ✅ DONE | 5 | 11 proxy routes for Salesforce APIs |
| S-014 | ElevenLabs Hook | ✅ DONE | 5 | useElevenLabsConversation WebSocket integration |
| S-015 | Avatar Library | ✅ DONE | 3 | 6 avatars (3 EN, 3 FR) with metadata |

**Subtotal Phase 0.2:** 23 points ✅

---

## PHASE 1: Foundation Fixes (Weeks 1-4)

### Epic 1.1: Critical Integration Fixes ✅

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| F-001 | Register Flask Blueprints | ✅ DONE | 1 | Dev 1 | 1 |
| | **AC:** Add `salesforce_rag_bp` and `salesforce_mcp_bp` to app.py | | | | |
| F-002 | Move Salesforce Credentials to .env | ✅ DONE | 2 | Dev 1 | 1 |
| | **AC:** SF_CLIENT_ID, SF_CLIENT_SECRET, SF_LOGIN_URL in environment | | | | |
| F-003 | Run Data Ingestion Pipeline | ✅ DONE | 3 | Dev 2 | 1 |
| | **AC:** FAISS index created at salesforce_data/salesforce.index (53MB) | | | | |
| F-004 | Index Salesforce Documentation | ✅ DONE | 5 | Dev 2 | 1 |
| | **AC:** 1000+ documents indexed with embeddings (170MB total indexed data) | | | | |
| F-005 | End-to-End Demo Testing | ✅ DONE | 3 | Dev 3 | 1 |
| | **AC:** Voice + text queries return real RAG responses | | | | |

**Sprint 1 Total:** 14 points (14/14 DONE) ✅

### Epic 1.2: Authentication System ✅

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| F-006 | User Database Model | ✅ DONE | 3 | Dev 1 | 2 |
| | **AC:** User table with id, email, password_hash, org_id, role, created_at | | | | |
| F-007 | Password Hashing (bcrypt) | ✅ DONE | 2 | Dev 1 | 2 |
| | **AC:** Passwords stored with bcrypt, verified on login | | | | |
| F-008 | JWT Token Generation | ✅ DONE | 3 | Dev 1 | 2 |
| | **AC:** Login returns JWT with 24h expiry, refresh token flow | | | | |
| F-009 | Auth Middleware | ✅ DONE | 3 | Dev 2 | 2 |
| | **AC:** @token_required decorator protects API routes | | | | |
| F-010 | Login API Endpoint | ✅ DONE | 3 | Dev 2 | 2 |
| | **AC:** POST /api/auth/login returns tokens | | | | |
| F-011 | Register API Endpoint | ✅ DONE | 3 | Dev 2 | 2 |
| | **AC:** POST /api/auth/register creates user | | | | |
| F-012 | Connect Login UI to Backend | ✅ DONE | 3 | Dev 3 | 2 |
| | **AC:** Login page calls real API, stores token, redirects | | | | |
| F-013 | Google OAuth Integration | ⬜ TODO | 5 | Dev 3 | 2 |
| | **AC:** "Continue with Google" completes OAuth flow | | | | |

**Sprint 2 Total:** 25 points (20/25 DONE)

### Epic 1.3: Multi-Tenancy Foundation ✅

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| F-014 | Organization Database Model | ✅ DONE | 5 | Dev 1 | 3 |
| | **AC:** Organization table with id, name, domain, plan, settings | | | | |
| F-015 | User-Organization Relationship | ✅ DONE | 3 | Dev 1 | 3 |
| | **AC:** Users belong to organizations, roles per org | | | | |
| F-016 | Organization API Endpoints | ✅ DONE | 5 | Dev 2 | 3 |
| | **AC:** CRUD for organizations, invite users | | | | |
| F-017 | Data Isolation Middleware | ✅ DONE | 5 | Dev 2 | 3 |
| | **AC:** All queries filtered by org_id | | | | |
| F-018 | Organization Selector UI | ✅ DONE | 3 | Dev 3 | 3 |
| | **AC:** Users can switch between orgs in header | | | | |

**Sprint 3 Total:** 21 points (21/21 DONE)

### Epic 1.4: Dashboard Enhancement 🔄

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| F-019 | Dashboard Home Completion | ✅ DONE | 5 | Dev 3 | 4 |
| | **AC:** Real stats from API, recent sessions, avatar access | | | | |
| F-020 | Session History | ✅ DONE | 5 | Dev 3 | 4 |
| | **AC:** List past conversations with timestamps, transcripts | | | | |
| F-021 | Usage Statistics API | ✅ DONE | 3 | Dev 2 | 4 |
| | **AC:** Track queries, tokens, sessions per org | | | | |
| F-022 | Settings Page | ⬜ TODO | 5 | Dev 3 | 4 |
| | **AC:** User profile, notification preferences, API keys | | | | |

**Sprint 4 Total:** 18 points (13/18 DONE)

**PHASE 1 TOTAL:** 78 points (4 sprints)

---

## PHASE 2: Governance Layer (Weeks 5-10)

### Epic 2.1: Harbyx Integration

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-001 | Harbyx SDK Setup | ⬜ TODO | 3 | Dev 1 | 5 |
| | **AC:** aasp-sdk installed, client wrapper created, health check passes | | | | |
| G-002 | Policy Configuration (Type A/B/C) | ⬜ TODO | 5 | Dev 1 | 5 |
| | **AC:** 3 policy types created in Harbyx dashboard | | | | |
| G-003 | @governed_action Decorator | ⬜ TODO | 8 | Dev 2 | 5 |
| | **AC:** Decorator wraps functions, calls Harbyx /ingest | | | | |
| G-004 | Handle ALLOW/BLOCK/REQUIRE_APPROVAL | ⬜ TODO | 5 | Dev 2 | 5 |
| | **AC:** Actions proceed, block, or wait based on Harbyx decision | | | | |

**Sprint 5 Total:** 21 points

### Epic 2.2: Approval Workflow UI

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-005 | Approval Inbox Page | ⬜ TODO | 8 | Dev 3 | 6 |
| | **AC:** List pending approvals from Harbyx API | | | | |
| G-006 | Approval Card Component | ⬜ TODO | 5 | Dev 3 | 6 |
| | **AC:** Show action context, diff view, approve/reject buttons | | | | |
| G-007 | Harbyx Webhook Handler | ⬜ TODO | 5 | Dev 1 | 6 |
| | **AC:** Receive and process Harbyx events | | | | |
| G-008 | Real-time Approval Updates | ⬜ TODO | 5 | Dev 2 | 6 |
| | **AC:** WebSocket pushes approval status changes to UI | | | | |

**Sprint 6 Total:** 23 points

### Epic 2.3: Audit & Compliance

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-009 | Audit Log Viewer | ⬜ TODO | 5 | Dev 3 | 7 |
| | **AC:** Fetch and display audit logs from Harbyx | | | | |
| G-010 | Audit Search & Filters | ⬜ TODO | 3 | Dev 3 | 7 |
| | **AC:** Filter by date, agent, action type, status | | | | |
| G-011 | Audit Export (CSV/JSON) | ⬜ TODO | 3 | Dev 3 | 7 |
| | **AC:** Download filtered audit logs | | | | |
| G-012 | Action Receipt Display | ⬜ TODO | 3 | Dev 2 | 7 |
| | **AC:** Show cryptographic receipt for each action | | | | |

**Sprint 7 Total:** 14 points

### Epic 2.4: Workflow Engine

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-013 | Workflow State Machine | ⬜ TODO | 8 | Dev 1 | 8 |
| | **AC:** BUILD workflow: Discovery → Scoping → Design → Review → Delivery | | | | |
| G-014 | Phase Gate UI | ⬜ TODO | 5 | Dev 3 | 8 |
| | **AC:** Visual workflow timeline with current phase highlighted | | | | |
| G-015 | Phase Transition API | ⬜ TODO | 5 | Dev 1 | 8 |
| | **AC:** POST /api/workflow/transition triggers Harbyx check | | | | |
| G-016 | Workflow History | ⬜ TODO | 3 | Dev 2 | 8 |
| | **AC:** Track all phase transitions with timestamps | | | | |

**Sprint 8 Total:** 21 points

### Epic 2.5: Backlog Management

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-017 | Backlog Item Model | ⬜ TODO | 3 | Dev 1 | 9 |
| | **AC:** Epic, Story, Task, Bug, Risk, ADR types with relationships | | | | |
| G-018 | Backlog CRUD API | ⬜ TODO | 5 | Dev 1 | 9 |
| | **AC:** Create, read, update, delete backlog items | | | | |
| G-019 | Kanban Board UI | ⬜ TODO | 8 | Dev 3 | 9 |
| | **AC:** Drag-and-drop columns, swimlanes, filters | | | | |
| G-020 | Backlog Item Detail View | ⬜ TODO | 5 | Dev 3 | 9 |
| | **AC:** Full item view with comments, history, attachments | | | | |

**Sprint 9 Total:** 21 points

### Epic 2.6: RUN Workflow Engine

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-021 | RUN Workflow State Machine | ⬜ TODO | 8 | Dev 1 | 8 |
| | **AC:** Intake → Triage → Execute → Validate → Close workflow | | | | |
| G-022 | RUN Triage UI | ⬜ TODO | 5 | Dev 3 | 8 |
| | **AC:** List pending items, assign severity, route to human/agent | | | | |
| G-023 | Low-Risk Auto-Execute | ⬜ TODO | 5 | Dev 2 | 9 |
| | **AC:** Pre-approved actions execute without human approval | | | | |
| G-024 | RUN Incident Reports | ⬜ TODO | 3 | Dev 3 | 9 |
| | **AC:** Generate incident/change reports automatically | | | | |

**Sprint 8 Total (RUN):** 13 points
**Sprint 9 Total (RUN):** 8 points

### Epic 2.7: Human Role Management

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| G-025 | Human Role Models | ⬜ TODO | 5 | Dev 1 | 9 |
| | **AC:** BusinessOwner, ProductOwner, TechLead, ReleaseMgr, SupportLead roles | | | | |
| G-026 | Role Assignment per Project | ⬜ TODO | 3 | Dev 1 | 9 |
| | **AC:** Assign users to roles per project/engagement | | | | |
| G-027 | Role-Based Approval Routing | ⬜ TODO | 5 | Dev 2 | 10 |
| | **AC:** Approvals route to correct role based on phase | | | | |

**Sprint 9 Total (Roles):** 8 points
**Sprint 10 Total (Roles):** 5 points

**PHASE 2 TOTAL:** 134 points (6 sprints)

---

## PHASE 3: Agent Runtime (Weeks 11-18)

### Epic 3.1: Agent Orchestration

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-001 | Agent Registry | ⬜ TODO | 5 | Dev 1 | 10 |
| | **AC:** 7 agent types registered with metadata | | | | |
| A-002 | Agent Configuration per Org | ⬜ TODO | 3 | Dev 1 | 10 |
| | **AC:** Enable/disable agents, set parameters per organization | | | | |
| A-003 | Agent Health Monitoring | ⬜ TODO | 3 | Dev 2 | 10 |
| | **AC:** Heartbeat, status, error tracking per agent | | | | |
| A-004 | Agent Orchestrator | ⬜ TODO | 8 | Dev 2 | 10 |
| | **AC:** Invoke agents by phase, pass input/output, handle errors | | | | |

**Sprint 10 Total:** 19 points

### Epic 3.2: Discovery Agent

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-005 | Platform Scanner | ⬜ TODO | 13 | Dev 1+2 | 11 |
| | **AC:** Scan Salesforce org, extract metadata, generate As-Is snapshot | | | | |
| A-006 | Risk Detection Engine | ⬜ TODO | 8 | Dev 3 | 11 |
| | **AC:** Identify security, performance, compliance, tech debt risks | | | | |
| A-007 | Discovery Report Generator | ⬜ TODO | 5 | Dev 3 | 11 |
| | **AC:** Generate PDF/HTML report with findings | | | | |

**Sprint 11 Total:** 26 points

### Epic 3.3: Scoping Agent

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-008 | Requirement Analyzer | ⬜ TODO | 8 | Dev 1 | 12 |
| | **AC:** Parse natural language requirements, extract entities | | | | |
| A-009 | Epic/Story Generator | ⬜ TODO | 8 | Dev 2 | 12 |
| | **AC:** Auto-generate epics, stories, acceptance criteria | | | | |
| A-010 | Estimation Engine | ⬜ TODO | 5 | Dev 3 | 12 |
| | **AC:** Suggest story points based on complexity analysis | | | | |

**Sprint 12 Total:** 21 points

### Epic 3.4: Solution Designer Agent

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-011 | Design Document Generator | ⬜ TODO | 13 | Dev 1+2 | 13 |
| | **AC:** Generate solution design with data model, integrations, config | | | | |
| A-012 | ADR Generator | ⬜ TODO | 5 | Dev 3 | 13 |
| | **AC:** Create Architecture Decision Records from decisions | | | | |
| A-013 | Diagram Generator (Mermaid) | ⬜ TODO | 5 | Dev 3 | 13 |
| | **AC:** Auto-generate ERD, sequence, architecture diagrams | | | | |

**Sprint 13 Total:** 23 points

### Epic 3.5: Challenger Agent

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-014 | Design Review Checklist | ⬜ TODO | 5 | Dev 1 | 14 |
| | **AC:** Automated checklist evaluation for designs | | | | |
| A-015 | Risk Identification | ⬜ TODO | 5 | Dev 1 | 14 |
| | **AC:** Flag potential issues in proposed solutions | | | | |
| A-016 | Alternative Suggestions | ⬜ TODO | 5 | Dev 2 | 14 |
| | **AC:** Propose alternative approaches when risks found | | | | |

**Sprint 14 Total:** 15 points

### Epic 3.6: Delivery Agent

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-017 | Enhanced MCP Integration | ⬜ TODO | 8 | Dev 1 | 15 |
| | **AC:** Execute complex multi-step Salesforce operations | | | | |
| A-018 | Deployment Orchestration | ⬜ TODO | 8 | Dev 2 | 15 |
| | **AC:** Coordinate deployment steps with validation | | | | |
| A-019 | Rollback Capability | ⬜ TODO | 5 | Dev 2 | 15 |
| | **AC:** Automatic rollback on deployment failure | | | | |
| A-020 | Type C Governance Integration | ⬜ TODO | 3 | Dev 1 | 15 |
| | **AC:** Production deployments require Type C approval | | | | |

**Sprint 15 Total:** 24 points

### Epic 3.7: QA & RunOps Agents

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| A-021 | Test Case Generator | ⬜ TODO | 5 | Dev 1 | 16 |
| | **AC:** Generate test cases from requirements | | | | |
| A-022 | Test Executor | ⬜ TODO | 5 | Dev 1 | 16 |
| | **AC:** Run tests against Salesforce org | | | | |
| A-023 | Incident Triage | ⬜ TODO | 5 | Dev 2 | 16 |
| | **AC:** Analyze incidents, suggest root cause | | | | |
| A-024 | Runbook Executor | ⬜ TODO | 5 | Dev 2 | 16 |
| | **AC:** Execute predefined operational runbooks | | | | |

**Sprint 16 Total:** 20 points

### Epic 3.8: Knowledge Layer

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| K-001 | Knowledge Base Model | ⬜ TODO | 5 | Dev 1 | 6 |
| | **AC:** Client context, standards, constraints stored per org | | | | |
| K-002 | Pattern Library | ⬜ TODO | 5 | Dev 2 | 7 |
| | **AC:** Allowed/forbidden patterns for platform best practices | | | | |
| K-003 | Template Management | ⬜ TODO | 5 | Dev 3 | 7 |
| | **AC:** Document templates (Design, ADR, Runbook, etc.) | | | | |
| K-004 | Agent Knowledge Injection | ⬜ TODO | 8 | Dev 1 | 8 |
| | **AC:** Agents access knowledge layer during execution | | | | |

**Sprint 6 Total (Knowledge):** 5 points
**Sprint 7 Total (Knowledge):** 10 points
**Sprint 8 Total (Knowledge):** 8 points

**PHASE 3 TOTAL:** 171 points (8 sprints)

---

## PHASE 4: Integrations (Weeks 19-22)

### Epic 4.1: Jira Integration

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| I-001 | Jira OAuth Connection | ⬜ TODO | 5 | Dev 1 | 17 |
| | **AC:** Connect to Jira Cloud via OAuth 2.0 | | | | |
| I-002 | Jira Bi-directional Sync | ⬜ TODO | 13 | Dev 1+2 | 17 |
| | **AC:** Sync Epic, Story, Task, Bug both ways | | | | |

**Sprint 17 Total:** 18 points

### Epic 4.2: GitHub Integration

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| I-003 | GitHub App Connection | ⬜ TODO | 5 | Dev 1 | 18 |
| | **AC:** Install GitHub App, access repositories | | | | |
| I-004 | PR and Issue Sync | ⬜ TODO | 8 | Dev 2 | 18 |
| | **AC:** Link PRs to backlog items, update on merge | | | | |

**Sprint 18 Total:** 13 points

### Epic 4.3: Teams/Slack Integration

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| I-005 | Slack App | ⬜ TODO | 5 | Dev 3 | 19 |
| | **AC:** Approval notifications, slash commands | | | | |
| I-006 | Teams App | ⬜ TODO | 5 | Dev 3 | 19 |
| | **AC:** Approval cards, channel notifications | | | | |
| I-007 | ServiceNow Connector | ⬜ TODO | 8 | Dev 1 | 19 |
| | **AC:** Sync incidents and change requests | | | | |

**Sprint 19 Total:** 18 points

### Epic 4.4: CI/CD Integration

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| I-008 | Copado/Gearset Integration | ⬜ TODO | 8 | Dev 2 | 20 |
| | **AC:** Trigger deployments, receive status | | | | |
| I-009 | Deployment Pipeline Governance | ⬜ TODO | 5 | Dev 2 | 20 |
| | **AC:** All production deployments governed by Harbyx | | | | |

**Sprint 20 Total:** 13 points

**PHASE 4 TOTAL:** 62 points (4 sprints)

---

## PHASE 5: Production Readiness (Weeks 23-26)

### Epic 5.1: Analytics & Billing

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| P-001 | Delivery Metrics Dashboard | ⬜ TODO | 8 | Dev 1 | 21 |
| | **AC:** Lead time, cycle time, throughput, deployment frequency | | | | |
| P-002 | Agent Metrics from Harbyx | ⬜ TODO | 3 | Dev 2 | 21 |
| | **AC:** Pull agent success rate, approval wait time | | | | |
| P-003 | Usage Tracking | ⬜ TODO | 5 | Dev 2 | 21 |
| | **AC:** Track API calls, tokens, sessions per org | | | | |
| P-004 | Stripe Integration | ⬜ TODO | 8 | Dev 3 | 21 |
| | **AC:** Subscription management, usage-based billing | | | | |

**Sprint 21 Total:** 24 points

### Epic 5.2: Security & Compliance

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| P-005 | Security Audit | ⬜ TODO | 8 | Dev 1 | 22 |
| | **AC:** OWASP Top 10 review, dependency audit | | | | |
| P-006 | Rate Limiting | ⬜ TODO | 3 | Dev 2 | 22 |
| | **AC:** API rate limits per organization | | | | |
| P-007 | Input Validation Hardening | ⬜ TODO | 3 | Dev 2 | 22 |
| | **AC:** All inputs validated, sanitized | | | | |
| P-008 | Secrets Management | ⬜ TODO | 5 | Dev 1 | 22 |
| | **AC:** All secrets in env vars or vault | | | | |

**Sprint 22 Total:** 19 points

### Epic 5.3: Infrastructure & Deployment

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| P-009 | Kubernetes Deployment | ⬜ TODO | 8 | DevOps | 23 |
| | **AC:** Helm charts, ConfigMaps, Secrets, TLS | | | | |
| P-010 | CI/CD Pipeline | ⬜ TODO | 5 | DevOps | 23 |
| | **AC:** GitHub Actions for build, test, deploy | | | | |
| P-011 | Performance Testing | ⬜ TODO | 5 | Dev 2 | 23 |
| | **AC:** Load test with 100 concurrent users | | | | |
| P-012 | Auto-scaling Setup | ⬜ TODO | 5 | DevOps | 23 |
| | **AC:** HPA configured for frontend and backend | | | | |

**Sprint 23 Total:** 23 points

### Epic 5.4: Launch Preparation

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| P-013 | Landing Page | ⬜ TODO | 8 | Dev 3 | 24 |
| | **AC:** Marketing page with features, pricing, CTA | | | | |
| P-014 | Documentation Site | ⬜ TODO | 5 | Dev 3 | 24 |
| | **AC:** User guide, API docs, getting started | | | | |
| P-015 | Onboarding Flow | ⬜ TODO | 5 | Dev 3 | 24 |
| | **AC:** Guided setup for new organizations | | | | |
| P-016 | Beta Testing | ⬜ TODO | 3 | All | 24 |
| | **AC:** 5 beta customers onboarded and providing feedback | | | | |

**Sprint 24 Total:** 21 points

### Epic 5.5: GDPR Compliance

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| P-017 | Data Localization | ⬜ TODO | 5 | DevOps | 22 |
| | **AC:** EU data stored in EU region, configurable per org | | | | |
| P-018 | Right to Erasure | ⬜ TODO | 5 | Dev 2 | 22 |
| | **AC:** User data deletion API with cascade | | | | |
| P-019 | Data Export | ⬜ TODO | 3 | Dev 2 | 23 |
| | **AC:** Export all user data in portable format | | | | |
| P-020 | Consent Management | ⬜ TODO | 3 | Dev 3 | 23 |
| | **AC:** Track and manage user consent records | | | | |

**Sprint 22 Total (GDPR):** 10 points
**Sprint 23 Total (GDPR):** 6 points

**PHASE 5 TOTAL:** 103 points (4 sprints)

---

## PHASE 6: Documentation & Templates (Weeks 27-28)

### Epic 6.1: Documentation Templates

| ID | Story | Status | Points | Owner | Sprint |
|----|-------|--------|--------|-------|--------|
| D-001 | Project Brief Template | ⬜ TODO | 2 | Dev 3 | 10 |
| | **AC:** Standardized project brief document template | | | | |
| D-002 | As-Is Snapshot Template | ⬜ TODO | 3 | Dev 3 | 10 |
| | **AC:** Template for documenting current state analysis | | | | |
| D-003 | Solution Design Template | ⬜ TODO | 3 | Dev 3 | 10 |
| | **AC:** Comprehensive solution design document structure | | | | |
| D-004 | Test Plan Template | ⬜ TODO | 2 | Dev 3 | 10 |
| | **AC:** Test plan document with coverage sections | | | | |
| D-005 | Release Plan Template | ⬜ TODO | 2 | Dev 3 | 11 |
| | **AC:** Release planning and rollout template | | | | |
| D-006 | Go-Live Report Template | ⬜ TODO | 2 | Dev 3 | 11 |
| | **AC:** Post-deployment go-live report template | | | | |
| D-007 | Runbook Template | ⬜ TODO | 2 | Dev 3 | 11 |
| | **AC:** Operational runbook document structure | | | | |
| D-008 | Incident Report Template | ⬜ TODO | 2 | Dev 3 | 11 |
| | **AC:** Incident documentation and post-mortem template | | | | |

**Sprint 10 Total (Templates):** 10 points
**Sprint 11 Total (Templates):** 8 points

**PHASE 6 TOTAL:** 18 points (spread across Sprints 10-11)

---

## Backlog Summary

### By Phase

| Phase | Sprints | Points | Duration | Key Milestone |
|-------|---------|--------|----------|---------------|
| Phase 0 (Done) | - | 80 | - | Salesforce Demo MVP |
| Phase 1 | 1-4 | 78 | 8 weeks | Auth + Multi-Tenant |
| Phase 2 | 5-10 | 134 | 12 weeks | Harbyx Governance + RUN + Roles |
| Phase 3 | 11-16 | 171 | 12 weeks | All 7 Agents + Knowledge Layer |
| Phase 4 | 17-20 | 62 | 8 weeks | Integrations Complete |
| Phase 5 | 21-24 | 103 | 8 weeks | Production Launch + GDPR |
| Phase 6 | 16 | 18 | Overlap | Documentation Templates |
| **TOTAL** | **24** | **646** | **48 weeks** | **GA** |

### Adjusted for 6 Months (26 weeks)

To hit product readiness in 6 months, we need to parallelize and reduce scope:

| Phase | Adjusted Sprints | Adjusted Weeks | Focus |
|-------|------------------|----------------|-------|
| Phase 1 | 1-2 | 4 weeks | Critical fixes + Auth |
| Phase 2 | 3-5 | 6 weeks | Harbyx + BUILD/RUN Workflows + Roles |
| Phase 3 | 6-9 | 8 weeks | 4 Core Agents + Knowledge Layer |
| Phase 4 | 10-11 | 4 weeks | Jira + Slack only |
| Phase 5 | 12-13 | 4 weeks | Production + GDPR Compliance |
| Phase 6 | Overlap | - | Documentation Templates |
| **TOTAL** | **13** | **26 weeks** | **MVP Launch** |

### Reduced Scope for 6 Months

**Included:**
- ✅ Authentication & Multi-Tenancy
- ✅ Harbyx Governance (Type A/B/C)
- ✅ Workflow Engine (BUILD + RUN phases)
- ✅ Human Role Management (5 roles)
- ✅ Knowledge Layer (Client context, patterns, templates)
- ✅ 4 Core Agents (Discovery, Scoping, Solution Designer, Delivery)
- ✅ Documentation Templates (8 templates)
- ✅ Jira Integration
- ✅ Slack Notifications
- ✅ GDPR Compliance (Data localization, erasure, export)
- ✅ Basic Analytics
- ✅ Kubernetes Deployment

**Deferred to Post-Launch:**
- ⏳ Challenger Agent
- ⏳ QA Agent
- ⏳ RunOps Agent
- ⏳ GitHub Integration
- ⏳ Teams Integration
- ⏳ ServiceNow Integration
- ⏳ CI/CD Integration (Copado/Gearset)
- ⏳ Stripe Billing (manual invoicing at launch)
- ⏳ Landing Page (use simple page)

---

## Milestones

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| **M1: Demo Fixed** | Week 2 | End-to-end Salesforce demo with real RAG |
| **M2: Auth Live** | Week 4 | Users can register, login, see dashboard |
| **M3: Governance MVP** | Week 10 | Harbyx integration, approval inbox working |
| **M4: First Agent** | Week 14 | Discovery Agent scanning orgs |
| **M5: Core Agents** | Week 18 | 4 agents operational with governance |
| **M6: Beta** | Week 22 | 5 beta customers using platform |
| **M7: GA** | Week 26 | Production launch |

---

## Team Allocation

| Role | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|------|---------|---------|---------|---------|---------|
| **Dev 1 (Backend)** | Auth, DB | Harbyx SDK, Workflow | Agents | Jira | Security |
| **Dev 2 (Backend)** | Data, APIs | Webhooks, Events | Agents | CI/CD | Infra |
| **Dev 3 (Frontend)** | UI fixes | Approval UI, Kanban | Agent UI | Slack | Landing |
| **DevOps** | - | - | - | - | K8s, CI/CD |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Harbyx API changes | Low | High | Service abstraction layer |
| Agent complexity | Medium | High | Start with simplest agents |
| Salesforce rate limits | Medium | Medium | Caching, batch operations |
| 6-month deadline | Medium | High | Reduce scope, parallel work |
| Team velocity | Medium | Medium | Track weekly, adjust plan |

---

## Next Actions

### Immediate (This Week)
1. [x] Register Flask blueprints in app.py (F-001) ✅
2. [x] Move Salesforce credentials to .env (F-002) ✅
3. [x] Run data ingestion scripts (F-003, F-004) ✅
4. [x] Test end-to-end demo flow (F-005) ✅

### Current Sprint (Sprint 5) Goals
- [ ] Settings Page (F-022) - 5 pts
- [ ] Harbyx SDK Setup (G-001) - 3 pts
- [ ] Policy Configuration (G-002) - 5 pts

### Progress Summary
| Sprint | Status | Points Done |
|--------|--------|-------------|
| Sprint 1 | ✅ | 14/14 |
| Sprint 2 | ✅ | 20/25 |
| Sprint 3 | ✅ | 21/21 |
| Sprint 4 | 🔄 | 13/18 |
| **Total** | | **68/78** (87%) |

---

*Document Version: 1.0*
*Created: January 9, 2026*
*Last Updated: January 9, 2026*
