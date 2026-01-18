# Consulting & Delivery Operating System - Implementation Roadmap V2

**Version:** 2.0
**Created:** January 9, 2026
**Updated:** January 9, 2026 (Harbyx Integration)
**Status:** BACKLOG READY
**Total Estimated Effort:** 4-6 months (3-4 developers)
**Key Change:** Integrated Harbyx (app.harbyx.com) as governance layer

---

## Executive Summary

### What Changed in V2

| Aspect | V1 (Build Everything) | V2 (Harbyx Integration) |
|--------|----------------------|-------------------------|
| **Governance Layer** | Build from scratch | Use Harbyx API |
| **Approval Engine** | 3 weeks dev | API integration (2 days) |
| **Audit Log** | 2.5 weeks dev | Harbyx immutable logs |
| **Policy Engine** | 2 weeks dev | Harbyx policy API |
| **Notifications** | 1 week dev | Harbyx webhooks + Slack |
| **Total Duration** | 38 weeks | **24 weeks** |
| **Cost Savings** | - | ~$80,000 dev cost |

### Harbyx Capabilities We're Using

From [app.harbyx.com](https://app.harbyx.com):

| Harbyx Feature | Our Use Case |
|----------------|--------------|
| `POST /api/v1/ingest` | Evaluate every agent action |
| `POST /api/v1/policies` | Define Type A/B/C approval rules |
| `GET /api/v1/approvals` | Check approval status |
| `POST /api/v1/approvals/{id}` | Resolve human decisions |
| `POST /api/v1/webhooks` | Real-time Slack notifications |
| Immutable Audit Logs | Compliance & traceability |
| SSO (Google, Okta, SAML) | Enterprise authentication |
| < 150ms latency | Production-ready performance |

---

## Revised Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 CONSULTING DELIVERY OS - ARCHITECTURE V2                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     EXPERIENCE LAYER (Your Build)                      │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ Portal   │  │ Project  │  │ Backlog  │  │ Approval │              │ │
│  │  │ Dashboard│  │ Workflow │  │ Kanban   │  │ Inbox UI │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘              │ │
│  └─────────────────────────────────────────────────┼─────────────────────┘ │
│                                                    │                        │
│  ┌─────────────────────────────────────────────────┼─────────────────────┐ │
│  │                     AGENT RUNTIME (Your Build)  │                      │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │                      │ │
│  │  │Discovery │  │ Scoping  │  │ Solution │      │                      │ │
│  │  │ Agent    │  │  Agent   │  │ Designer │      │                      │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘      │                      │ │
│  │       │             │             │            │                      │ │
│  │  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐      │                      │ │
│  │  │Challenger│  │ Delivery │  │QA Agent  │      │                      │ │
│  │  │  Agent   │  │  Agent   │  │          │      │                      │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘      │                      │ │
│  │       │             │             │            │                      │ │
│  │       └─────────────┴─────────────┴────────────┘                      │ │
│  │                             │                                          │ │
│  │                    Agent Orchestrator                                  │ │
│  └─────────────────────────────┼─────────────────────────────────────────┘ │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ██  HARBYX GOVERNANCE  ██                         │   │
│  │                    app.harbyx.com/api/v1                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ /ingest  │  │/policies │  │/approvals│  │/webhooks │             │   │
│  │  │          │  │          │  │          │  │          │             │   │
│  │  │ ALLOW    │  │ Type A   │  │ Human    │  │ Slack    │             │   │
│  │  │ BLOCK    │  │ Type B   │  │ Decision │  │ Notify   │             │   │
│  │  │ REQUIRE  │  │ Type C   │  │ Routing  │  │ Events   │             │   │
│  │  │ APPROVAL │  │ Rules    │  │          │  │          │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │   │
│  │                                                                      │   │
│  │  + Immutable Audit Logs | SSO | RBAC | < 150ms | SOC 2 Ready        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PLATFORM ADAPTERS (Your Build)                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │Salesforce│  │Microsoft │  │  Adobe   │  │ Generic  │             │   │
│  │  │   MCP    │  │ Dataverse│  │   AEP    │  │ Adapter  │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    INTEGRATION HUB (Your Build)                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │   Jira   │  │  GitHub  │  │ServiceNow│  │  CI/CD   │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Overview (Revised)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REVISED IMPLEMENTATION TIMELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1          PHASE 2          PHASE 3          PHASE 4                │
│  Foundation       Harbyx +         Agents           Integration             │
│  ━━━━━━━━━━       Workflow         ━━━━━━           ━━━━━━━━━━━            │
│  Weeks 1-4        Weeks 5-8        Weeks 9-16       Weeks 17-22            │
│                                                                             │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐            │
│  │ Portal  │      │ Harbyx  │      │ 7 AI    │      │ Jira    │            │
│  │ Auth    │─────▶│ Integr. │─────▶│ Agents  │─────▶│ GitHub  │            │
│  │ DB      │      │ Workflow│      │ Orchestr│      │ Teams   │            │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘            │
│                                                                             │
│                   PHASE 5          PHASE 6                                  │
│                   Platforms        Production                               │
│                   ━━━━━━━━━        ━━━━━━━━━━                               │
│                   Weeks 23-26      Weeks 27-30                              │
│                                                                             │
│                   ┌─────────┐      ┌─────────┐                              │
│                   │Microsoft│      │Security │                              │
│                   │ Adobe   │      │ Scale   │                              │
│                   └─────────┘      └─────────┘                              │
│                                                                             │
│  ████████████████████████████████████████████████████████████████████████  │
│  TOTAL: 30 WEEKS (vs 38 weeks in V1) - 21% FASTER                          │
│  ████████████████████████████████████████████████████████████████████████  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Summary (Revised)

| Phase | Name | Duration | V1 Duration | Savings |
|-------|------|----------|-------------|---------|
| **1** | Core Platform Foundation | 4 weeks | 4 weeks | - |
| **2** | Harbyx Integration + Workflow | 4 weeks | 6 weeks | **2 weeks** |
| **3** | Agent Runtime System | 8 weeks | 8 weeks | - |
| **4** | Integration Hub | 6 weeks | 6 weeks | - |
| **5** | Multi-Platform Adapters | 4 weeks | 6 weeks | **2 weeks** |
| **6** | Production & Enterprise | 4 weeks | 8 weeks | **4 weeks** |
| **TOTAL** | | **30 weeks** | 38 weeks | **8 weeks** |

---

## Phase 1: Core Platform Foundation (Unchanged)

**Duration:** 4 weeks
**Sprints:** 1-2

### Epic 1.1: Multi-Tenant Portal

| Story | Points | Sprint |
|-------|--------|--------|
| User Authentication System | 8 | 1 |
| Organization & Tenant Management | 8 | 2 |
| Portal Dashboard UI | 8 | 2 |

### Epic 1.2: Database Architecture

| Story | Points | Sprint |
|-------|--------|--------|
| Core Database Schema | 5 | 1 |
| Vector Database for RAG | 5 | 2 |

### Epic 1.3: API Gateway

| Story | Points | Sprint |
|-------|--------|--------|
| RESTful API Framework | 5 | 1 |
| WebSocket Infrastructure | 5 | 2 |

**Phase 1 Total: 44 points, 4 weeks**

---

## Phase 2: Harbyx Integration + Workflow Engine (REVISED)

**Duration:** 4 weeks (was 6 weeks)
**Sprints:** 3-4
**Key Change:** Replace custom governance with Harbyx API

### Epic 2.1: Harbyx Integration (NEW)

**Business Value:** Leverage production-ready governance instead of building from scratch

#### Story 2.1.1: Harbyx SDK Setup

**As a** developer
**I want to** integrate the Harbyx SDK
**So that** all agent actions are governed

**Acceptance Criteria:**
- [ ] Install `aasp-sdk` package
- [ ] Configure API key securely (env vars)
- [ ] Create Harbyx client wrapper service
- [ ] Test connection to Harbyx API
- [ ] Handle authentication errors gracefully

**Technical Tasks:**
```
□ T-2.1.1.1: Install aasp-sdk via pip
□ T-2.1.1.2: Create HarbyxService class
□ T-2.1.1.3: Implement secure API key storage
□ T-2.1.1.4: Create connection health check
□ T-2.1.1.5: Add retry logic for API calls
□ T-2.1.1.6: Write integration tests
```

**Estimate:** 3 story points (2 days)

---

#### Story 2.1.2: Policy Configuration

**As a** platform administrator
**I want to** define approval policies in Harbyx
**So that** agent actions are evaluated correctly

**Acceptance Criteria:**
- [ ] Create Type A policy (Acknowledgement - auto-approve low-risk)
- [ ] Create Type B policy (Approval required - scope, design)
- [ ] Create Type C policy (Go/No-Go - production deployments)
- [ ] Map policies to agent action types
- [ ] Configure approver roles per policy

**Policy Definitions:**

```python
# Type A - Acknowledgement (Low-risk, auto-log)
TYPE_A_POLICY = {
    "name": "type_a_acknowledgement",
    "description": "Low-risk actions that are logged but auto-approved",
    "target": "agent_action",
    "conditions": [
        {"field": "risk_level", "operator": "eq", "value": "low"},
        {"field": "action_type", "operator": "in", "value": [
            "discovery_scan", "report_generation", "data_read"
        ]}
    ],
    "action": "allow",  # Auto-approve but log
    "metadata": {"approval_type": "type_a"}
}

# Type B - Approval Required (Scope, Design, UAT)
TYPE_B_POLICY = {
    "name": "type_b_approval",
    "description": "Medium-risk actions requiring human approval",
    "target": "agent_action",
    "conditions": [
        {"field": "action_type", "operator": "in", "value": [
            "scope_change", "design_approval", "uat_signoff",
            "field_creation", "flow_modification"
        ]}
    ],
    "action": "require_approval",
    "approvers": ["product_owner", "design_authority", "tech_lead"],
    "timeout_hours": 24,
    "escalation_hours": 4,
    "metadata": {"approval_type": "type_b"}
}

# Type C - Go/No-Go (Production Deployments)
TYPE_C_POLICY = {
    "name": "type_c_go_nogo",
    "description": "High-risk production actions requiring release manager",
    "target": "agent_action",
    "conditions": [
        {"field": "environment", "operator": "eq", "value": "production"},
        {"field": "action_type", "operator": "in", "value": [
            "deployment", "data_migration", "apex_execution",
            "permission_change", "profile_modification"
        ]}
    ],
    "action": "require_approval",
    "approvers": ["release_manager"],
    "timeout_hours": 4,
    "escalation_hours": 1,
    "metadata": {"approval_type": "type_c"}
}
```

**Technical Tasks:**
```
□ T-2.1.2.1: Define Type A policy via Harbyx API
□ T-2.1.2.2: Define Type B policy via Harbyx API
□ T-2.1.2.3: Define Type C policy via Harbyx API
□ T-2.1.2.4: Create policy management UI
□ T-2.1.2.5: Implement policy versioning
□ T-2.1.2.6: Test policy evaluation
```

**Estimate:** 5 story points (3 days)

---

#### Story 2.1.3: Agent Action Decorator

**As a** developer
**I want to** wrap agent actions with Harbyx governance
**So that** every action is evaluated before execution

**Acceptance Criteria:**
- [ ] Create `@governed_action` decorator
- [ ] Automatically call Harbyx `/ingest` endpoint
- [ ] Handle ALLOW → proceed with action
- [ ] Handle BLOCK → raise exception with reason
- [ ] Handle REQUIRE_APPROVAL → wait for decision
- [ ] Pass approval context to agent

**Implementation:**

```python
from functools import wraps
from harbyx_service import HarbyxClient
from typing import Callable, Any

harbyx = HarbyxClient()

def governed_action(
    action_type: str,
    risk_level: str = "medium",
    environment: str = "development"
):
    """
    Decorator to wrap agent actions with Harbyx governance.

    Usage:
        @governed_action(action_type="field_creation", risk_level="medium")
        async def create_custom_field(object_name: str, field_def: dict):
            # Your implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build action context
            action_context = {
                "action_type": action_type,
                "risk_level": risk_level,
                "environment": environment,
                "agent": func.__module__,
                "function": func.__name__,
                "input": {"args": str(args), "kwargs": kwargs},
                "timestamp": datetime.utcnow().isoformat()
            }

            # Call Harbyx for decision
            decision = await harbyx.ingest(action_context)

            if decision.action == "allow":
                # Execute and log result
                result = await func(*args, **kwargs)
                await harbyx.log_result(decision.id, "success", result)
                return result

            elif decision.action == "block":
                raise ActionBlockedError(
                    f"Action blocked by policy: {decision.reason}"
                )

            elif decision.action == "require_approval":
                # Wait for human decision
                approval = await harbyx.wait_for_approval(
                    decision.approval_id,
                    timeout=decision.timeout_hours * 3600
                )

                if approval.status == "approved":
                    result = await func(*args, **kwargs)
                    await harbyx.log_result(decision.id, "success", result)
                    return result
                else:
                    raise ApprovalRejectedError(
                        f"Action rejected: {approval.reason}"
                    )

        return wrapper
    return decorator


# Example usage in Delivery Agent
class DeliveryAgent:

    @governed_action(
        action_type="field_creation",
        risk_level="medium",
        environment="production"
    )
    async def create_custom_field(
        self,
        object_name: str,
        field_name: str,
        field_type: str
    ) -> dict:
        """Create a custom field in Salesforce."""
        return await self.salesforce_mcp.create_field(
            object_name, field_name, field_type
        )
```

**Technical Tasks:**
```
□ T-2.1.3.1: Create governed_action decorator
□ T-2.1.3.2: Implement Harbyx ingest call
□ T-2.1.3.3: Handle ALLOW response
□ T-2.1.3.4: Handle BLOCK response with exception
□ T-2.1.3.5: Implement approval waiting logic
□ T-2.1.3.6: Add timeout handling
□ T-2.1.3.7: Create unit tests
□ T-2.1.3.8: Document decorator usage
```

**Estimate:** 8 story points (1 week)

---

#### Story 2.1.4: Approval Inbox UI

**As an** approver
**I want to** see and act on pending approvals
**So that** I can make timely decisions

**Acceptance Criteria:**
- [ ] Fetch pending approvals from Harbyx API
- [ ] Display approval cards with context
- [ ] Show action details and diff when available
- [ ] Approve/Reject buttons with comment
- [ ] Real-time updates via Harbyx webhooks
- [ ] Filter by type, agent, status

**Technical Tasks:**
```
□ T-2.1.4.1: Create ApprovalInbox component
□ T-2.1.4.2: Implement Harbyx approvals API call
□ T-2.1.4.3: Build approval card component
□ T-2.1.4.4: Add approve/reject actions
□ T-2.1.4.5: Integrate webhook for real-time updates
□ T-2.1.4.6: Add filtering and search
□ T-2.1.4.7: Mobile-responsive design
```

**Estimate:** 8 story points (1 week)

---

#### Story 2.1.5: Audit Log Viewer

**As an** administrator
**I want to** view the audit trail from Harbyx
**So that** I can investigate and demonstrate compliance

**Acceptance Criteria:**
- [ ] Fetch audit logs from Harbyx
- [ ] Display in timeline view
- [ ] Search by date, agent, action type
- [ ] Show full action context on expand
- [ ] Export to CSV/JSON
- [ ] Link to related approvals

**Technical Tasks:**
```
□ T-2.1.5.1: Create AuditLogViewer component
□ T-2.1.5.2: Implement Harbyx audit API call
□ T-2.1.5.3: Build timeline view
□ T-2.1.5.4: Add search and filters
□ T-2.1.5.5: Create detail expansion panel
□ T-2.1.5.6: Implement export functionality
```

**Estimate:** 5 story points (3 days)

---

#### Story 2.1.6: Webhook Integration

**As a** platform
**I want to** receive real-time events from Harbyx
**So that** the UI updates immediately

**Acceptance Criteria:**
- [ ] Configure Harbyx webhook endpoint
- [ ] Handle `action.blocked` events
- [ ] Handle `approval.requested` events
- [ ] Handle `approval.resolved` events
- [ ] Push updates to frontend via WebSocket
- [ ] Verify webhook signatures

**Technical Tasks:**
```
□ T-2.1.6.1: Create webhook endpoint
□ T-2.1.6.2: Implement signature verification
□ T-2.1.6.3: Handle event types
□ T-2.1.6.4: Push to WebSocket clients
□ T-2.1.6.5: Add error handling and retry
```

**Estimate:** 5 story points (3 days)

---

### Epic 2.2: Workflow Engine (Simplified)

**Business Value:** Manage project phases while Harbyx handles approvals

#### Story 2.2.1: Workflow State Machine

**As a** project manager
**I want to** move projects through defined phases
**So that** delivery follows a consistent process

**Acceptance Criteria:**
- [ ] BUILD workflow phases defined
- [ ] RUN workflow phases defined
- [ ] State transitions trigger Harbyx policy check
- [ ] Phase history tracked
- [ ] Workflow visualization

**Note:** Phase transitions that require approval will be governed by Harbyx policies, not custom approval logic.

**Technical Tasks:**
```
□ T-2.2.1.1: Define workflow YAML schema
□ T-2.2.1.2: Implement state machine engine
□ T-2.2.1.3: Create phase transition API
□ T-2.2.1.4: Integrate with Harbyx for approval gates
□ T-2.2.1.5: Track phase history
□ T-2.2.1.6: Write tests
```

**Estimate:** 8 story points (1 week)

---

#### Story 2.2.2: Phase Gate UI

**As a** stakeholder
**I want to** see project progress visually
**So that** I understand the current state

**Acceptance Criteria:**
- [ ] Visual workflow timeline
- [ ] Current phase highlighted
- [ ] Approval status from Harbyx
- [ ] Click to view phase details
- [ ] Transition buttons (governed by Harbyx)

**Technical Tasks:**
```
□ T-2.2.2.1: Create WorkflowTimeline component
□ T-2.2.2.2: Build phase card component
□ T-2.2.2.3: Integrate Harbyx approval status
□ T-2.2.2.4: Add transition actions
□ T-2.2.2.5: Implement animations
```

**Estimate:** 5 story points (3 days)

---

### Epic 2.3: Backlog Management (Unchanged)

| Story | Points | Sprint |
|-------|--------|--------|
| Backlog Item CRUD | 8 | 4 |
| Backlog Board (Kanban) | 8 | 4 |

---

### Phase 2 Summary (Revised)

| Epic | Stories | Total Points | Duration |
|------|---------|--------------|----------|
| 2.1 Harbyx Integration | 6 | 34 | 2.5 weeks |
| 2.2 Workflow Engine | 2 | 13 | 1 week |
| 2.3 Backlog Management | 2 | 16 | 1 week |
| **Phase 2 Total** | **10** | **63** | **4 weeks** |

**Savings vs V1:** 24 story points, 2 weeks

---

## Phase 3: Agent Runtime System (Unchanged)

**Duration:** 8 weeks
**Sprints:** 5-8

All agents now use the `@governed_action` decorator from Phase 2.

### Epic 3.1: Agent Orchestration

| Story | Points | Sprint |
|-------|--------|--------|
| Agent Registry & Lifecycle | 13 | 5 |
| Agent Orchestrator | 13 | 6 |

### Epic 3.2: Discovery Agent

| Story | Points | Sprint |
|-------|--------|--------|
| Platform Scanner | 13 | 7 |
| Risk Detection Engine | 8 | 7 |

### Epic 3.3: Scoping Agent

| Story | Points | Sprint |
|-------|--------|--------|
| Requirement Analyzer | 8 | 8 |
| Epic/Story Generator | 8 | 8 |

### Epic 3.4: Solution Designer Agent

| Story | Points | Sprint |
|-------|--------|--------|
| Design Document Generator | 13 | 9 |
| ADR Generator | 5 | 9 |

### Epic 3.5: Remaining Agents

| Story | Points | Sprint |
|-------|--------|--------|
| Challenger Agent | 8 | 10 |
| Delivery Agent | 13 | 10 |
| QA Agent | 8 | 11 |
| RunOps Agent | 8 | 11 |

**Phase 3 Total: 118 points, 8 weeks**

---

## Phase 4: Integration Hub (Unchanged)

**Duration:** 6 weeks
**Sprints:** 12-14

### Integrations

| Epic | Stories | Points | Sprint |
|------|---------|--------|--------|
| 4.1 Jira | 2 | 18 | 12 |
| 4.2 GitHub | 2 | 13 | 13 |
| 4.3 Teams | 1 | 8 | 13 |
| 4.4 ServiceNow | 1 | 8 | 14 |
| 4.5 CI/CD | 1 | 8 | 14 |

**Note:** Slack is already handled by Harbyx webhooks (no additional work needed)

**Phase 4 Total: 55 points, 6 weeks**

---

## Phase 5: Multi-Platform Adapters (Reduced)

**Duration:** 4 weeks (was 6 weeks)
**Sprints:** 15-16

### Epic 5.1: Microsoft Adapter

| Story | Points | Sprint |
|-------|--------|--------|
| Dataverse Connection | 8 | 15 |
| Power Platform Scanner | 8 | 15 |

### Epic 5.2: Adobe Adapter

| Story | Points | Sprint |
|-------|--------|--------|
| AEP Connection | 8 | 16 |
| Journey Orchestration | 8 | 16 |

### Epic 5.3: Generic Adapter SDK

| Story | Points | Sprint |
|-------|--------|--------|
| Adapter SDK | 8 | 16 |

**Phase 5 Total: 40 points, 4 weeks**

---

## Phase 6: Production & Enterprise (Consolidated)

**Duration:** 4 weeks (was 8 weeks)
**Sprints:** 17-18

### Epic 6.1: Analytics & Reporting

| Story | Points | Sprint |
|-------|--------|--------|
| Delivery Metrics Dashboard | 8 | 17 |
| Agent Metrics (from Harbyx) | 3 | 17 |
| Report Builder | 5 | 17 |

**Note:** Agent metrics can be pulled from Harbyx audit logs, reducing custom work.

### Epic 6.2: Billing

| Story | Points | Sprint |
|-------|--------|--------|
| Usage Tracking | 5 | 18 |
| Stripe Integration | 8 | 18 |

### Epic 6.3: Security & Deployment

| Story | Points | Sprint |
|-------|--------|--------|
| Security Audit | 5 | 18 |
| Kubernetes Deployment | 8 | 18 |
| CI/CD Pipeline | 5 | 18 |

**Note:** SOC 2 compliance is partially covered by Harbyx (they're SOC 2 ready).

**Phase 6 Total: 47 points, 4 weeks**

---

## Total Project Summary (Revised)

| Phase | Duration | Story Points | Stories |
|-------|----------|--------------|---------|
| 1. Foundation | 4 weeks | 44 | 7 |
| 2. Harbyx + Workflow | 4 weeks | 63 | 10 |
| 3. Agent Runtime | 8 weeks | 118 | 12 |
| 4. Integration Hub | 6 weeks | 55 | 7 |
| 5. Multi-Platform | 4 weeks | 40 | 5 |
| 6. Production | 4 weeks | 47 | 8 |
| **TOTAL** | **30 weeks** | **367** | **49** |

### Comparison with V1

| Metric | V1 (Build All) | V2 (Harbyx) | Savings |
|--------|----------------|-------------|---------|
| Duration | 38 weeks | 30 weeks | **8 weeks (21%)** |
| Story Points | 435 | 367 | **68 points (16%)** |
| Stories | 53 | 49 | 4 stories |
| Dev Cost (est.) | $380,000 | $300,000 | **$80,000** |
| Harbyx Cost | $0 | ~$30,000/yr | - |
| **Net Savings** | - | - | **~$50,000 + 8 weeks** |

---

## Harbyx Cost Analysis

| Plan | Monthly | Annual | Best For |
|------|---------|--------|----------|
| Growth | $899 | $10,788 | MVP/pilot phase |
| Fintech Control Plane | $2,500 | $30,000 | Production |

**Recommendation:** Start with Growth plan ($899/mo) during development, upgrade to Fintech Control Plane at production launch.

**ROI Calculation:**
- Dev time saved: 8 weeks × 3 devs × $75/hr × 40 hrs = **$72,000**
- Harbyx Year 1 cost: **$30,000**
- **Net savings: $42,000 + faster time-to-market**

---

## Key Milestones (Revised)

| Milestone | Week | Sprint | Deliverable |
|-----------|------|--------|-------------|
| **MVP 0** | 4 | 2 | Portal with auth |
| **MVP 1** | 8 | 4 | **Harbyx governance + workflow** |
| **MVP 2** | 16 | 8 | 4 core agents operational |
| **MVP 3** | 22 | 11 | All 7 agents + Jira |
| **Beta** | 26 | 13 | Multi-platform support |
| **GA** | 30 | 15 | Production launch |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Harbyx downtime | Implement circuit breaker; queue actions |
| Harbyx API changes | Abstract behind service layer |
| Harbyx pricing changes | Budget buffer; evaluate alternatives |
| Vendor lock-in | Service abstraction allows swap |

---

## Appendix: Harbyx API Reference

### Authentication

```bash
# All requests require API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://app.harbyx.com/api/v1/...
```

### Key Endpoints We Use

```yaml
# Evaluate action
POST /api/v1/ingest
Request:
  action_type: string
  context: object
  agent: string
Response:
  decision: "allow" | "block" | "require_approval"
  approval_id?: string
  reason?: string

# Create policy
POST /api/v1/policies
Request:
  name: string
  target: string
  conditions: array
  action: string
  approvers?: array

# Get pending approvals
GET /api/v1/approvals?status=pending

# Resolve approval
POST /api/v1/approvals/{id}
Request:
  decision: "approved" | "rejected"
  comment?: string

# Configure webhook
POST /api/v1/webhooks
Request:
  url: string
  events: array
  secret: string
```

---

*Document Version: 2.0*
*Created: January 9, 2026*
*Harbyx Integration: app.harbyx.com*
