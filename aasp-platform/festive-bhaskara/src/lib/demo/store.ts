// In-memory store for demo state
import type { Action, Policy, ApprovalRequest, DemoState } from './types';
import { defaultPolicies, demoAgents } from '@/lib/data/demo-data';

// Initialize state
let state: DemoState = {
  agents: [...demoAgents],
  policies: [...defaultPolicies],
  actions: [],
  approvals: [],
};

// Event listeners for SSE
type EventListener = (event: { type: string; data: unknown }) => void;
const listeners: Set<EventListener> = new Set();

// Emit events to all listeners
function emit(type: string, data: unknown) {
  listeners.forEach((listener) => {
    listener({ type, data });
  });
}

// Store API
export const demoStore = {
  // Get current state
  getState: (): DemoState => ({ ...state }),

  // Reset state to initial
  reset: () => {
    state = {
      agents: [...demoAgents],
      policies: [...defaultPolicies],
      actions: [],
      approvals: [],
    };
    emit('reset', null);
  },

  // Actions
  getActions: (limit = 50, filters?: { agentId?: string; decision?: string }) => {
    let actions = [...state.actions].reverse();

    if (filters?.agentId) {
      actions = actions.filter((a) => a.agentId === filters.agentId);
    }
    if (filters?.decision) {
      actions = actions.filter((a) => a.decision === filters.decision);
    }

    return actions.slice(0, limit);
  },

  addAction: (action: Action) => {
    state.actions.push(action);
    emit('action', action);
    return action;
  },

  getActionById: (id: string) => {
    return state.actions.find((a) => a.id === id);
  },

  // Policies
  getPolicies: () => [...state.policies],

  getPolicyById: (id: string) => {
    return state.policies.find((p) => p.id === id);
  },

  addPolicy: (policy: Policy) => {
    state.policies.push(policy);
    emit('policy', { action: 'created', policy });
    return policy;
  },

  updatePolicy: (id: string, updates: Partial<Policy>) => {
    const index = state.policies.findIndex((p) => p.id === id);
    if (index === -1) return null;

    state.policies[index] = { ...state.policies[index], ...updates };
    emit('policy', { action: 'updated', policy: state.policies[index] });
    return state.policies[index];
  },

  deletePolicy: (id: string) => {
    const index = state.policies.findIndex((p) => p.id === id);
    if (index === -1) return false;

    const deleted = state.policies.splice(index, 1)[0];
    emit('policy', { action: 'deleted', policy: deleted });
    return true;
  },

  togglePolicy: (id: string) => {
    const policy = state.policies.find((p) => p.id === id);
    if (!policy) return null;

    policy.enabled = !policy.enabled;
    emit('policy', { action: 'toggled', policy });
    return policy;
  },

  // Approvals
  getApprovals: (status?: 'pending' | 'approved' | 'rejected') => {
    let approvals = [...state.approvals].reverse();

    if (status) {
      approvals = approvals.filter((a) => a.status === status);
    }

    return approvals;
  },

  addApproval: (approval: ApprovalRequest) => {
    state.approvals.push(approval);
    emit('approval', { action: 'created', approval });
    return approval;
  },

  resolveApproval: (
    id: string,
    decision: 'approved' | 'rejected',
    decidedBy: string,
    reason?: string
  ) => {
    const approval = state.approvals.find((a) => a.id === id);
    if (!approval || approval.status !== 'pending') return null;

    approval.status = decision;
    approval.decidedAt = new Date().toISOString();
    approval.decidedBy = decidedBy;
    approval.reason = reason;

    // Update the associated action
    const action = state.actions.find((a) => a.id === approval.actionId);
    if (action) {
      action.decision = decision === 'approved' ? 'allow' : 'block';
      action.reason = reason || `${decision} by ${decidedBy}`;
    }

    emit('approval', { action: 'resolved', approval });
    return approval;
  },

  // Event subscription
  subscribe: (listener: EventListener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  // Get agent by ID
  getAgent: (id: string) => {
    return state.agents.find((a) => a.id === id);
  },

  // Stats
  getStats: () => {
    const actions = state.actions;
    const last24h = actions.filter(
      (a) => new Date(a.timestamp).getTime() > Date.now() - 24 * 60 * 60 * 1000
    );

    return {
      totalActions: actions.length,
      actionsLast24h: last24h.length,
      allowed: actions.filter((a) => a.decision === 'allow').length,
      blocked: actions.filter((a) => a.decision === 'block').length,
      pendingApprovals: state.approvals.filter((a) => a.status === 'pending').length,
      activePolicies: state.policies.filter((p) => p.enabled).length,
    };
  },
};

export type DemoStore = typeof demoStore;
