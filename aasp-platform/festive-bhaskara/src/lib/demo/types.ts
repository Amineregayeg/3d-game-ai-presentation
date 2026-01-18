// Demo Types for AASP Interactive Demo

export type ActionType = 'api_call' | 'db_query' | 'file_access' | 'external';
export type Decision = 'allow' | 'block' | 'require_approval';
export type ApprovalStatus = 'pending' | 'approved' | 'rejected';

export interface DemoAgent {
  id: string;
  name: string;
  icon: string;
  description: string;
  color: string;
}

export interface ActionTemplate {
  type: ActionType;
  label: string;
  icon: string;
  placeholder: string;
  sampleParams: Record<string, unknown>;
}

export interface Condition {
  field: string;
  operator: 'eq' | 'neq' | 'gt' | 'lt' | 'gte' | 'lte' | 'contains' | 'matches';
  value: string | number | boolean;
}

export interface PolicyRule {
  id: string;
  actionType?: ActionType;
  targetPattern?: string;
  conditions?: Condition[];
  effect: Decision;
  approvers?: string[];
}

export interface Policy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  rules: PolicyRule[];
  createdAt: string;
}

export interface ActionRequest {
  agentId: string;
  actionType: ActionType;
  target: string;
  params: Record<string, unknown>;
}

export interface Action {
  id: string;
  agentId: string;
  agentName: string;
  actionType: ActionType;
  target: string;
  params: Record<string, unknown>;
  decision: Decision;
  reason: string;
  policyId?: string;
  policyName?: string;
  timestamp: string;
  duration?: number;
}

export interface ApprovalRequest {
  id: string;
  actionId: string;
  action: Action;
  requestedAt: string;
  status: ApprovalStatus;
  decidedAt?: string;
  decidedBy?: string;
  reason?: string;
}

export interface DemoState {
  agents: DemoAgent[];
  policies: Policy[];
  actions: Action[];
  approvals: ApprovalRequest[];
}

export interface EvaluationResult {
  decision: Decision;
  reason: string;
  matchedPolicy?: Policy;
  matchedRule?: PolicyRule;
}

export interface StreamEvent {
  type: 'action' | 'approval' | 'policy';
  data: Action | ApprovalRequest | Policy;
  timestamp: string;
}
