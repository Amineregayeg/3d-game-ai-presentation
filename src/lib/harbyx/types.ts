// Harbyx Governance SDK Types

export type ActionType =
  | 'tool_call'
  | 'api_call'
  | 'db_query'
  | 'file_access'
  | 'external_request';

export type Decision = 'allow' | 'block' | 'require_approval';

export interface IngestParams {
  agentId: string;
  actionType: ActionType;
  target: string;
  params?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  success: boolean;
  decision: Decision;
  action_id: string;
  reason: string;
  approval_id?: string;
  policy_id?: string;
}

export interface BatchIngestParams {
  actions: IngestParams[];
}

export interface BatchIngestResponse {
  success: boolean;
  results: IngestResponse[];
}

export interface ApprovalStatus {
  approval_id: string;
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  decision?: 'approve' | 'reject';
  decided_by?: string;
  decided_at?: string;
  reason?: string;
}

export interface PolicyConfig {
  id: string;
  name: string;
  description?: string;
  rules: PolicyRule[];
  isActive: boolean;
}

export interface PolicyRule {
  id: string;
  actionType: ActionType;
  target: string | RegExp;
  decision: Decision;
  conditions?: RuleCondition[];
}

export interface RuleCondition {
  field: string;
  operator: 'eq' | 'ne' | 'contains' | 'startsWith' | 'endsWith' | 'regex';
  value: string | number | boolean;
}

export interface WebhookEvent {
  event: 'approval.created' | 'approval.decided' | 'policy.updated';
  data: Record<string, unknown>;
  timestamp: string;
}

export interface HarbyxConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  defaultAgentId?: string;
}

export class HarbyxError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public code?: string
  ) {
    super(message);
    this.name = 'HarbyxError';
  }
}
