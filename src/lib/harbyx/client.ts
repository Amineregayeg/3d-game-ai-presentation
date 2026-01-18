// Harbyx Governance SDK Client - Salesforce Consultant Agent

import {
  HarbyxConfig,
  HarbyxError,
  IngestParams,
  IngestResponse,
  BatchIngestParams,
  BatchIngestResponse,
  ApprovalStatus,
} from './types';

const DEFAULT_BASE_URL = 'https://app.harbyx.com';
const DEFAULT_TIMEOUT = 30000; // 30 seconds

export class HarbyxClient {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  private defaultAgentId: string;

  constructor(config: HarbyxConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || DEFAULT_BASE_URL;
    this.timeout = config.timeout || DEFAULT_TIMEOUT;
    this.defaultAgentId = config.defaultAgentId || 'salesforce-consultant-agent';
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          ...options.headers,
        },
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new HarbyxError(
          errorData.message || `Request failed with status ${response.status}`,
          response.status,
          errorData.code
        );
      }

      return response.json();
    } catch (error) {
      if (error instanceof HarbyxError) throw error;
      if (error instanceof Error && error.name === 'AbortError') {
        throw new HarbyxError('Request timed out', 408, 'TIMEOUT');
      }
      throw new HarbyxError(
        error instanceof Error ? error.message : 'Unknown error occurred'
      );
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Evaluate a single action against governance policies
   */
  async evaluateAction(params: IngestParams): Promise<IngestResponse> {
    return this.request<IngestResponse>('/api/v1/ingest', {
      method: 'POST',
      body: JSON.stringify({
        agent_id: params.agentId || this.defaultAgentId,
        action_type: params.actionType,
        target: params.target,
        params: params.params || {},
        metadata: params.metadata || {},
      }),
    });
  }

  /**
   * Evaluate a single action in dry-run mode (no side effects)
   */
  async evaluateActionDryRun(params: IngestParams): Promise<IngestResponse> {
    return this.request<IngestResponse>('/api/v1/ingest?dry_run=true', {
      method: 'POST',
      body: JSON.stringify({
        agent_id: params.agentId || this.defaultAgentId,
        action_type: params.actionType,
        target: params.target,
        params: params.params || {},
        metadata: params.metadata || {},
      }),
    });
  }

  /**
   * Evaluate multiple actions in batch (max 100 per request)
   */
  async evaluateActionsBatch(
    params: BatchIngestParams
  ): Promise<BatchIngestResponse> {
    if (params.actions.length > 100) {
      throw new HarbyxError(
        'Batch size cannot exceed 100 actions',
        400,
        'BATCH_TOO_LARGE'
      );
    }

    return this.request<BatchIngestResponse>('/api/v1/ingest/batch', {
      method: 'POST',
      body: JSON.stringify({
        actions: params.actions.map((action) => ({
          agent_id: action.agentId || this.defaultAgentId,
          action_type: action.actionType,
          target: action.target,
          params: action.params || {},
          metadata: action.metadata || {},
        })),
      }),
    });
  }

  /**
   * Check the status of an approval request
   */
  async getApprovalStatus(approvalId: string): Promise<ApprovalStatus> {
    return this.request<ApprovalStatus>(
      `/api/v1/approvals/${approvalId}/status`
    );
  }

  /**
   * List pending approvals
   */
  async listPendingApprovals(): Promise<ApprovalStatus[]> {
    return this.request<ApprovalStatus[]>('/api/v1/approvals?status=pending');
  }

  /**
   * Approve or reject an action
   */
  async decideApproval(
    approvalId: string,
    decision: 'approve' | 'reject',
    reason?: string
  ): Promise<ApprovalStatus> {
    return this.request<ApprovalStatus>(`/api/v1/approvals/${approvalId}`, {
      method: 'POST',
      body: JSON.stringify({ decision, reason }),
    });
  }

  /**
   * Register a webhook for real-time notifications
   */
  async registerWebhook(
    url: string,
    events: ('approval.created' | 'approval.decided' | 'policy.updated')[]
  ): Promise<{ webhook_id: string }> {
    return this.request<{ webhook_id: string }>('/api/v1/webhooks', {
      method: 'POST',
      body: JSON.stringify({ url, events }),
    });
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; connected: boolean }> {
    try {
      // Try a dry-run request to verify connectivity
      await this.evaluateActionDryRun({
        agentId: 'health-check',
        actionType: 'api_call',
        target: 'health',
      });
      return { status: 'ok', connected: true };
    } catch (error) {
      return {
        status: error instanceof HarbyxError ? error.message : 'unknown error',
        connected: false,
      };
    }
  }

  // ==================== Policy Management ====================

  /**
   * Create a new policy in Harbyx
   */
  async createPolicy(policy: {
    name: string;
    description?: string;
    priority?: number;
    rules: Array<{
      actionType: string;
      targetPattern: string;
      conditions?: Array<{
        field: string;
        operator: string;
        value: string | number | boolean;
      }>;
      effect: 'allow' | 'block' | 'require_approval';
    }>;
  }): Promise<{ policy_id: string; name: string }> {
    return this.request<{ policy_id: string; name: string }>('/api/v1/policies', {
      method: 'POST',
      body: JSON.stringify(policy),
    });
  }

  /**
   * List all policies
   */
  async listPolicies(): Promise<Array<{
    id: string;
    name: string;
    description?: string;
    priority: number;
    rules: unknown[];
    created_at: string;
    updated_at: string;
  }>> {
    return this.request('/api/v1/policies');
  }

  /**
   * Delete a policy
   */
  async deletePolicy(policyId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/api/v1/policies/${policyId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Update a policy
   */
  async updatePolicy(
    policyId: string,
    policy: {
      name?: string;
      description?: string;
      priority?: number;
      rules?: Array<{
        actionType: string;
        targetPattern: string;
        conditions?: Array<{
          field: string;
          operator: string;
          value: string | number | boolean;
        }>;
        effect: 'allow' | 'block' | 'require_approval';
      }>;
    }
  ): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/api/v1/policies/${policyId}`, {
      method: 'PATCH',
      body: JSON.stringify(policy),
    });
  }
}

// Singleton instance for convenience
let clientInstance: HarbyxClient | null = null;

export function getHarbyxClient(): HarbyxClient {
  if (!clientInstance) {
    const apiKey = process.env.HARBYX_API_KEY;
    if (!apiKey) {
      throw new HarbyxError(
        'HARBYX_API_KEY environment variable is not set',
        500,
        'MISSING_API_KEY'
      );
    }
    clientInstance = new HarbyxClient({ apiKey });
  }
  return clientInstance;
}

// Convenience function for quick action evaluation
export async function evaluateAction(
  params: IngestParams
): Promise<IngestResponse> {
  return getHarbyxClient().evaluateAction(params);
}
