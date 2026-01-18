// Harbyx Policy Configuration - Salesforce Consultant Agent
// Policies specifically designed for Salesforce consulting delivery operations

import { PolicyConfig, PolicyRule, ActionType, Decision } from './types';

// ==================== Salesforce Consultant Policy Definitions ====================

/**
 * Salesforce Query Policy (Priority 100)
 * Controls read-only Salesforce operations - generally allowed
 */
export const salesforceQueryPolicy: PolicyConfig = {
  id: 'sf-consultant-query-policy',
  name: 'Salesforce Query Policy',
  description: 'Controls read-only Salesforce query operations',
  isActive: true,
  rules: [
    // SOQL queries - Allowed
    { id: 'allow-sf-query', actionType: 'api_call', target: 'salesforce:query', decision: 'allow' },
    { id: 'allow-sf-query-all', actionType: 'api_call', target: 'salesforce:queryAll', decision: 'allow' },

    // Metadata describe - Allowed
    { id: 'allow-sf-describe', actionType: 'api_call', target: 'salesforce:describe', decision: 'allow' },
    { id: 'allow-sf-describe-global', actionType: 'api_call', target: 'salesforce:describeGlobal', decision: 'allow' },
    { id: 'allow-sf-describe-sobject', actionType: 'api_call', target: 'salesforce:describeSObject', decision: 'allow' },

    // Org limits and info - Allowed
    { id: 'allow-sf-limits', actionType: 'api_call', target: 'salesforce:limits', decision: 'allow' },
    { id: 'allow-sf-org-info', actionType: 'api_call', target: 'salesforce:orgInfo', decision: 'allow' },

    // SOSL search - Allowed
    { id: 'allow-sf-search', actionType: 'api_call', target: 'salesforce:search', decision: 'allow' },

    // Record retrieval - Allowed
    { id: 'allow-sf-retrieve', actionType: 'api_call', target: 'salesforce:retrieve', decision: 'allow' },
    { id: 'allow-sf-get-record', actionType: 'api_call', target: 'salesforce:getRecord', decision: 'allow' },
  ],
};

/**
 * Salesforce Write Policy (Priority 90)
 * Controls data modification operations - requires approval
 */
export const salesforceWritePolicy: PolicyConfig = {
  id: 'sf-consultant-write-policy',
  name: 'Salesforce Write Policy',
  description: 'Controls Salesforce data modification operations',
  isActive: true,
  rules: [
    // Insert operations - Require approval
    { id: 'require-approval-sf-insert', actionType: 'api_call', target: 'salesforce:insert', decision: 'require_approval' },
    { id: 'require-approval-sf-create', actionType: 'api_call', target: 'salesforce:create', decision: 'require_approval' },

    // Update operations - Require approval
    { id: 'require-approval-sf-update', actionType: 'api_call', target: 'salesforce:update', decision: 'require_approval' },
    { id: 'require-approval-sf-patch', actionType: 'api_call', target: 'salesforce:patch', decision: 'require_approval' },

    // Upsert operations - Require approval
    { id: 'require-approval-sf-upsert', actionType: 'api_call', target: 'salesforce:upsert', decision: 'require_approval' },

    // Delete operations - BLOCKED (never allow)
    { id: 'block-sf-delete', actionType: 'api_call', target: 'salesforce:delete', decision: 'block' },
    { id: 'block-sf-hard-delete', actionType: 'api_call', target: 'salesforce:hardDelete', decision: 'block' },
    { id: 'block-sf-purge', actionType: 'api_call', target: 'salesforce:purge', decision: 'block' },

    // Bulk operations - Require approval
    { id: 'require-approval-sf-bulk-insert', actionType: 'api_call', target: 'salesforce:bulk:insert', decision: 'require_approval' },
    { id: 'require-approval-sf-bulk-update', actionType: 'api_call', target: 'salesforce:bulk:update', decision: 'require_approval' },
    { id: 'block-sf-bulk-delete', actionType: 'api_call', target: 'salesforce:bulk:delete', decision: 'block' },
  ],
};

/**
 * Salesforce Deployment Policy (Priority 80)
 * Controls metadata and deployment operations
 */
export const salesforceDeployPolicy: PolicyConfig = {
  id: 'sf-consultant-deploy-policy',
  name: 'Salesforce Deployment Policy',
  description: 'Controls Salesforce metadata and deployment operations',
  isActive: true,
  rules: [
    // Metadata read - Allowed
    { id: 'allow-sf-metadata-read', actionType: 'api_call', target: 'salesforce:metadata:read', decision: 'allow' },
    { id: 'allow-sf-metadata-describe', actionType: 'api_call', target: 'salesforce:metadata:describe', decision: 'allow' },
    { id: 'allow-sf-metadata-list', actionType: 'api_call', target: 'salesforce:metadata:list', decision: 'allow' },

    // Metadata write - Require approval
    { id: 'require-approval-sf-metadata-write', actionType: 'api_call', target: 'salesforce:metadata:write', decision: 'require_approval' },
    { id: 'require-approval-sf-metadata-create', actionType: 'api_call', target: 'salesforce:metadata:create', decision: 'require_approval' },
    { id: 'require-approval-sf-metadata-update', actionType: 'api_call', target: 'salesforce:metadata:update', decision: 'require_approval' },

    // Sandbox deployment - Require approval
    { id: 'require-approval-sf-deploy', actionType: 'api_call', target: 'salesforce:deploy', decision: 'require_approval' },
    { id: 'require-approval-sf-deploy-sandbox', actionType: 'api_call', target: 'salesforce:deploy:sandbox', decision: 'require_approval' },

    // Production deployment - BLOCKED (Type C - too risky)
    { id: 'block-sf-deploy-production', actionType: 'api_call', target: 'salesforce:deploy:production', decision: 'block' },
    { id: 'block-sf-deploy-prod', actionType: 'api_call', target: 'salesforce:deploy:prod', decision: 'block' },

    // Destructive changes - BLOCKED
    { id: 'block-sf-metadata-delete', actionType: 'api_call', target: 'salesforce:metadata:delete', decision: 'block' },
    { id: 'block-sf-destructive-deploy', actionType: 'api_call', target: 'salesforce:deploy:destructive', decision: 'block' },
  ],
};

/**
 * AI Agent Tool Policy (Priority 70)
 * Controls what tools AI agents can use
 */
export const agentToolPolicy: PolicyConfig = {
  id: 'sf-consultant-tool-policy',
  name: 'Agent Tool Policy',
  description: 'Controls AI agent tool usage and code execution',
  isActive: true,
  rules: [
    // RAG operations - Allowed
    { id: 'allow-rag-query', actionType: 'tool_call', target: 'rag:query', decision: 'allow' },
    { id: 'allow-rag-search', actionType: 'tool_call', target: 'rag:search', decision: 'allow' },
    { id: 'allow-rag-retrieve', actionType: 'tool_call', target: 'rag:retrieve', decision: 'allow' },

    // MCP Salesforce read operations - Allowed
    { id: 'allow-mcp-sf-read', actionType: 'tool_call', target: 'mcp:salesforce:*:read', decision: 'allow' },
    { id: 'allow-mcp-sf-query', actionType: 'tool_call', target: 'mcp:salesforce:*:query', decision: 'allow' },
    { id: 'allow-mcp-sf-describe', actionType: 'tool_call', target: 'mcp:salesforce:*:describe', decision: 'allow' },

    // MCP Salesforce write operations - Require approval
    { id: 'require-approval-mcp-sf-write', actionType: 'tool_call', target: 'mcp:salesforce:*:write', decision: 'require_approval' },
    { id: 'require-approval-mcp-sf-create', actionType: 'tool_call', target: 'mcp:salesforce:*:create', decision: 'require_approval' },
    { id: 'require-approval-mcp-sf-update', actionType: 'tool_call', target: 'mcp:salesforce:*:update', decision: 'require_approval' },

    // Code execution - Require approval
    { id: 'require-approval-code-execute', actionType: 'tool_call', target: 'code:execute', decision: 'require_approval' },
    { id: 'require-approval-code-apex', actionType: 'tool_call', target: 'code:apex', decision: 'require_approval' },
    { id: 'require-approval-code-soql', actionType: 'tool_call', target: 'code:soql', decision: 'allow' },

    // Analysis tools - Allowed
    { id: 'allow-analysis', actionType: 'tool_call', target: 'analysis:*', decision: 'allow' },
    { id: 'allow-report-generator', actionType: 'tool_call', target: 'report:*', decision: 'allow' },

    // Dangerous tools - Blocked
    { id: 'block-shell-command', actionType: 'tool_call', target: 'shell:*', decision: 'block' },
    { id: 'block-system-access', actionType: 'tool_call', target: 'system:*', decision: 'block' },
  ],
};

/**
 * Database Access Policy (Priority 60)
 * Controls local database operations for the consulting platform
 */
export const databasePolicy: PolicyConfig = {
  id: 'sf-consultant-db-policy',
  name: 'Database Access Policy',
  description: 'Controls local database operations for consulting platform',
  isActive: true,
  rules: [
    // Read operations - Allowed
    { id: 'allow-read-sessions', actionType: 'db_query', target: 'sessions:read', decision: 'allow' },
    { id: 'allow-read-organizations', actionType: 'db_query', target: 'organizations:read', decision: 'allow' },
    { id: 'allow-read-users', actionType: 'db_query', target: 'users:read', decision: 'allow' },
    { id: 'allow-read-knowledge', actionType: 'db_query', target: 'knowledge_base:read', decision: 'allow' },
    { id: 'allow-read-patterns', actionType: 'db_query', target: 'patterns:read', decision: 'allow' },
    { id: 'allow-read-audit', actionType: 'db_query', target: 'audit_logs:read', decision: 'allow' },

    // Create operations - Allowed
    { id: 'allow-create-sessions', actionType: 'db_query', target: 'sessions:create', decision: 'allow' },
    { id: 'allow-create-audit', actionType: 'db_query', target: 'audit_logs:create', decision: 'allow' },

    // Update operations - Mixed
    { id: 'allow-update-sessions', actionType: 'db_query', target: 'sessions:update', decision: 'allow' },
    { id: 'require-approval-update-users', actionType: 'db_query', target: 'users:update', decision: 'require_approval' },
    { id: 'require-approval-update-orgs', actionType: 'db_query', target: 'organizations:update', decision: 'require_approval' },

    // Delete operations - Require approval or block
    { id: 'require-approval-delete-sessions', actionType: 'db_query', target: 'sessions:delete', decision: 'require_approval' },
    { id: 'block-delete-users', actionType: 'db_query', target: 'users:delete', decision: 'block' },
    { id: 'block-delete-orgs', actionType: 'db_query', target: 'organizations:delete', decision: 'block' },
    { id: 'block-delete-audit', actionType: 'db_query', target: 'audit_logs:delete', decision: 'block' },
  ],
};

/**
 * External API Policy (Priority 50)
 * Controls calls to external services
 */
export const externalApiPolicy: PolicyConfig = {
  id: 'sf-consultant-api-policy',
  name: 'External API Policy',
  description: 'Controls external API calls and integrations',
  isActive: true,
  rules: [
    // AI Services - Allowed
    { id: 'allow-openai', actionType: 'api_call', target: 'openai:*', decision: 'allow' },
    { id: 'allow-anthropic', actionType: 'api_call', target: 'anthropic:*', decision: 'allow' },
    { id: 'allow-harbyx', actionType: 'api_call', target: 'harbyx:*', decision: 'allow' },
    { id: 'allow-elevenlabs', actionType: 'api_call', target: 'elevenlabs:*', decision: 'allow' },

    // Jira Integration - Require approval
    { id: 'require-approval-jira-read', actionType: 'api_call', target: 'jira:read', decision: 'allow' },
    { id: 'require-approval-jira-write', actionType: 'api_call', target: 'jira:write', decision: 'require_approval' },
    { id: 'require-approval-jira-create', actionType: 'api_call', target: 'jira:create', decision: 'require_approval' },

    // Slack Integration - Require approval
    { id: 'allow-slack-read', actionType: 'api_call', target: 'slack:read', decision: 'allow' },
    { id: 'require-approval-slack-post', actionType: 'api_call', target: 'slack:post', decision: 'require_approval' },

    // Unknown external requests - Require approval
    { id: 'require-approval-external', actionType: 'external_request', target: '*', decision: 'require_approval' },
  ],
};

// Combine all policies (ordered by priority)
export const defaultPolicies: PolicyConfig[] = [
  salesforceQueryPolicy,     // Priority 100 - SF reads
  salesforceWritePolicy,     // Priority 90 - SF writes
  salesforceDeployPolicy,    // Priority 80 - SF deployments
  agentToolPolicy,           // Priority 70 - Agent tools
  databasePolicy,            // Priority 60 - Local DB
  externalApiPolicy,         // Priority 50 - External APIs
];

// Helper to match actions against rules
export function matchesRule(
  actionType: ActionType,
  target: string,
  rule: PolicyRule
): boolean {
  if (rule.actionType !== actionType) return false;

  const ruleTarget = rule.target;
  if (typeof ruleTarget === 'string') {
    // Handle wildcards
    if (ruleTarget === '*') return true;
    if (ruleTarget.includes('*')) {
      const regex = new RegExp(
        '^' + ruleTarget.replace(/\*/g, '.*').replace(/\|/g, '|') + '$'
      );
      return regex.test(target);
    }
    // Handle pipe-separated values
    if (ruleTarget.includes('|')) {
      return ruleTarget.split('|').some((t) => t === target);
    }
    return ruleTarget === target;
  }
  return ruleTarget.test(target);
}

// Evaluate action against local policies (fallback when API unavailable)
export function evaluateLocally(
  actionType: ActionType,
  target: string,
  policies: PolicyConfig[] = defaultPolicies
): Decision {
  for (const policy of policies) {
    if (!policy.isActive) continue;

    for (const rule of policy.rules) {
      if (matchesRule(actionType, target, rule)) {
        return rule.decision;
      }
    }
  }

  // Default: require approval for unknown actions
  return 'require_approval';
}

// Environment configuration
export const harbyxConfig = {
  apiKey: process.env.HARBYX_API_KEY || '',
  baseUrl: process.env.HARBYX_BASE_URL || 'https://app.harbyx.com',
  defaultAgentId: process.env.HARBYX_AGENT_ID || 'salesforce-consultant-agent',
  enableLocalFallback: process.env.HARBYX_LOCAL_FALLBACK !== 'false',
  mockMode: process.env.HARBYX_MOCK_MODE === 'true',
};
