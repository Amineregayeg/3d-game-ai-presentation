import type { DemoAgent, ActionTemplate, Policy } from '@/lib/demo/types';

// Pre-configured demo agents
export const demoAgents: DemoAgent[] = [
  {
    id: 'invoice-bot',
    name: 'Invoice Processor',
    icon: 'FileText',
    description: 'Automates invoice processing and payment workflows',
    color: '#3B82F6',
  },
  {
    id: 'data-analyst',
    name: 'Data Analyst',
    icon: 'BarChart3',
    description: 'Queries databases and generates analytical reports',
    color: '#10B981',
  },
  {
    id: 'customer-support',
    name: 'Support Agent',
    icon: 'MessageSquare',
    description: 'Handles customer inquiries and ticket resolution',
    color: '#8B5CF6',
  },
  {
    id: 'code-assistant',
    name: 'Code Assistant',
    icon: 'Code',
    description: 'Assists with code generation and repository access',
    color: '#F59E0B',
  },
  {
    id: 'security-scanner',
    name: 'Security Scanner',
    icon: 'Shield',
    description: 'Performs automated security audits and vulnerability scans',
    color: '#EF4444',
  },
];

// Action type templates
export const actionTemplates: ActionTemplate[] = [
  {
    type: 'api_call',
    label: 'API Call',
    icon: 'Globe',
    placeholder: 'https://api.stripe.com/v1/charges',
    sampleParams: { amount: 5000, currency: 'usd' },
  },
  {
    type: 'db_query',
    label: 'Database Query',
    icon: 'Database',
    placeholder: 'SELECT * FROM customers WHERE id = ?',
    sampleParams: { table: 'customers', operation: 'read' },
  },
  {
    type: 'file_access',
    label: 'File Access',
    icon: 'FileText',
    placeholder: '/data/reports/financial-2024.xlsx',
    sampleParams: { operation: 'read', sensitive: false },
  },
  {
    type: 'external',
    label: 'External Service',
    icon: 'ExternalLink',
    placeholder: 'slack://post-message',
    sampleParams: { channel: '#general', message: 'Alert notification' },
  },
];

// Default demo policies
export const defaultPolicies: Policy[] = [
  {
    id: 'payment-controls',
    name: 'Payment Controls',
    description: 'Require approval for payments over $10,000',
    enabled: true,
    rules: [
      {
        id: 'pc-1',
        actionType: 'api_call',
        targetPattern: '.*stripe.*|.*payment.*',
        conditions: [{ field: 'amount', operator: 'gt', value: 10000 }],
        effect: 'require_approval',
        approvers: ['finance-team'],
      },
      {
        id: 'pc-2',
        actionType: 'api_call',
        targetPattern: '.*stripe.*|.*payment.*',
        effect: 'allow',
      },
    ],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'data-access',
    name: 'Sensitive Data Access',
    description: 'Block access to PII and financial data without authorization',
    enabled: true,
    rules: [
      {
        id: 'da-1',
        actionType: 'db_query',
        targetPattern: '.*users.*|.*customers.*|.*pii.*',
        conditions: [{ field: 'operation', operator: 'eq', value: 'delete' }],
        effect: 'block',
      },
      {
        id: 'da-2',
        actionType: 'db_query',
        targetPattern: '.*financial.*|.*salary.*|.*ssn.*',
        effect: 'require_approval',
        approvers: ['data-governance'],
      },
    ],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'file-restrictions',
    name: 'File Access Restrictions',
    description: 'Control access to sensitive file paths',
    enabled: true,
    rules: [
      {
        id: 'fr-1',
        actionType: 'file_access',
        targetPattern: '.*\\.env.*|.*credentials.*|.*secret.*',
        effect: 'block',
      },
      {
        id: 'fr-2',
        actionType: 'file_access',
        targetPattern: '.*/confidential/.*',
        effect: 'require_approval',
        approvers: ['compliance-team'],
      },
    ],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'rate-limiting',
    name: 'Rate Limiting',
    description: 'Prevent excessive API calls',
    enabled: false,
    rules: [
      {
        id: 'rl-1',
        actionType: 'api_call',
        conditions: [{ field: 'rate', operator: 'gt', value: 100 }],
        effect: 'block',
      },
    ],
    createdAt: new Date().toISOString(),
  },
  {
    id: 'external-comms',
    name: 'External Communications',
    description: 'Monitor and control external messaging',
    enabled: true,
    rules: [
      {
        id: 'ec-1',
        actionType: 'external',
        targetPattern: '.*slack.*|.*email.*|.*teams.*',
        conditions: [{ field: 'channel', operator: 'contains', value: 'external' }],
        effect: 'require_approval',
        approvers: ['communications-team'],
      },
    ],
    createdAt: new Date().toISOString(),
  },
];

// Sample actions for demo
export const sampleActions = [
  {
    agentId: 'invoice-bot',
    actionType: 'api_call' as const,
    target: 'https://api.stripe.com/v1/charges',
    params: { amount: 15000, currency: 'usd', description: 'Invoice #INV-2024-001' },
  },
  {
    agentId: 'data-analyst',
    actionType: 'db_query' as const,
    target: 'SELECT * FROM customers WHERE region = ?',
    params: { table: 'customers', operation: 'read', region: 'EMEA' },
  },
  {
    agentId: 'code-assistant',
    actionType: 'file_access' as const,
    target: '/src/config/.env.production',
    params: { operation: 'read' },
  },
  {
    agentId: 'customer-support',
    actionType: 'external' as const,
    target: 'slack://post-message',
    params: { channel: '#support-external', message: 'Customer escalation resolved' },
  },
  {
    agentId: 'security-scanner',
    actionType: 'api_call' as const,
    target: 'https://api.github.com/repos/org/app/vulnerabilities',
    params: { scan_type: 'full', include_dependencies: true },
  },
];

// Policy templates for the builder
export const policyTemplates = [
  {
    id: 'template-payment',
    name: 'Payment Threshold',
    description: 'Require approval for payments above a threshold',
    icon: 'DollarSign',
    baseRule: {
      actionType: 'api_call' as const,
      targetPattern: '.*payment.*|.*stripe.*|.*charge.*',
      conditions: [{ field: 'amount', operator: 'gt' as const, value: 1000 }],
      effect: 'require_approval' as const,
    },
  },
  {
    id: 'template-data',
    name: 'Data Protection',
    description: 'Block access to sensitive data tables',
    icon: 'Shield',
    baseRule: {
      actionType: 'db_query' as const,
      targetPattern: '.*pii.*|.*sensitive.*',
      effect: 'block' as const,
    },
  },
  {
    id: 'template-file',
    name: 'File Restrictions',
    description: 'Control access to specific file paths',
    icon: 'FileX',
    baseRule: {
      actionType: 'file_access' as const,
      targetPattern: '.*secret.*|.*credential.*',
      effect: 'block' as const,
    },
  },
  {
    id: 'template-external',
    name: 'External Service Control',
    description: 'Monitor external service communications',
    icon: 'ExternalLink',
    baseRule: {
      actionType: 'external' as const,
      effect: 'require_approval' as const,
    },
  },
];

// Condition operators for policy builder
export const conditionOperators = [
  { value: 'eq', label: 'Equals', symbol: '=' },
  { value: 'neq', label: 'Not Equals', symbol: '!=' },
  { value: 'gt', label: 'Greater Than', symbol: '>' },
  { value: 'lt', label: 'Less Than', symbol: '<' },
  { value: 'gte', label: 'Greater or Equal', symbol: '>=' },
  { value: 'lte', label: 'Less or Equal', symbol: '<=' },
  { value: 'contains', label: 'Contains', symbol: '~' },
  { value: 'matches', label: 'Matches Pattern', symbol: '~=' },
];

// Decision badge styles
export const decisionStyles = {
  allow: {
    bg: 'bg-green-500/20',
    text: 'text-green-400',
    border: 'border-green-500/30',
    icon: 'CheckCircle',
  },
  block: {
    bg: 'bg-red-500/20',
    text: 'text-red-400',
    border: 'border-red-500/30',
    icon: 'XCircle',
  },
  require_approval: {
    bg: 'bg-yellow-500/20',
    text: 'text-yellow-400',
    border: 'border-yellow-500/30',
    icon: 'Clock',
  },
};

// Approval status styles
export const approvalStatusStyles = {
  pending: {
    bg: 'bg-yellow-500/20',
    text: 'text-yellow-400',
    label: 'Pending Review',
  },
  approved: {
    bg: 'bg-green-500/20',
    text: 'text-green-400',
    label: 'Approved',
  },
  rejected: {
    bg: 'bg-red-500/20',
    text: 'text-red-400',
    label: 'Rejected',
  },
};
