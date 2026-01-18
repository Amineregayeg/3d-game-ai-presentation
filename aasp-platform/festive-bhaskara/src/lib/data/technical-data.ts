import {
  Database,
  Shield,
  Network,
  Cpu,
  Lock,
  FileCode,
  Layers,
  GitBranch,
  Server,
  Cloud,
  Key,
  Bell,
  BarChart3,
  RefreshCw,
} from 'lucide-react';

export const architectureData = {
  title: 'Platform Architecture',
  subtitle: 'A multi-layered security platform designed for scale, reliability, and real-time enforcement.',
  overview: {
    description: 'AASP operates as a control plane between AI agents and their target systems. Every action flows through our security layer, enabling real-time policy enforcement and comprehensive audit logging.',
  },
  layers: [
    {
      id: 'ingestion',
      name: 'Ingestion Layer',
      icon: Network,
      description: 'Captures agent actions from any source',
      components: [
        'MCP Proxy',
        'SDK Interceptors',
        'API Gateway',
        'Webhook Listeners',
      ],
      color: '#3B82F6',
    },
    {
      id: 'identity',
      name: 'Identity Layer',
      icon: Key,
      description: 'First-class agent identity management',
      components: [
        'Agent Registry',
        'Credential Vault',
        'Permission Scopes',
        'Lineage Tracking',
      ],
      color: '#10B981',
    },
    {
      id: 'policy',
      name: 'Policy Engine',
      icon: Shield,
      description: 'Real-time rule evaluation and enforcement',
      components: [
        'Rule DSL',
        'Allow/Block Lists',
        'Rate Limiters',
        'Anomaly Detection',
      ],
      color: '#8B5CF6',
    },
    {
      id: 'audit',
      name: 'Audit Layer',
      icon: Database,
      description: 'Immutable, queryable action logs',
      components: [
        'Event Store',
        'Context Capture',
        'Replay Engine',
        'Export APIs',
      ],
      color: '#F59E0B',
    },
    {
      id: 'workflow',
      name: 'Workflow Layer',
      icon: GitBranch,
      description: 'Human-in-the-loop approval flows',
      components: [
        'Approval Queues',
        'Escalation Rules',
        'Notification Hub',
        'Delegation',
      ],
      color: '#EC4899',
    },
  ],
  dataFlow: [
    { from: 'Agent', to: 'Ingestion', label: 'Action Request' },
    { from: 'Ingestion', to: 'Identity', label: 'Authenticate' },
    { from: 'Identity', to: 'Policy', label: 'Evaluate' },
    { from: 'Policy', to: 'Workflow', label: 'If Required' },
    { from: 'Policy', to: 'Target', label: 'Execute' },
    { from: 'Target', to: 'Audit', label: 'Log Result' },
  ],
};

export const coreLayersData = {
  title: 'Core Capabilities',
  subtitle: 'Deep dive into each layer of the security platform.',
  layers: [
    {
      id: 'registry',
      title: 'Agent Registry & Identity',
      icon: Key,
      description: 'Before you can secure agents, you need to know they exist. Our registry is the source of truth for all AI agents in your organization.',
      capabilities: [
        {
          name: 'Agent Registration',
          description: 'Register every agent with name, owner, purpose, and permissions',
        },
        {
          name: 'Credential Issuance',
          description: 'Agent-specific credentials that replace shared user tokens',
        },
        {
          name: 'Lineage Tracking',
          description: 'Track who created each agent, what framework, what model',
        },
        {
          name: 'Permission Scopes',
          description: 'Fine-grained scopes: read, write, external, sensitive',
        },
      ],
      codeExample: `// Register a new agent
const agent = await aasp.agents.register({
  name: "invoice-processor",
  owner: "finance-team",
  framework: "langchain",
  model: "gpt-4",
  scopes: ["read:invoices", "write:payments"],
  maxActionsPerMinute: 100,
});

// Agent receives unique credentials
console.log(agent.credentials);
// { agentId: "agt_xxx", apiKey: "sk_xxx" }`,
    },
    {
      id: 'logging',
      title: 'Action Logging & Replay',
      icon: Database,
      description: 'Every action captured with full context. Immutable audit logs that enable replay and investigation.',
      capabilities: [
        {
          name: 'Action Capture',
          description: 'Tool calls, API calls, data access, file operations',
        },
        {
          name: 'Context Preservation',
          description: 'Full conversation context, decision path, input/output',
        },
        {
          name: 'Immutable Storage',
          description: 'Append-only logs with cryptographic integrity',
        },
        {
          name: 'Replay Engine',
          description: 'Reconstruct exactly what happened at any point in time',
        },
      ],
      codeExample: `// Query agent actions
const actions = await aasp.audit.query({
  agentId: "agt_xxx",
  timeRange: { from: "2024-01-01", to: "2024-01-31" },
  actionTypes: ["api_call", "data_write"],
  status: "blocked",
});

// Each action includes full context
console.log(actions[0]);
// {
//   id: "act_xxx",
//   agent: "invoice-processor",
//   type: "api_call",
//   target: "payments.stripe.com",
//   input: { amount: 50000, recipient: "..." },
//   policy: "high_value_payment",
//   decision: "blocked",
//   reason: "Exceeds $10K threshold without approval"
// }`,
    },
    {
      id: 'policy',
      title: 'Policy Engine',
      icon: Shield,
      description: 'Explicit rules, not AI inference. Humans define what\'s sensitive. Zero false positives on explicit rules.',
      capabilities: [
        {
          name: 'Declarative Rules',
          description: 'Simple, readable policy definitions',
        },
        {
          name: 'Allow/Block Lists',
          description: 'Explicit lists for domains, tables, operations',
        },
        {
          name: 'Rate Limiting',
          description: 'Action frequency and cost controls',
        },
        {
          name: 'Conditional Logic',
          description: 'Context-aware rules based on agent, time, data',
        },
      ],
      codeExample: `// Define a policy
const policy = await aasp.policies.create({
  name: "payment-controls",
  rules: [
    {
      action: "api_call",
      target: "*.stripe.com/*",
      conditions: [
        { field: "amount", operator: "gt", value: 10000 },
      ],
      effect: "require_approval",
      approvers: ["finance-manager"],
    },
    {
      action: "data_read",
      target: "tables.pii_*",
      effect: "block",
      unless: { agent: { hasScope: "read:pii" } },
    },
  ],
});`,
    },
    {
      id: 'hitl',
      title: 'Human-in-the-Loop',
      icon: Bell,
      description: 'When policies trigger, route to humans seamlessly. Slack, Teams, email—wherever your team works.',
      capabilities: [
        {
          name: 'Approval Workflows',
          description: 'Route sensitive actions to appropriate approvers',
        },
        {
          name: 'Multi-Channel',
          description: 'Slack, Teams, email, custom webhooks',
        },
        {
          name: 'Escalation',
          description: 'Timeout-based escalation to backup approvers',
        },
        {
          name: 'Delegation',
          description: 'Temporary delegation for OOO coverage',
        },
      ],
      codeExample: `// Configure approval workflow
const workflow = await aasp.workflows.create({
  name: "high-value-approvals",
  trigger: { policy: "payment-controls" },
  steps: [
    {
      type: "approval",
      approvers: ["@finance-manager"],
      channels: ["slack:#finance-approvals"],
      timeout: "1h",
      escalateTo: ["@cfo"],
    },
  ],
  onApprove: "execute",
  onReject: "block_and_notify",
  onTimeout: "escalate",
});`,
    },
  ],
};

export const integrationData = {
  title: 'Integrations',
  subtitle: 'Works with your existing stack. Deploy in minutes, not months.',
  categories: [
    {
      name: 'Agent Frameworks',
      items: [
        { name: 'MCP', status: 'supported', logo: '/logos/mcp.svg' },
        { name: 'LangChain', status: 'supported', logo: '/logos/langchain.svg' },
        { name: 'AutoGen', status: 'supported', logo: '/logos/autogen.svg' },
        { name: 'CrewAI', status: 'coming', logo: '/logos/crewai.svg' },
        { name: 'Custom', status: 'supported', logo: '/logos/custom.svg' },
      ],
    },
    {
      name: 'Identity Providers',
      items: [
        { name: 'Okta', status: 'supported', logo: '/logos/okta.svg' },
        { name: 'Azure AD', status: 'supported', logo: '/logos/azure.svg' },
        { name: 'Google Workspace', status: 'supported', logo: '/logos/google.svg' },
        { name: 'SAML', status: 'supported', logo: '/logos/saml.svg' },
      ],
    },
    {
      name: 'Notification Channels',
      items: [
        { name: 'Slack', status: 'supported', logo: '/logos/slack.svg' },
        { name: 'Microsoft Teams', status: 'supported', logo: '/logos/teams.svg' },
        { name: 'Email', status: 'supported', logo: '/logos/email.svg' },
        { name: 'PagerDuty', status: 'coming', logo: '/logos/pagerduty.svg' },
      ],
    },
    {
      name: 'SIEM/SOAR',
      items: [
        { name: 'Splunk', status: 'supported', logo: '/logos/splunk.svg' },
        { name: 'Datadog', status: 'supported', logo: '/logos/datadog.svg' },
        { name: 'Sentinel', status: 'coming', logo: '/logos/sentinel.svg' },
        { name: 'Chronicle', status: 'coming', logo: '/logos/chronicle.svg' },
      ],
    },
  ],
  deployment: [
    {
      name: 'Cloud (SaaS)',
      description: 'Fully managed, SOC2 certified',
      features: ['Auto-scaling', 'Global edge', '99.99% SLA'],
    },
    {
      name: 'Hybrid',
      description: 'Control plane in cloud, data plane on-prem',
      features: ['Data sovereignty', 'Low latency', 'Flexible'],
    },
    {
      name: 'On-Premise',
      description: 'Full deployment in your infrastructure',
      features: ['Air-gapped', 'Full control', 'Enterprise'],
    },
  ],
};

export const techStackData = {
  title: 'Technology Stack',
  subtitle: 'Built for scale, reliability, and performance.',
  stack: [
    {
      category: 'Backend',
      items: ['Go', 'Rust', 'PostgreSQL', 'Redis', 'Kafka'],
    },
    {
      category: 'Frontend',
      items: ['Next.js', 'React', 'TypeScript', 'TailwindCSS'],
    },
    {
      category: 'Infrastructure',
      items: ['Kubernetes', 'Terraform', 'AWS/GCP', 'Cloudflare'],
    },
    {
      category: 'Security',
      items: ['mTLS', 'KMS', 'Vault', 'SOC2'],
    },
  ],
  performance: [
    { metric: 'Latency (p99)', value: '<10ms', description: 'Policy evaluation' },
    { metric: 'Throughput', value: '100K+', description: 'Actions per second' },
    { metric: 'Availability', value: '99.99%', description: 'SLA guarantee' },
    { metric: 'Retention', value: '7 years', description: 'Configurable logs' },
  ],
};

// Core capabilities for the core-layers component
export const coreCapabilities = [
  {
    name: 'Action Interception',
    description: 'Capture and control every action AI agents take in real-time.',
    features: [
      'Transparent proxy for MCP protocol',
      'SDK interceptors for popular frameworks',
      'Zero-latency inline processing',
      'Request/response modification support',
    ],
    codeExample: `// Wrap your agent with AASP
import { AASPClient } from '@aasp/sdk';

const aasp = new AASPClient({
  apiKey: process.env.AASP_API_KEY,
});

// All tool calls now flow through AASP
const agent = aasp.wrap(myAgent, {
  agentId: 'invoice-processor',
  policies: ['payment-controls'],
});`,
  },
  {
    name: 'Policy Engine',
    description: 'Explicit rules, not AI inference. Humans define what\'s allowed.',
    features: [
      'Declarative policy language',
      'Allow/block lists for domains and operations',
      'Rate limiting and cost controls',
      'Context-aware conditional logic',
    ],
    codeExample: `// Define explicit security policies
const policy = {
  name: "payment-controls",
  rules: [
    {
      action: "api_call",
      target: "*.stripe.com/*",
      when: { amount: { gt: 10000 } },
      effect: "require_approval",
    },
    {
      action: "data_read",
      target: "tables.pii_*",
      effect: "block",
    },
  ],
};`,
  },
  {
    name: 'Audit Trail',
    description: 'Immutable, queryable logs with full context preservation.',
    features: [
      'Every action captured with context',
      'Cryptographically signed logs',
      'Full conversation replay',
      'Compliance-ready exports',
    ],
    codeExample: `// Query the audit trail
const actions = await aasp.audit.query({
  agentId: "invoice-processor",
  timeRange: { from: "2024-01-01" },
  actionTypes: ["api_call"],
  status: "blocked",
});

// Each action includes full context
// {
//   id: "act_xxx",
//   agent: "invoice-processor",
//   decision: "blocked",
//   reason: "Exceeds threshold"
// }`,
  },
  {
    name: 'Human-in-the-Loop',
    description: 'Route sensitive actions to humans seamlessly.',
    features: [
      'Slack and Teams integration',
      'Customizable approval workflows',
      'Timeout-based escalation',
      'Delegation and OOO support',
    ],
    codeExample: `// Configure approval workflow
const workflow = {
  trigger: { policy: "payment-controls" },
  steps: [{
    type: "approval",
    approvers: ["@finance-manager"],
    channels: ["slack:#approvals"],
    timeout: "1h",
    escalateTo: ["@cfo"],
  }],
  onApprove: "execute",
  onReject: "block_and_notify",
};`,
  },
];

// Tech stack for display
export const techStack = [
  {
    category: 'Backend',
    technologies: [
      { name: 'Go', purpose: 'Core services' },
      { name: 'Rust', purpose: 'Performance critical' },
      { name: 'PostgreSQL', purpose: 'Primary database' },
      { name: 'Redis', purpose: 'Caching layer' },
    ],
  },
  {
    category: 'Frontend',
    technologies: [
      { name: 'Next.js', purpose: 'Dashboard' },
      { name: 'React', purpose: 'UI components' },
      { name: 'TypeScript', purpose: 'Type safety' },
      { name: 'TailwindCSS', purpose: 'Styling' },
    ],
  },
  {
    category: 'Infrastructure',
    technologies: [
      { name: 'Kubernetes', purpose: 'Orchestration' },
      { name: 'Terraform', purpose: 'IaC' },
      { name: 'AWS/GCP', purpose: 'Cloud providers' },
      { name: 'Cloudflare', purpose: 'Edge/CDN' },
    ],
  },
  {
    category: 'Security',
    technologies: [
      { name: 'mTLS', purpose: 'Transport security' },
      { name: 'Vault', purpose: 'Secrets management' },
      { name: 'KMS', purpose: 'Key management' },
      { name: 'SOC2', purpose: 'Compliance' },
    ],
  },
];

// Integrations data
export const integrations = [
  {
    category: 'AI Platforms',
    items: [
      { name: 'MCP Protocol', status: 'available' },
      { name: 'LangChain', status: 'available' },
      { name: 'AutoGen', status: 'beta' },
      { name: 'CrewAI', status: 'coming' },
    ],
  },
  {
    category: 'Identity Providers',
    items: [
      { name: 'Okta', status: 'available' },
      { name: 'Azure AD', status: 'available' },
      { name: 'Google Workspace', status: 'available' },
      { name: 'Auth0', status: 'beta' },
    ],
  },
  {
    category: 'Observability',
    items: [
      { name: 'Datadog', status: 'available' },
      { name: 'Splunk', status: 'available' },
      { name: 'New Relic', status: 'beta' },
      { name: 'Grafana', status: 'coming' },
    ],
  },
  {
    category: 'Developer Tools',
    items: [
      { name: 'GitHub', status: 'available' },
      { name: 'GitLab', status: 'available' },
      { name: 'Jira', status: 'beta' },
      { name: 'Linear', status: 'coming' },
    ],
  },
];

// Deployment options
export const deploymentOptions = [
  {
    name: 'Cloud (SaaS)',
    description: 'Fully managed deployment with automatic updates and scaling.',
    features: [
      'Auto-scaling infrastructure',
      'Global edge deployment',
      '99.99% uptime SLA',
      'SOC 2 Type II certified',
    ],
    recommended: true,
  },
  {
    name: 'Hybrid',
    description: 'Control plane in cloud, data plane in your infrastructure.',
    features: [
      'Data sovereignty compliance',
      'Low-latency local processing',
      'Centralized management',
      'Flexible architecture',
    ],
    recommended: false,
  },
  {
    name: 'On-Premise',
    description: 'Full deployment within your own infrastructure.',
    features: [
      'Air-gapped deployment',
      'Complete data control',
      'Custom integrations',
      'Enterprise support',
    ],
    recommended: false,
  },
];
