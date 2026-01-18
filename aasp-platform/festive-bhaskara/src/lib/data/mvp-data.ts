import {
  CheckCircle2,
  Clock,
  Circle,
  Rocket,
  Shield,
  Database,
  GitBranch,
  BarChart3,
  Users,
  Zap,
  Target,
  Building2,
} from 'lucide-react';

export const mvpScopeData = {
  title: 'MVP Scope',
  subtitle: 'Focused, lethal, and shippable. Start narrow, expand with validation.',
  inScope: [
    {
      title: 'Agent Registry',
      description: 'Register agents, issue credentials, track lineage',
      priority: 'P0',
      status: 'building',
    },
    {
      title: 'Action Logging',
      description: 'Capture all actions with full context',
      priority: 'P0',
      status: 'building',
    },
    {
      title: 'MCP Proxy',
      description: 'Intercept and control MCP tool calls',
      priority: 'P0',
      status: 'building',
    },
    {
      title: 'Basic Policy Engine',
      description: 'Allow/block rules, rate limits',
      priority: 'P0',
      status: 'planned',
    },
    {
      title: 'Slack Approvals',
      description: 'Simple HITL via Slack',
      priority: 'P1',
      status: 'planned',
    },
    {
      title: 'Query Interface',
      description: 'Search and filter action logs',
      priority: 'P1',
      status: 'planned',
    },
  ],
  outOfScope: [
    'Advanced anomaly detection (Phase 2)',
    'Multi-framework SDKs (Phase 2)',
    'On-premise deployment (Phase 3)',
    'Custom compliance reports (Phase 3)',
    'AI-powered policy suggestions (Future)',
  ],
  principles: [
    {
      title: 'Explicit over Inferred',
      description: 'Humans define rules. No AI magic that customers can\'t trust.',
    },
    {
      title: 'Audit First',
      description: 'If we can\'t log it with context, we can\'t secure it.',
    },
    {
      title: 'Zero False Positives',
      description: 'Explicit rules mean predictable outcomes.',
    },
    {
      title: 'Framework Agnostic',
      description: 'Design for any agent, not just MCP.',
    },
  ],
};

export const featuresData = {
  title: 'Feature Breakdown',
  subtitle: 'Detailed view of MVP capabilities.',
  features: [
    {
      id: 'registry',
      name: 'Agent Registry',
      description: 'Central registry for all AI agents in your organization',
      icon: Database,
      status: 'in_progress',
      progress: 65,
      capabilities: [
        { name: 'Agent CRUD', status: 'done' },
        { name: 'Credential Generation', status: 'done' },
        { name: 'Scope Management', status: 'in_progress' },
        { name: 'Lineage Tracking', status: 'planned' },
      ],
    },
    {
      id: 'logging',
      name: 'Action Logging',
      description: 'Immutable audit logs with full context preservation',
      icon: BarChart3,
      status: 'in_progress',
      progress: 50,
      capabilities: [
        { name: 'Event Capture', status: 'done' },
        { name: 'Context Storage', status: 'in_progress' },
        { name: 'Query API', status: 'planned' },
        { name: 'Export/Replay', status: 'planned' },
      ],
    },
    {
      id: 'proxy',
      name: 'MCP Proxy',
      description: 'Transparent proxy for MCP tool calls',
      icon: Shield,
      status: 'in_progress',
      progress: 40,
      capabilities: [
        { name: 'Tool Interception', status: 'done' },
        { name: 'Request/Response Logging', status: 'in_progress' },
        { name: 'Policy Hook', status: 'planned' },
        { name: 'Modification Support', status: 'planned' },
      ],
    },
    {
      id: 'policy',
      name: 'Policy Engine',
      description: 'Rule-based policy evaluation and enforcement',
      icon: Shield,
      status: 'planned',
      progress: 10,
      capabilities: [
        { name: 'Rule DSL', status: 'in_progress' },
        { name: 'Allow/Block Lists', status: 'planned' },
        { name: 'Rate Limiting', status: 'planned' },
        { name: 'Conditional Logic', status: 'planned' },
      ],
    },
    {
      id: 'hitl',
      name: 'Human-in-the-Loop',
      description: 'Approval workflows via Slack',
      icon: Users,
      status: 'planned',
      progress: 0,
      capabilities: [
        { name: 'Slack Integration', status: 'planned' },
        { name: 'Approval Flow', status: 'planned' },
        { name: 'Timeout Handling', status: 'planned' },
        { name: 'Audit Trail', status: 'planned' },
      ],
    },
  ],
};

export const roadmapData = {
  title: 'Product Roadmap',
  subtitle: 'Our path from MVP to platform.',
  phases: [
    {
      id: 'phase1',
      name: 'Phase 1: Foundation',
      status: 'current',
      description: 'Core infrastructure and MCP support',
      milestones: [
        { title: 'Agent Registry', status: 'completed', date: 'Complete' },
        { title: 'Action Logging', status: 'completed', date: 'Complete' },
        { title: 'MCP Proxy', status: 'current', date: 'In Progress' },
        { title: 'Basic Policies', status: 'upcoming', date: 'Next' },
        { title: 'Slack HITL', status: 'upcoming', date: 'Next' },
      ],
      deliverables: [
        'Working MCP proxy',
        'Basic policy enforcement',
        'Audit log queries',
        'Design partner deployments',
      ],
    },
    {
      id: 'phase2',
      name: 'Phase 2: Expansion',
      status: 'upcoming',
      description: 'Multi-framework support and advanced features',
      milestones: [
        { title: 'LangChain SDK', status: 'upcoming', date: 'Planned' },
        { title: 'AutoGen SDK', status: 'upcoming', date: 'Planned' },
        { title: 'Advanced Policies', status: 'upcoming', date: 'Planned' },
        { title: 'Anomaly Detection', status: 'upcoming', date: 'Planned' },
        { title: 'SIEM Integration', status: 'upcoming', date: 'Planned' },
      ],
      deliverables: [
        'Framework-agnostic coverage',
        'Statistical anomaly alerts',
        'Enterprise integrations',
        'First paying customers',
      ],
    },
    {
      id: 'phase3',
      name: 'Phase 3: Platform',
      status: 'upcoming',
      description: 'Enterprise features and scale',
      milestones: [
        { title: 'Compliance Dashboards', status: 'upcoming', date: 'Planned' },
        { title: 'Risk Scoring', status: 'upcoming', date: 'Planned' },
        { title: 'Action Simulation', status: 'upcoming', date: 'Planned' },
        { title: 'On-Premise Option', status: 'upcoming', date: 'Planned' },
        { title: 'Marketplace', status: 'upcoming', date: 'Planned' },
      ],
      deliverables: [
        'Full compliance automation',
        'Predictive risk analysis',
        'Policy testing sandbox',
        'Partner ecosystem',
      ],
    },
  ],
};

export const metricsData = {
  title: 'Success Metrics',
  subtitle: 'How we measure progress and validate the product.',
  categories: [
    {
      name: 'Product Metrics',
      metrics: [
        { name: 'Agents Protected', target: '1,000+', current: '127', unit: '' },
        { name: 'Actions Logged', target: '10M/mo', current: '2.3M', unit: '' },
        { name: 'Policy Evaluations', target: '<10ms p99', current: '8ms', unit: '' },
        { name: 'Uptime', target: '99.9%', current: '99.95%', unit: '' },
      ],
    },
    {
      name: 'Business Metrics',
      metrics: [
        { name: 'Design Partners', target: '10', current: '4', unit: '' },
        { name: 'Pilots', target: '25', current: '2', unit: '' },
        { name: 'MRR', target: '$50K', current: '$0', unit: '' },
        { name: 'NPS', target: '50+', current: 'N/A', unit: '' },
      ],
    },
    {
      name: 'Engagement Metrics',
      metrics: [
        { name: 'Daily Active Orgs', target: '80%', current: '67%', unit: '' },
        { name: 'Actions/Agent/Day', target: '500+', current: '340', unit: '' },
        { name: 'Policy Hit Rate', target: '15%', current: '12%', unit: '' },
        { name: 'HITL Response Time', target: '<5min', current: '3.2min', unit: '' },
      ],
    },
  ],
};

export const ctaData = {
  title: 'Ready to Secure Your AI Agents?',
  subtitle: 'Join our design partner program and shape the future of AI agent security.',
  options: [
    {
      title: 'Design Partner',
      description: 'Work directly with our team to shape the product',
      benefits: [
        'Direct access to founders',
        'Custom feature prioritization',
        'Discounted pricing at GA',
        'Early access to all features',
      ],
      cta: 'Apply Now',
      highlighted: true,
    },
    {
      title: 'Waitlist',
      description: 'Get notified when we launch publicly',
      benefits: [
        'Priority access',
        'Product updates',
        'Beta invitations',
      ],
      cta: 'Join Waitlist',
      highlighted: false,
    },
    {
      title: 'Learn More',
      description: 'Schedule a demo with our team',
      benefits: [
        'Live product demo',
        'Technical deep-dive',
        'Q&A session',
      ],
      cta: 'Book Demo',
      highlighted: false,
    },
  ],
  contact: {
    email: 'founders@aasp.security',
    calendly: 'https://calendly.com/aasp-demo',
  },
};

// Aliases and additional exports for components

// MVP Scope export
export const mvpScope = {
  title: 'MVP Scope Definition',
  subtitle: 'Focused execution with clear boundaries. Ship fast, learn faster.',
  timeline: 'Q1 2025',
  inScope: [
    { feature: 'Agent Registry', description: 'Central registry for all AI agents with credential management' },
    { feature: 'Action Logging', description: 'Comprehensive audit trail with full context preservation' },
    { feature: 'MCP Proxy', description: 'Transparent interception layer for MCP protocol' },
    { feature: 'Basic Policy Engine', description: 'Allow/block rules with rate limiting' },
    { feature: 'Slack Approvals', description: 'Human-in-the-loop via Slack integration' },
    { feature: 'Query Dashboard', description: 'Search and filter action logs' },
  ],
  outOfScope: [
    { feature: 'Advanced Anomaly Detection', reason: 'Phase 2 - requires baseline data' },
    { feature: 'Multi-Framework SDKs', reason: 'Phase 2 - focus on MCP first' },
    { feature: 'On-Premise Deployment', reason: 'Phase 3 - enterprise feature' },
    { feature: 'AI Policy Suggestions', reason: 'Future - maintain explicit-only approach' },
  ],
  successCriteria: [
    { metric: '10 Design Partners', target: '10', description: 'Engaged beta customers' },
    { metric: '<10ms Latency', target: '<10ms', description: 'P99 policy evaluation' },
    { metric: '99.9% Uptime', target: '99.9%', description: 'Service availability' },
    { metric: '1M Actions/Day', target: '1M', description: 'Daily processing capacity' },
  ],
};

// Feature breakdown export
export const featureBreakdown = [
  {
    name: 'Action Interception',
    description: 'Capture and control agent actions in real-time',
    progress: 75,
    subFeatures: [
      { name: 'MCP Proxy Server', status: 'done' },
      { name: 'SDK Interceptors', status: 'done' },
      { name: 'Request Modification', status: 'in-progress' },
      { name: 'Response Filtering', status: 'pending' },
    ],
  },
  {
    name: 'Policy Engine',
    description: 'Rule-based security policy evaluation',
    progress: 45,
    subFeatures: [
      { name: 'Rule DSL Parser', status: 'done' },
      { name: 'Allow/Block Lists', status: 'in-progress' },
      { name: 'Rate Limiting', status: 'in-progress' },
      { name: 'Conditional Logic', status: 'pending' },
    ],
  },
  {
    name: 'Audit Dashboard',
    description: 'Visualize and query agent activity',
    progress: 60,
    subFeatures: [
      { name: 'Event Timeline', status: 'done' },
      { name: 'Search & Filter', status: 'done' },
      { name: 'Export Functions', status: 'in-progress' },
      { name: 'Replay Engine', status: 'pending' },
    ],
  },
  {
    name: 'HITL Workflows',
    description: 'Human approval for sensitive actions',
    progress: 30,
    subFeatures: [
      { name: 'Slack Integration', status: 'in-progress' },
      { name: 'Approval Queue', status: 'pending' },
      { name: 'Escalation Rules', status: 'pending' },
      { name: 'Delegation', status: 'pending' },
    ],
  },
];

// Roadmap export
export const roadmap = [
  {
    id: 'foundation',
    name: 'Phase 1: Foundation',
    timeline: 'Q1 2025',
    status: 'in-progress',
    description: 'Core platform infrastructure and MCP support',
    goals: [
      { title: 'Agent Registry', description: 'Central identity management', status: 'done' },
      { title: 'Action Logging', description: 'Immutable audit trail', status: 'done' },
      { title: 'MCP Proxy', description: 'Protocol interception', status: 'in-progress' },
      { title: 'Basic Policies', description: 'Allow/block rules', status: 'pending' },
    ],
    milestone: '10 Design Partners Live',
  },
  {
    id: 'expansion',
    name: 'Phase 2: Expansion',
    timeline: 'Q2 2025',
    status: 'upcoming',
    description: 'Multi-framework support and advanced features',
    goals: [
      { title: 'LangChain SDK', description: 'Framework integration', status: 'pending' },
      { title: 'Advanced Policies', description: 'Complex rule logic', status: 'pending' },
      { title: 'Anomaly Detection', description: 'Baseline monitoring', status: 'pending' },
      { title: 'SIEM Export', description: 'Enterprise integrations', status: 'pending' },
    ],
    milestone: 'First Paying Customers',
  },
  {
    id: 'platform',
    name: 'Phase 3: Platform',
    timeline: 'Q3-Q4 2025',
    status: 'upcoming',
    description: 'Enterprise features and ecosystem',
    goals: [
      { title: 'Compliance Suite', description: 'Automated reporting', status: 'pending' },
      { title: 'Risk Scoring', description: 'Predictive analytics', status: 'pending' },
      { title: 'On-Premise', description: 'Self-hosted option', status: 'pending' },
      { title: 'Marketplace', description: 'Policy templates', status: 'pending' },
    ],
    milestone: '$1M ARR',
  },
];

// Success metrics export
export const successMetrics = {
  kpis: [
    { name: 'Protected Agents', target: '1,000+', trend: '+24%', description: 'Active agents under management' },
    { name: 'Actions/Month', target: '10M+', trend: '+45%', description: 'Monthly action volume' },
    { name: 'Policy Latency', target: '<5ms', trend: '-12%', description: 'P99 evaluation time' },
    { name: 'Customer NPS', target: '70+', trend: '+8', description: 'Net promoter score' },
  ],
  targets: [
    { metric: 'Design Partners', target: '10', current: 65, status: 'On Track' },
    { metric: 'Beta Users', target: '50', current: 42, status: 'On Track' },
    { metric: 'Actions Logged', target: '10M', current: 78, status: 'Ahead' },
    { metric: 'Uptime SLA', target: '99.9%', current: 100, status: 'Achieved' },
  ],
};

// CTA options export
export const ctaOptions = [
  {
    title: 'Early Access',
    description: 'Be among the first to secure your AI agents',
    action: 'Request Access',
    primary: true,
  },
  {
    title: 'Schedule Demo',
    description: 'See the platform in action with your use case',
    action: 'Book Time',
    primary: false,
  },
  {
    title: 'Documentation',
    description: 'Explore our technical docs and API reference',
    action: 'Read Docs',
    primary: false,
  },
  {
    title: 'Contact Sales',
    description: 'Talk to our team about enterprise needs',
    action: 'Get in Touch',
    primary: false,
  },
];
