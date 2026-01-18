import {
  Shield,
  Lock,
  Eye,
  FileSearch,
  Zap,
  Globe,
  Building2,
  TrendingUp,
  Users,
  DollarSign,
  Target,
  Award,
  CheckCircle2,
  AlertTriangle,
  XCircle,
} from 'lucide-react';

export const heroData = {
  tagline: 'AI Agent Security Platform',
  headline: 'Secure What AI Agents Do',
  subheadline: 'Not Just What They Can Access',
  description:
    'The first real-time control plane for AI agent actions. Monitor, audit, and enforce policies across all your AI agents—regardless of framework.',
  stats: [
    { value: '$15.8B', label: 'AI Security Market by 2030' },
    { value: '89%', label: 'Enterprises Planning AI Agents' },
    { value: '340%', label: 'YoY Growth in Agent Deployments' },
  ],
};

export const problemData = {
  title: 'The Security Gap in AI Agents',
  subtitle: 'Classic security infrastructure was built for humans and static services. AI agents break every assumption.',
  problems: [
    {
      icon: AlertTriangle,
      title: 'Autonomous Decision Making',
      description: 'AI agents decide what to do, when to do it, and how to chain actions—without human oversight.',
      severity: 'critical',
    },
    {
      icon: Globe,
      title: 'Cross-System Operations',
      description: 'Agents access multiple systems, APIs, and data sources in unpredictable sequences.',
      severity: 'critical',
    },
    {
      icon: Eye,
      title: 'No Audit Trail',
      description: 'When an agent acts, there\'s no standardized way to know what it did, why, or what data it touched.',
      severity: 'high',
    },
    {
      icon: Lock,
      title: 'Permission Creep',
      description: 'Agents impersonate users, use shared tokens, and accumulate permissions beyond their intended scope.',
      severity: 'high',
    },
  ],
  comparison: {
    traditional: [
      { label: 'Fixed APIs', status: 'assumed' },
      { label: 'Explicit Users', status: 'assumed' },
      { label: 'Predictable Flows', status: 'assumed' },
      { label: 'Predefined Scopes', status: 'assumed' },
    ],
    agenticAI: [
      { label: 'Dynamic Tool Selection', status: 'broken' },
      { label: 'Agent Identities', status: 'broken' },
      { label: 'Emergent Behavior', status: 'broken' },
      { label: 'Runtime Permissions', status: 'broken' },
    ],
  },
};

export const marketData = {
  title: 'Market Opportunity',
  subtitle: 'The AI security market is experiencing explosive growth as enterprises rush to deploy agents.',
  tam: {
    value: '$45.2B',
    label: 'Total Addressable Market',
    description: 'Global enterprise security software market (2028)',
  },
  sam: {
    value: '$15.8B',
    label: 'Serviceable Addressable Market',
    description: 'AI-specific security tools and platforms (2030)',
  },
  som: {
    value: '$890M',
    label: 'Serviceable Obtainable Market',
    description: 'Agent security and governance (Year 5)',
  },
  growth: [
    { year: '2024', tam: 28.5, sam: 4.2, som: 0.1 },
    { year: '2025', tam: 32.1, sam: 6.8, som: 0.25 },
    { year: '2026', tam: 36.4, sam: 9.5, som: 0.45 },
    { year: '2027', tam: 40.8, sam: 12.3, som: 0.65 },
    { year: '2028', tam: 45.2, sam: 15.8, som: 0.89 },
  ],
  drivers: [
    {
      title: 'Enterprise AI Adoption',
      value: '89%',
      description: 'of enterprises plan to deploy AI agents by 2026',
    },
    {
      title: 'Regulatory Pressure',
      value: '73%',
      description: 'increase in AI-related compliance requirements',
    },
    {
      title: 'Security Incidents',
      value: '340%',
      description: 'YoY increase in AI-related security incidents',
    },
    {
      title: 'Budget Allocation',
      value: '$2.1M',
      description: 'average enterprise AI security budget (2025)',
    },
  ],
  segments: [
    { name: 'Financial Services', percentage: 28, color: '#3B82F6' },
    { name: 'Healthcare', percentage: 22, color: '#10B981' },
    { name: 'Technology', percentage: 20, color: '#8B5CF6' },
    { name: 'Government', percentage: 15, color: '#F59E0B' },
    { name: 'Other', percentage: 15, color: '#6B7280' },
  ],
};

export const competitiveData = {
  title: 'Competitive Landscape',
  subtitle: 'We occupy a unique position at the intersection of AI observability and security enforcement.',
  competitors: [
    {
      name: 'Traditional IAM',
      examples: ['Okta', 'Azure AD', 'Ping'],
      strengths: ['Established', 'Enterprise trust', 'Identity expertise'],
      weaknesses: ['Human-centric', 'No agent awareness', 'Static permissions'],
      threat: 'medium',
    },
    {
      name: 'LLM Security',
      examples: ['Prompt Guard', 'Rebuff', 'Lakera'],
      strengths: ['AI-native', 'Prompt protection', 'Fast deployment'],
      weaknesses: ['Prompt-focused only', 'No action control', 'Limited scope'],
      threat: 'low',
    },
    {
      name: 'API Security',
      examples: ['Salt', 'Noname', '42Crunch'],
      strengths: ['Runtime protection', 'API expertise', 'Enterprise ready'],
      weaknesses: ['No AI context', 'Endpoint-focused', 'No agent identity'],
      threat: 'medium',
    },
    {
      name: 'Cloud Security',
      examples: ['Wiz', 'Orca', 'Lacework'],
      strengths: ['Cloud-native', 'Broad coverage', 'Strong funding'],
      weaknesses: ['Infrastructure focus', 'Not agent-aware', 'Reactive'],
      threat: 'medium',
    },
  ],
  positioning: {
    x: { label: 'AI-Native', min: 0, max: 100 },
    y: { label: 'Action Control', min: 0, max: 100 },
    players: [
      { name: 'AASP', x: 90, y: 95, size: 'large', highlight: true },
      { name: 'Traditional IAM', x: 20, y: 40, size: 'medium' },
      { name: 'LLM Security', x: 80, y: 30, size: 'small' },
      { name: 'API Security', x: 35, y: 70, size: 'medium' },
      { name: 'Cloud Security', x: 50, y: 55, size: 'large' },
    ],
  },
  advantages: [
    {
      icon: Zap,
      title: 'Real-time Enforcement',
      description: 'Block or modify actions before they execute, not after.',
    },
    {
      icon: Globe,
      title: 'Framework Agnostic',
      description: 'Works with any agent framework: MCP, LangChain, AutoGen, custom.',
    },
    {
      icon: FileSearch,
      title: 'Action-Level Audit',
      description: 'Full context on what agents do, why, and what data they touch.',
    },
    {
      icon: Shield,
      title: 'Zero Trust for Agents',
      description: 'First-class agent identities with scoped, auditable permissions.',
    },
  ],
};

export const businessModelData = {
  title: 'Business Model',
  subtitle: 'Usage-based pricing aligned with customer value and growth.',
  pricing: [
    {
      tier: 'Starter',
      price: '$499',
      period: '/month',
      description: 'For teams starting with AI agents',
      features: [
        'Up to 10 agents',
        '100K actions/month',
        'Basic policy engine',
        'Email support',
        '7-day log retention',
      ],
      highlighted: false,
    },
    {
      tier: 'Professional',
      price: '$2,499',
      period: '/month',
      description: 'For growing organizations',
      features: [
        'Up to 100 agents',
        '1M actions/month',
        'Advanced policies',
        'HITL workflows',
        '30-day log retention',
        'Priority support',
        'SSO integration',
      ],
      highlighted: true,
    },
    {
      tier: 'Enterprise',
      price: 'Custom',
      period: '',
      description: 'For large-scale deployments',
      features: [
        'Unlimited agents',
        'Unlimited actions',
        'Custom policies',
        'Dedicated support',
        'Custom retention',
        'SLA guarantees',
        'On-premise option',
        'Compliance reports',
      ],
      highlighted: false,
    },
  ],
  revenueProjections: [
    { year: 'Year 1', arr: 0.5, customers: 15 },
    { year: 'Year 2', arr: 2.8, customers: 85 },
    { year: 'Year 3', arr: 12.5, customers: 320 },
    { year: 'Year 4', arr: 35, customers: 850 },
    { year: 'Year 5', arr: 85, customers: 2100 },
  ],
  metrics: {
    targetACV: '$150K',
    targetLTV: '$750K',
    targetCAC: '$45K',
    targetNRR: '135%',
  },
  unitEconomics: [
    { metric: 'Average Contract Value', value: '$150K' },
    { metric: 'Customer Lifetime Value', value: '$750K' },
    { metric: 'Customer Acquisition Cost', value: '$45K' },
    { metric: 'LTV:CAC Ratio', value: '16.7x' },
    { metric: 'Net Revenue Retention', value: '135%' },
    { metric: 'Gross Margin', value: '85%' },
  ],
  funding: {
    amount: '$5M',
    use: 'Seed round to achieve product-market fit and first $1M ARR',
    breakdown: [
      { category: 'Engineering', percentage: 55 },
      { category: 'Go-to-Market', percentage: 25 },
      { category: 'Operations', percentage: 15 },
      { category: 'Reserve', percentage: 5 },
    ],
  },
};

export const teamData = {
  title: 'Leadership Team',
  subtitle: 'Experienced founders with deep expertise in security and AI systems.',
  members: [
    {
      name: 'Founder 1',
      role: 'CEO & Co-Founder',
      background: 'Ex-Google Security, Stanford CS',
      expertise: ['Security Architecture', 'Enterprise Sales', 'Product Strategy'],
    },
    {
      name: 'Founder 2',
      role: 'CTO & Co-Founder',
      background: 'Ex-OpenAI, MIT AI Lab',
      expertise: ['AI/ML Systems', 'Distributed Systems', 'Agent Frameworks'],
    },
  ],
  advisors: [
    { name: 'Security Advisor', role: 'Former CISO, Fortune 100' },
    { name: 'GTM Advisor', role: 'VP Sales, Leading Security Company' },
    { name: 'Technical Advisor', role: 'AI Research Lead, Major Lab' },
  ],
};

export const goToMarketData = {
  title: 'Go-to-Market Strategy',
  subtitle: 'Land with compliance pain, expand with platform value.',
  icp: {
    title: 'Ideal Customer Profile',
    characteristics: [
      { label: 'Company Size', value: '1,000-50,000 employees' },
      { label: 'Industry', value: 'Financial Services, Healthcare, Tech' },
      { label: 'AI Maturity', value: 'Piloting or deploying AI agents' },
      { label: 'Buying Center', value: 'CISO, VP Engineering, Head of AI' },
      { label: 'Budget', value: '$100K-$500K annual security spend' },
    ],
    painPoints: [
      'Board asking about AI governance',
      'Compliance audit on AI systems upcoming',
      'Recent AI-related incident or near-miss',
      'Scaling agent deployment with no visibility',
    ],
  },
  channels: [
    {
      name: 'Direct Sales',
      percentage: 50,
      description: 'Enterprise accounts with dedicated AEs',
    },
    {
      name: 'Partnerships',
      percentage: 30,
      description: 'SI partners, cloud marketplaces, agent frameworks',
    },
    {
      name: 'Product-Led',
      percentage: 20,
      description: 'Self-serve for mid-market, freemium trials',
    },
  ],
  milestones: [
    {
      phase: 'Phase 1',
      title: 'Design Partners',
      target: '5-10 design partners',
      timeline: 'Q1-Q2',
    },
    {
      phase: 'Phase 2',
      title: 'Early Adopters',
      target: '25 paying customers',
      timeline: 'Q3-Q4',
    },
    {
      phase: 'Phase 3',
      title: 'Scale',
      target: '100+ customers, Series A',
      timeline: 'Year 2',
    },
  ],
  phases: [
    {
      name: 'Discovery',
      timeline: 'Months 1-3',
      focus: 'Validate problem-solution fit with design partners',
      activities: [
        'Deep customer discovery interviews',
        'MVP co-development with partners',
        'Iterate on core value proposition',
      ],
      target: '10 Design Partners',
    },
    {
      name: 'Validation',
      timeline: 'Months 4-6',
      focus: 'Prove commercial viability and product-market fit',
      activities: [
        'Close first paid pilots',
        'Develop case studies',
        'Refine pricing model',
      ],
      target: '$100K Pipeline',
    },
    {
      name: 'Acceleration',
      timeline: 'Months 7-9',
      focus: 'Scale go-to-market engine',
      activities: [
        'Hire first sales team',
        'Launch partner program',
        'Content marketing push',
      ],
      target: '25 Customers',
    },
    {
      name: 'Scale',
      timeline: 'Months 10-12',
      focus: 'Establish market leadership position',
      activities: [
        'Expand to new segments',
        'International expansion prep',
        'Series A fundraise',
      ],
      target: '$1M ARR',
    },
  ],
  partnerships: [
    {
      type: 'System Integrators',
      value: 'Implementation and deployment partnerships',
      examples: ['Accenture', 'Deloitte', 'KPMG'],
    },
    {
      type: 'AI Platforms',
      value: 'Native integrations with agent frameworks',
      examples: ['Anthropic', 'OpenAI', 'LangChain'],
    },
    {
      type: 'Cloud Providers',
      value: 'Marketplace listings and co-sell',
      examples: ['AWS', 'Azure', 'GCP'],
    },
  ],
  successMetrics: [
    {
      metric: 'Design Partners',
      target: '10',
      description: 'Active beta customers',
    },
    {
      metric: 'Time to Value',
      target: '<1 day',
      description: 'Deployment to first action logged',
    },
    {
      metric: 'NPS Score',
      target: '70+',
      description: 'Customer satisfaction',
    },
    {
      metric: 'Expansion Rate',
      target: '150%',
      description: 'Net dollar retention',
    },
  ],
};
