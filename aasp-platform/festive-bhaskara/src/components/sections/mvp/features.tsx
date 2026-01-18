'use client';

import { motion } from 'motion/react';
import { Shield, Eye, Lock, Workflow, CheckCircle2, Circle, Clock } from 'lucide-react';
import { featureBreakdown } from '@/lib/data/mvp-data';

const featureIcons = {
  'Action Interception': Shield,
  'Policy Engine': Lock,
  'Audit Dashboard': Eye,
  'HITL Workflows': Workflow,
};

export function FeaturesSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          Feature Breakdown
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          Detailed breakdown of MVP features and their implementation progress
        </motion.p>
      </div>

      {/* Feature Cards */}
      <div className="space-y-8">
        {featureBreakdown.map((feature, index) => {
          const Icon = featureIcons[feature.name as keyof typeof featureIcons] || Shield;

          return (
            <motion.div
              key={feature.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 * index }}
              className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
            >
              {/* Feature Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-foreground">{feature.name}</h3>
                    <p className="text-sm text-muted-foreground">{feature.description}</p>
                  </div>
                </div>

                {/* Progress */}
                <div className="text-right">
                  <p className="text-2xl font-bold text-primary">{feature.progress}%</p>
                  <p className="text-xs text-muted-foreground">Complete</p>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="h-2 bg-secondary/50 rounded-full overflow-hidden mb-6">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: `${feature.progress}%` }}
                  viewport={{ once: true }}
                  transition={{ duration: 1, delay: 0.2 }}
                  className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                />
              </div>

              {/* Sub-features */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {feature.subFeatures.map((subFeature) => (
                  <div
                    key={subFeature.name}
                    className={`flex items-start gap-3 p-3 rounded-lg ${
                      subFeature.status === 'done'
                        ? 'bg-green-500/10 border border-green-500/20'
                        : subFeature.status === 'in-progress'
                        ? 'bg-yellow-500/10 border border-yellow-500/20'
                        : 'bg-secondary/30 border border-border/30'
                    }`}
                  >
                    {subFeature.status === 'done' ? (
                      <CheckCircle2 className="h-5 w-5 text-green-400 shrink-0" />
                    ) : subFeature.status === 'in-progress' ? (
                      <Clock className="h-5 w-5 text-yellow-400 shrink-0" />
                    ) : (
                      <Circle className="h-5 w-5 text-muted-foreground shrink-0" />
                    )}
                    <div>
                      <p className="text-sm font-medium text-foreground">{subFeature.name}</p>
                      <p className="text-xs text-muted-foreground capitalize">{subFeature.status.replace('-', ' ')}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Implementation Priority */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">Implementation Priority Matrix</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* High Priority */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-red-400" />
              <h4 className="font-medium text-foreground">P0 - Critical Path</h4>
            </div>
            <div className="space-y-2">
              {[
                'SDK integration layer',
                'Basic policy evaluation',
                'Action logging infrastructure',
                'Authentication flow',
              ].map((item) => (
                <div key={item} className="flex items-center gap-2 p-2 rounded-lg bg-red-500/10">
                  <span className="text-sm text-foreground">{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Medium Priority */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-yellow-400" />
              <h4 className="font-medium text-foreground">P1 - Important</h4>
            </div>
            <div className="space-y-2">
              {[
                'Dashboard UI components',
                'Policy builder interface',
                'Webhook notifications',
                'Basic analytics',
              ].map((item) => (
                <div key={item} className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10">
                  <span className="text-sm text-foreground">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Tech Debt & Risks */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Known Tech Debt */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-yellow-500/30 bg-yellow-500/5 p-6"
        >
          <h3 className="text-lg font-semibold mb-4 text-foreground">Planned Tech Debt (MVP)</h3>
          <ul className="space-y-3">
            {[
              { debt: 'Simplified caching layer', plan: 'Replace with Redis cluster post-MVP' },
              { debt: 'Single-region deployment', plan: 'Multi-region in v1.1' },
              { debt: 'Basic retry logic', plan: 'Circuit breaker pattern post-MVP' },
              { debt: 'Manual scaling triggers', plan: 'Auto-scaling in v1.1' },
            ].map((item) => (
              <li key={item.debt} className="p-3 rounded-lg bg-yellow-500/10">
                <p className="text-sm font-medium text-foreground">{item.debt}</p>
                <p className="text-xs text-muted-foreground">{item.plan}</p>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Risks & Mitigations */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-red-500/30 bg-red-500/5 p-6"
        >
          <h3 className="text-lg font-semibold mb-4 text-foreground">Risks & Mitigations</h3>
          <ul className="space-y-3">
            {[
              { risk: 'SDK adoption friction', mitigation: 'Simple 3-line integration' },
              { risk: 'Latency impact', mitigation: 'Async-first architecture' },
              { risk: 'Policy complexity', mitigation: 'Template-based quick starts' },
              { risk: 'Integration gaps', mitigation: 'Prioritize top 5 platforms' },
            ].map((item) => (
              <li key={item.risk} className="p-3 rounded-lg bg-red-500/10">
                <p className="text-sm font-medium text-foreground">{item.risk}</p>
                <p className="text-xs text-green-400">{item.mitigation}</p>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>
    </div>
  );
}
