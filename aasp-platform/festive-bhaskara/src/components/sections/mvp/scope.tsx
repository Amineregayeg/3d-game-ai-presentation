'use client';

import { motion } from 'motion/react';
import { CheckCircle2, XCircle, Target, Clock, Users, Zap } from 'lucide-react';
import { mvpScope } from '@/lib/data/mvp-data';

export function MVPScopeSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {mvpScope.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {mvpScope.subtitle}
        </motion.p>
      </div>

      {/* MVP Overview Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-4 gap-6"
      >
        {[
          { icon: Target, label: 'Target Launch', value: mvpScope.timeline, color: 'text-blue-400' },
          { icon: Clock, label: 'Development', value: '12 Weeks', color: 'text-green-400' },
          { icon: Users, label: 'Beta Users', value: '10 Companies', color: 'text-purple-400' },
          { icon: Zap, label: 'Core Features', value: `${mvpScope.inScope.length} Features`, color: 'text-yellow-400' },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index }}
            className="relative overflow-hidden rounded-xl border border-border/50 bg-card/30 p-5 backdrop-blur-sm text-center"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent" />
            <div className="relative">
              <stat.icon className={`h-6 w-6 ${stat.color} mx-auto mb-2`} />
              <p className="text-2xl font-bold text-foreground mb-1">{stat.value}</p>
              <p className="text-sm text-muted-foreground">{stat.label}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* In Scope / Out of Scope */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* In Scope */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-green-500/30 bg-green-500/5 p-6"
        >
          <div className="flex items-center gap-2 mb-6">
            <CheckCircle2 className="h-6 w-6 text-green-400" />
            <h3 className="text-xl font-semibold text-foreground">MVP Scope (In)</h3>
          </div>

          <div className="space-y-3">
            {mvpScope.inScope.map((item, index) => (
              <motion.div
                key={item.feature}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.05 * index }}
                className="flex items-start gap-3 p-3 rounded-lg bg-green-500/10 border border-green-500/20"
              >
                <CheckCircle2 className="h-5 w-5 text-green-400 shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">{item.feature}</p>
                  <p className="text-sm text-muted-foreground">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Out of Scope */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-red-500/30 bg-red-500/5 p-6"
        >
          <div className="flex items-center gap-2 mb-6">
            <XCircle className="h-6 w-6 text-red-400" />
            <h3 className="text-xl font-semibold text-foreground">Post-MVP (Out)</h3>
          </div>

          <div className="space-y-3">
            {mvpScope.outOfScope.map((item, index) => (
              <motion.div
                key={item.feature}
                initial={{ opacity: 0, x: 10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.05 * index }}
                className="flex items-start gap-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20"
              >
                <XCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">{item.feature}</p>
                  <p className="text-sm text-muted-foreground">{item.reason}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Success Criteria */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">MVP Success Criteria</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {mvpScope.successCriteria.map((criteria, index) => (
            <motion.div
              key={criteria.metric}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 * index }}
              className="text-center p-4 rounded-xl bg-secondary/30"
            >
              <p className="text-3xl font-bold text-primary mb-2">{criteria.target}</p>
              <p className="text-sm font-medium text-foreground">{criteria.metric}</p>
              <p className="text-xs text-muted-foreground mt-1">{criteria.description}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Technical Requirements */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-primary/30 bg-primary/5 p-8"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">Technical Requirements</h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              category: 'Performance',
              items: ['< 10ms latency P95', '99.9% uptime SLA', '1000 req/s baseline'],
            },
            {
              category: 'Security',
              items: ['SOC 2 Type I audit', 'mTLS encryption', 'Zero-trust architecture'],
            },
            {
              category: 'Scalability',
              items: ['Multi-tenant ready', 'Horizontal scaling', 'Region failover'],
            },
          ].map((section) => (
            <div key={section.category} className="space-y-3">
              <h4 className="font-medium text-primary uppercase tracking-wider text-sm">
                {section.category}
              </h4>
              <ul className="space-y-2">
                {section.items.map((item) => (
                  <li key={item} className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
