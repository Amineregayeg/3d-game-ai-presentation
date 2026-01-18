'use client';

import { motion } from 'motion/react';
import { Code2, Terminal, Shield, Zap, Database, Eye } from 'lucide-react';
import { coreCapabilities, techStack } from '@/lib/data/technical-data';

const capabilityIcons = {
  'Action Interception': Shield,
  'Policy Engine': Code2,
  'Audit Trail': Database,
  'Human-in-the-Loop': Eye,
};

export function CoreLayersSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          Core Capabilities
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          Deep dive into the technical capabilities that power the AASP platform
        </motion.p>
      </div>

      {/* Core Capabilities with Code Examples */}
      <div className="space-y-8">
        {coreCapabilities.map((capability, index) => {
          const Icon = capabilityIcons[capability.name as keyof typeof capabilityIcons] || Shield;
          const isEven = index % 2 === 0;

          return (
            <motion.div
              key={capability.name}
              initial={{ opacity: 0, x: isEven ? -20 : 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className={`grid grid-cols-1 lg:grid-cols-2 gap-8 ${isEven ? '' : 'lg:direction-rtl'}`}
            >
              {/* Description */}
              <div className={`flex flex-col justify-center ${isEven ? '' : 'lg:order-2'}`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-2xl font-semibold text-foreground">{capability.name}</h3>
                </div>
                <p className="text-muted-foreground mb-4">{capability.description}</p>

                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Key Features</p>
                  <ul className="space-y-2">
                    {capability.features.map((feature) => (
                      <li key={feature} className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Zap className="h-4 w-4 text-primary shrink-0" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Code Example */}
              <div className={`${isEven ? '' : 'lg:order-1'}`}>
                <div className="rounded-xl border border-border/50 bg-[#0d1117] overflow-hidden">
                  <div className="flex items-center gap-2 px-4 py-2 bg-[#161b22] border-b border-border/30">
                    <div className="flex gap-1.5">
                      <div className="h-3 w-3 rounded-full bg-red-500/80" />
                      <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
                      <div className="h-3 w-3 rounded-full bg-green-500/80" />
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <Terminal className="h-4 w-4 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground font-mono">{capability.name.toLowerCase().replace(/ /g, '-')}.ts</span>
                    </div>
                  </div>
                  <pre className="p-4 overflow-x-auto">
                    <code className="text-sm font-mono text-gray-300">
                      {capability.codeExample}
                    </code>
                  </pre>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Tech Stack */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-8 text-center">Technology Stack</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {techStack.map((category) => (
            <div key={category.category} className="space-y-4">
              <h4 className="text-sm font-medium text-primary uppercase tracking-wider">
                {category.category}
              </h4>
              <div className="space-y-2">
                {category.technologies.map((tech) => (
                  <div
                    key={tech.name}
                    className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Code2 className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-foreground">{tech.name}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="grid grid-cols-1 md:grid-cols-4 gap-6"
      >
        {[
          { metric: 'Latency', value: '<5ms', desc: 'P99 processing time' },
          { metric: 'Throughput', value: '100K+', desc: 'Requests per second' },
          { metric: 'Uptime', value: '99.99%', desc: 'SLA guarantee' },
          { metric: 'Scale', value: '∞', desc: 'Horizontal scaling' },
        ].map((item, index) => (
          <motion.div
            key={item.metric}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 * index }}
            className="relative overflow-hidden rounded-xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm text-center"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent" />
            <div className="relative">
              <p className="text-3xl font-bold text-primary mb-1">{item.value}</p>
              <p className="text-sm font-medium text-foreground">{item.metric}</p>
              <p className="text-xs text-muted-foreground">{item.desc}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Security Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-primary/30 bg-primary/5 p-8"
      >
        <div className="flex items-center justify-center gap-3 mb-6">
          <Shield className="h-6 w-6 text-primary" />
          <h3 className="text-xl font-semibold text-foreground">Security-First Architecture</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              title: 'Zero Trust',
              items: ['mTLS everywhere', 'Certificate pinning', 'No implicit trust'],
            },
            {
              title: 'Data Protection',
              items: ['AES-256 encryption', 'Field-level encryption', 'Secure key management'],
            },
            {
              title: 'Compliance',
              items: ['SOC 2 Type II', 'GDPR compliant', 'ISO 27001 ready'],
            },
          ].map((section) => (
            <div key={section.title} className="space-y-3">
              <h4 className="font-medium text-foreground">{section.title}</h4>
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
