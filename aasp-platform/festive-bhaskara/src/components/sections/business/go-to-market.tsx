'use client';

import { motion } from 'motion/react';
import { Target, Users, Megaphone, TrendingUp, CheckCircle2, ArrowRight } from 'lucide-react';
import { goToMarketData } from '@/lib/data/business-data';

const phaseIcons = [Target, Users, Megaphone, TrendingUp];

export function GoToMarketSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {goToMarketData.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {goToMarketData.subtitle}
        </motion.p>
      </div>

      {/* GTM Phases Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="relative"
      >
        {/* Connection Line */}
        <div className="absolute top-24 left-0 right-0 h-0.5 bg-gradient-to-r from-primary via-accent to-primary hidden lg:block" />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {goToMarketData.phases.map((phase, index) => {
            const Icon = phaseIcons[index] || Target;
            return (
              <motion.div
                key={phase.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                className="relative"
              >
                {/* Phase Number */}
                <div className="flex items-center justify-center mb-4">
                  <div className="relative z-10 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent text-white font-bold">
                    {index + 1}
                  </div>
                </div>

                <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm h-full">
                  <div className="flex items-center gap-2 mb-3">
                    <Icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{phase.name}</h3>
                  </div>
                  <p className="text-xs text-muted-foreground mb-4">{phase.timeline}</p>
                  <p className="text-sm text-muted-foreground mb-4">{phase.focus}</p>

                  <div className="space-y-2">
                    {phase.activities.map((activity) => (
                      <div key={activity} className="flex items-start gap-2">
                        <CheckCircle2 className="h-4 w-4 text-green-400 shrink-0 mt-0.5" />
                        <span className="text-xs text-muted-foreground">{activity}</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 pt-4 border-t border-border/30">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Target</span>
                      <span className="text-sm font-semibold text-primary">{phase.target}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Channel Strategy */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">Channel Strategy</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {goToMarketData.channels.map((channel, index) => (
            <div
              key={channel.name}
              className="relative overflow-hidden rounded-xl border border-border/30 bg-secondary/20 p-5"
            >
              <div className="absolute top-0 right-0 w-20 h-20 bg-primary/5 rounded-full blur-2xl" />
              <div className="relative">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-foreground">{channel.name}</h4>
                  <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded-full">
                    {channel.percentage}%
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-4">{channel.description}</p>

                {/* Progress bar */}
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${channel.percentage}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1, delay: 0.2 * index }}
                    className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Partnership Strategy */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Strategic Partners */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-xl font-semibold mb-6">Strategic Partnerships</h3>
          <div className="space-y-4">
            {goToMarketData.partnerships.map((partner) => (
              <div
                key={partner.type}
                className="flex items-start gap-4 p-4 rounded-lg bg-secondary/20 hover:bg-secondary/30 transition-colors"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/20 shrink-0">
                  <Users className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h4 className="font-medium text-foreground mb-1">{partner.type}</h4>
                  <p className="text-sm text-muted-foreground">{partner.value}</p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {partner.examples.map((ex) => (
                      <span
                        key={ex}
                        className="text-xs bg-secondary px-2 py-1 rounded text-muted-foreground"
                      >
                        {ex}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Customer Success Metrics */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-xl font-semibold mb-6">Success Metrics</h3>
          <div className="space-y-4">
            {goToMarketData.successMetrics.map((metric) => (
              <div
                key={metric.metric}
                className="flex items-center justify-between p-4 rounded-lg bg-secondary/20"
              >
                <div>
                  <p className="font-medium text-foreground">{metric.metric}</p>
                  <p className="text-xs text-muted-foreground">{metric.description}</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-primary">{metric.target}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        className="relative overflow-hidden rounded-2xl border border-primary/30 bg-gradient-to-br from-primary/10 via-transparent to-accent/10 p-8 text-center"
      >
        <div className="relative">
          <p className="text-2xl font-bold text-foreground mb-4">
            Ready to Secure AI Agent Actions?
          </p>
          <p className="text-muted-foreground max-w-xl mx-auto mb-6">
            Join leading enterprises in building trust for the agentic AI era.
            Get early access to the platform that&apos;s defining the future of AI security.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="group flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90 hover:shadow-lg hover:shadow-primary/20">
              Request Demo
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </button>
            <button className="flex items-center gap-2 rounded-xl border border-border bg-secondary/50 px-6 py-3 text-sm font-semibold text-foreground transition-all hover:bg-secondary">
              View Documentation
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
