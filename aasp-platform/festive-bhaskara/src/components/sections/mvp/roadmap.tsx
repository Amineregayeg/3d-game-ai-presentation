'use client';

import { motion } from 'motion/react';
import { Flag, CheckCircle2, Clock, Circle, ArrowRight } from 'lucide-react';
import { roadmap } from '@/lib/data/mvp-data';

const phaseColors = {
  foundation: { bg: 'from-blue-500/20 to-blue-600/20', border: 'border-blue-500/30', text: 'text-blue-400' },
  expansion: { bg: 'from-purple-500/20 to-purple-600/20', border: 'border-purple-500/30', text: 'text-purple-400' },
  platform: { bg: 'from-green-500/20 to-green-600/20', border: 'border-green-500/30', text: 'text-green-400' },
};

export function RoadmapSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          Product Roadmap
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          From MVP to market-leading platform - our 18-month journey
        </motion.p>
      </div>

      {/* Timeline View */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="relative"
      >
        {/* Timeline Line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-500 via-purple-500 to-green-500 hidden md:block" />

        <div className="space-y-8">
          {roadmap.map((phase, phaseIndex) => {
            const colors = phaseColors[phase.id as keyof typeof phaseColors] || phaseColors.foundation;

            return (
              <motion.div
                key={phase.id}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 * phaseIndex }}
                className="relative"
              >
                {/* Phase Marker */}
                <div className="absolute left-0 md:left-4 top-0 flex h-8 w-8 items-center justify-center rounded-full bg-background border-2 border-primary z-10">
                  <Flag className="h-4 w-4 text-primary" />
                </div>

                {/* Phase Content */}
                <div className="ml-12 md:ml-16">
                  <div className={`rounded-2xl border ${colors.border} bg-gradient-to-r ${colors.bg} p-6`}>
                    {/* Phase Header */}
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className={`text-xl font-semibold ${colors.text}`}>{phase.name}</h3>
                        <p className="text-sm text-muted-foreground">{phase.timeline}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        {phase.status === 'in-progress' && (
                          <span className="flex items-center gap-1 text-sm bg-yellow-500/20 text-yellow-400 px-3 py-1 rounded-full">
                            <Clock className="h-3 w-3" />
                            In Progress
                          </span>
                        )}
                        {phase.status === 'upcoming' && (
                          <span className="flex items-center gap-1 text-sm bg-gray-500/20 text-gray-400 px-3 py-1 rounded-full">
                            <Circle className="h-3 w-3" />
                            Upcoming
                          </span>
                        )}
                      </div>
                    </div>

                    <p className="text-muted-foreground mb-4">{phase.description}</p>

                    {/* Goals */}
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium text-foreground uppercase tracking-wider">Key Deliverables</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {phase.goals.map((goal) => (
                          <div
                            key={goal.title}
                            className={`flex items-start gap-3 p-3 rounded-lg bg-background/50 ${
                              goal.status === 'done'
                                ? 'border border-green-500/30'
                                : goal.status === 'in-progress'
                                ? 'border border-yellow-500/30'
                                : 'border border-border/30'
                            }`}
                          >
                            {goal.status === 'done' ? (
                              <CheckCircle2 className="h-5 w-5 text-green-400 shrink-0" />
                            ) : goal.status === 'in-progress' ? (
                              <Clock className="h-5 w-5 text-yellow-400 shrink-0" />
                            ) : (
                              <Circle className="h-5 w-5 text-muted-foreground shrink-0" />
                            )}
                            <div>
                              <p className="text-sm font-medium text-foreground">{goal.title}</p>
                              <p className="text-xs text-muted-foreground">{goal.description}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Milestone */}
                    <div className="mt-4 pt-4 border-t border-border/30">
                      <div className="flex items-center gap-2">
                        <Flag className={`h-4 w-4 ${colors.text}`} />
                        <span className="text-sm font-medium text-foreground">Milestone:</span>
                        <span className="text-sm text-muted-foreground">{phase.milestone}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Quarterly Breakdown */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">Quarterly Objectives</h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { quarter: 'Q1 2025', focus: 'Foundation', items: ['Core SDK', 'Basic policies', 'Audit logging', 'Beta launch'] },
            { quarter: 'Q2 2025', focus: 'Growth', items: ['GA release', 'Advanced policies', 'HITL workflows', '50 customers'] },
            { quarter: 'Q3 2025', focus: 'Scale', items: ['Enterprise features', 'Multi-region', 'Compliance certs', '200 customers'] },
            { quarter: 'Q4 2025', focus: 'Expand', items: ['Marketplace', 'Partner program', 'AI features', '500 customers'] },
          ].map((q, index) => (
            <motion.div
              key={q.quarter}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 * index }}
              className="relative"
            >
              {index < 3 && (
                <div className="absolute top-1/2 -right-3 hidden md:block">
                  <ArrowRight className="h-6 w-6 text-muted-foreground" />
                </div>
              )}
              <div className="rounded-xl border border-border/50 bg-secondary/20 p-4 h-full">
                <div className="text-center mb-4">
                  <p className="text-lg font-bold text-primary">{q.quarter}</p>
                  <p className="text-sm text-muted-foreground">{q.focus}</p>
                </div>
                <ul className="space-y-2">
                  {q.items.map((item) => (
                    <li key={item} className="flex items-center gap-2 text-sm text-muted-foreground">
                      <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Long-term Vision */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        className="relative overflow-hidden rounded-2xl border border-primary/30 bg-gradient-to-br from-primary/10 via-transparent to-accent/10 p-8 text-center"
      >
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-primary/5 via-transparent to-transparent" />
        <div className="relative">
          <p className="text-sm font-medium text-primary uppercase tracking-wider mb-2">
            2026 Vision
          </p>
          <p className="text-2xl font-bold text-foreground mb-4">
            The Standard for AI Agent Security
          </p>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            By 2026, AASP will be the de facto security layer for AI agent deployments,
            protecting billions of agent actions daily across thousands of enterprises worldwide.
          </p>
        </div>
      </motion.div>
    </div>
  );
}
