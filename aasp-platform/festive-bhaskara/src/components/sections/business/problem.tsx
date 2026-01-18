'use client';

import { motion } from 'motion/react';
import { AlertTriangle, Globe, Eye, Lock, CheckCircle2, XCircle } from 'lucide-react';
import { problemData } from '@/lib/data/business-data';
import { FeatureCard } from '@/components/ui/feature-card';

const iconMap = {
  AlertTriangle,
  Globe,
  Eye,
  Lock,
};

export function ProblemSection() {
  return (
    <div className="space-y-16">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {problemData.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {problemData.subtitle}
        </motion.p>
      </div>

      {/* Problem Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {problemData.problems.map((problem, index) => {
          const Icon = iconMap[problem.icon.name as keyof typeof iconMap] || AlertTriangle;
          return (
            <FeatureCard
              key={problem.title}
              title={problem.title}
              description={problem.description}
              icon={<Icon className="h-6 w-6" />}
              index={index}
              gradient={problem.severity === 'critical'}
            />
          );
        })}
      </div>

      {/* Comparison Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="relative overflow-hidden rounded-2xl border border-border/50 bg-card/30 backdrop-blur-sm"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-red-500/5 via-transparent to-green-500/5" />

        <div className="relative p-8">
          <h3 className="text-xl font-semibold text-center mb-8">
            Classic Security vs. Agentic AI Reality
          </h3>

          <div className="grid grid-cols-2 gap-8">
            {/* Traditional Security */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-4">
                <span className="text-sm font-medium uppercase tracking-wider">Traditional Security Assumes</span>
              </div>
              {problemData.comparison.traditional.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center gap-3 rounded-lg bg-secondary/30 px-4 py-3"
                >
                  <CheckCircle2 className="h-5 w-5 text-muted-foreground" />
                  <span className="text-foreground">{item.label}</span>
                </div>
              ))}
            </div>

            {/* Agentic AI Reality */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-red-400 mb-4">
                <span className="text-sm font-medium uppercase tracking-wider">Agentic AI Breaks</span>
              </div>
              {problemData.comparison.agenticAI.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center gap-3 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3"
                >
                  <XCircle className="h-5 w-5 text-red-400" />
                  <span className="text-foreground">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Key Insight */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        className="relative overflow-hidden rounded-2xl border border-primary/30 bg-primary/5 p-8 text-center"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-accent/10" />
        <div className="relative">
          <p className="text-xl font-medium text-foreground mb-2">
            The Real Wedge
          </p>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            AI agents are becoming autonomous actors that can access tools, data, and systems—but
            our security infrastructure is still built for humans and static services.
            <span className="text-primary font-semibold"> This mismatch is permanent and growing.</span>
          </p>
        </div>
      </motion.div>
    </div>
  );
}
