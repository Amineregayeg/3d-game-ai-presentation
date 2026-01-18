'use client';

import { motion } from 'motion/react';
import { Server, Shield, Database, Activity, Users, Workflow, Cloud, Lock } from 'lucide-react';
import { architectureData } from '@/lib/data/technical-data';

const layerIcons = {
  'Ingestion Layer': Server,
  'Identity Layer': Users,
  'Policy Layer': Shield,
  'Audit Layer': Database,
  'Workflow Layer': Workflow,
};

export function ArchitectureSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {architectureData.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {architectureData.subtitle}
        </motion.p>
      </div>

      {/* Architecture Diagram */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-8 text-center">System Architecture Overview</h3>

        <div className="relative">
          {/* Top Row - AI Agents */}
          <div className="flex justify-center mb-8">
            <div className="flex items-center gap-4 px-6 py-4 rounded-xl bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border border-blue-500/30">
              <Cloud className="h-6 w-6 text-blue-400" />
              <div>
                <p className="font-semibold text-foreground">AI Agents</p>
                <p className="text-xs text-muted-foreground">Claude, GPT, Custom Agents</p>
              </div>
            </div>
          </div>

          {/* Connection Line */}
          <div className="flex justify-center mb-4">
            <div className="w-0.5 h-8 bg-gradient-to-b from-blue-500/50 to-primary/50" />
          </div>

          {/* AASP Core */}
          <div className="relative rounded-2xl border-2 border-primary/30 bg-primary/5 p-6 mb-8">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-background px-4">
              <span className="text-sm font-semibold text-primary">AASP Core</span>
            </div>

            {/* Layers Grid */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mt-4">
              {architectureData.layers.map((layer, index) => {
                const Icon = layerIcons[layer.name as keyof typeof layerIcons] || Shield;
                return (
                  <motion.div
                    key={layer.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index }}
                    className="relative group"
                  >
                    <div className="rounded-xl border border-border/50 bg-card/50 p-4 h-full transition-all group-hover:border-primary/50 group-hover:bg-primary/5">
                      <div className="flex flex-col items-center text-center">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary mb-3">
                          <Icon className="h-5 w-5 text-primary" />
                        </div>
                        <h4 className="text-sm font-semibold text-foreground mb-1">{layer.name}</h4>
                        <p className="text-xs text-muted-foreground line-clamp-2">{layer.description}</p>
                      </div>
                    </div>

                    {/* Layer number */}
                    <div className="absolute -top-2 -right-2 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                      {index + 1}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Connection Lines */}
          <div className="flex justify-around mb-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="w-0.5 h-8 bg-gradient-to-b from-primary/50 to-green-500/50" />
            ))}
          </div>

          {/* Bottom Row - External Systems */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-secondary/30 border border-border/30">
              <Database className="h-5 w-5 text-green-400" />
              <div>
                <p className="text-sm font-medium text-foreground">External APIs</p>
                <p className="text-xs text-muted-foreground">Databases, Services</p>
              </div>
            </div>
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-secondary/30 border border-border/30">
              <Lock className="h-5 w-5 text-yellow-400" />
              <div>
                <p className="text-sm font-medium text-foreground">Identity Providers</p>
                <p className="text-xs text-muted-foreground">OAuth, SAML, SSO</p>
              </div>
            </div>
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-secondary/30 border border-border/30">
              <Activity className="h-5 w-5 text-purple-400" />
              <div>
                <p className="text-sm font-medium text-foreground">SIEM/SOAR</p>
                <p className="text-xs text-muted-foreground">Security Tools</p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Layer Details */}
      <div className="space-y-6">
        <h3 className="text-2xl font-semibold text-center">Platform Layers</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {architectureData.layers.map((layer, index) => {
            const Icon = layerIcons[layer.name as keyof typeof layerIcons] || Shield;
            return (
              <motion.div
                key={layer.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 * index }}
                className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-accent/20">
                    <Icon className="h-5 w-5 text-primary" />
                  </div>
                  <h4 className="font-semibold text-foreground">{layer.name}</h4>
                </div>

                <p className="text-sm text-muted-foreground mb-4">{layer.description}</p>

                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Components</p>
                  <div className="flex flex-wrap gap-2">
                    {layer.components.map((component) => (
                      <span
                        key={component}
                        className="text-xs bg-secondary/50 text-muted-foreground px-2 py-1 rounded"
                      >
                        {component}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Data Flow */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">Request Processing Flow</h3>

        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {[
            { step: 1, label: 'Agent Request', desc: 'Action intercepted' },
            { step: 2, label: 'Identity Check', desc: 'Agent verified' },
            { step: 3, label: 'Policy Eval', desc: 'Rules applied' },
            { step: 4, label: 'Decision', desc: 'Allow/Deny/Escalate' },
            { step: 5, label: 'Audit Log', desc: 'Action recorded' },
          ].map((item, index, arr) => (
            <div key={item.step} className="flex items-center">
              <div className="flex flex-col items-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent text-white font-bold mb-2">
                  {item.step}
                </div>
                <p className="text-sm font-medium text-foreground">{item.label}</p>
                <p className="text-xs text-muted-foreground">{item.desc}</p>
              </div>
              {index < arr.length - 1 && (
                <div className="hidden md:block w-12 h-0.5 bg-gradient-to-r from-primary/50 to-accent/50 mx-2" />
              )}
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
