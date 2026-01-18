'use client';

import { motion } from 'motion/react';
import { Plug, Cloud, Database, Shield, Code2, ArrowRight, CheckCircle2 } from 'lucide-react';
import { integrations, deploymentOptions } from '@/lib/data/technical-data';

const categoryIcons = {
  'AI Platforms': Cloud,
  'Identity Providers': Shield,
  'Observability': Database,
  'Developer Tools': Code2,
};

export function IntegrationsSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          Integrations & Deployment
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          Seamlessly connect with your existing infrastructure and deploy anywhere
        </motion.p>
      </div>

      {/* Integration Categories */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
      >
        {integrations.map((category, index) => {
          const Icon = categoryIcons[category.category as keyof typeof categoryIcons] || Plug;
          return (
            <motion.div
              key={category.category}
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
                <h3 className="text-lg font-semibold text-foreground">{category.category}</h3>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {category.items.map((item) => (
                  <div
                    key={item.name}
                    className="flex items-center gap-2 p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div
                      className={`h-2 w-2 rounded-full ${
                        item.status === 'available'
                          ? 'bg-green-400'
                          : item.status === 'beta'
                          ? 'bg-yellow-400'
                          : 'bg-gray-400'
                      }`}
                    />
                    <span className="text-sm text-foreground">{item.name}</span>
                    {item.status === 'beta' && (
                      <span className="text-xs bg-yellow-500/20 text-yellow-400 px-1.5 py-0.5 rounded">
                        Beta
                      </span>
                    )}
                    {item.status === 'coming' && (
                      <span className="text-xs bg-gray-500/20 text-gray-400 px-1.5 py-0.5 rounded">
                        Soon
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* SDK Support */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6 text-center">Native SDK Support</h3>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
          {[
            { name: 'Python', color: '#3776AB' },
            { name: 'TypeScript', color: '#3178C6' },
            { name: 'Go', color: '#00ADD8' },
            { name: 'Rust', color: '#DEA584' },
            { name: 'Java', color: '#ED8B00' },
            { name: 'C#', color: '#239120' },
            { name: 'Ruby', color: '#CC342D' },
            { name: 'PHP', color: '#777BB4' },
          ].map((lang) => (
            <div
              key={lang.name}
              className="flex flex-col items-center gap-2 p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer"
            >
              <div
                className="h-10 w-10 rounded-lg flex items-center justify-center font-bold text-white text-sm"
                style={{ backgroundColor: lang.color }}
              >
                {lang.name.substring(0, 2)}
              </div>
              <span className="text-sm text-foreground">{lang.name}</span>
            </div>
          ))}
        </div>

        {/* Code Example */}
        <div className="mt-8">
          <div className="rounded-xl border border-border/50 bg-[#0d1117] overflow-hidden">
            <div className="flex items-center gap-4 px-4 py-2 bg-[#161b22] border-b border-border/30">
              <span className="text-xs text-muted-foreground font-mono">Quick Start</span>
            </div>
            <pre className="p-4 overflow-x-auto">
              <code className="text-sm font-mono text-gray-300">
{`# Install the SDK
pip install aasp-python

# Initialize the client
from aasp import AASPClient

client = AASPClient(
    api_key="your-api-key",
    environment="production"
)

# Wrap your AI agent actions
@client.protect
async def agent_action(action: str, params: dict):
    # Your agent logic here
    return await execute_action(action, params)`}
              </code>
            </pre>
          </div>
        </div>
      </motion.div>

      {/* Deployment Options */}
      <div>
        <h3 className="text-2xl font-semibold text-center mb-8">Deployment Options</h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {deploymentOptions.map((option, index) => (
            <motion.div
              key={option.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 * index }}
              className={`relative overflow-hidden rounded-2xl border p-6 ${
                option.recommended
                  ? 'border-primary/50 bg-primary/5'
                  : 'border-border/50 bg-card/30'
              } backdrop-blur-sm`}
            >
              {option.recommended && (
                <div className="absolute top-0 right-0 bg-primary text-primary-foreground text-xs font-semibold px-3 py-1 rounded-bl-lg">
                  Recommended
                </div>
              )}

              <div className="flex items-center gap-3 mb-4">
                <Cloud className={`h-6 w-6 ${option.recommended ? 'text-primary' : 'text-muted-foreground'}`} />
                <h4 className="text-lg font-semibold text-foreground">{option.name}</h4>
              </div>

              <p className="text-sm text-muted-foreground mb-4">{option.description}</p>

              <ul className="space-y-2 mb-6">
                {option.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-2 text-sm text-muted-foreground">
                    <CheckCircle2 className={`h-4 w-4 shrink-0 ${
                      option.recommended ? 'text-primary' : 'text-green-400'
                    }`} />
                    {feature}
                  </li>
                ))}
              </ul>

              <button
                className={`w-full flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-medium transition-all ${
                  option.recommended
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'bg-secondary text-foreground hover:bg-secondary/80'
                }`}
              >
                Learn More
                <ArrowRight className="h-4 w-4" />
              </button>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Integration Architecture */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-8 text-center">Integration Architecture</h3>

        <div className="relative">
          {/* Your Application */}
          <div className="flex justify-center mb-6">
            <div className="px-8 py-4 rounded-xl bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border border-blue-500/30">
              <p className="font-semibold text-foreground">Your Application</p>
            </div>
          </div>

          {/* Arrow */}
          <div className="flex justify-center mb-4">
            <div className="flex flex-col items-center">
              <div className="w-0.5 h-6 bg-gradient-to-b from-blue-500/50 to-primary/50" />
              <span className="text-xs text-muted-foreground my-1">SDK Call</span>
              <div className="w-0.5 h-6 bg-gradient-to-b from-primary/50 to-green-500/50" />
            </div>
          </div>

          {/* AASP */}
          <div className="flex justify-center mb-6">
            <div className="px-8 py-4 rounded-xl bg-gradient-to-r from-primary/20 to-accent/20 border border-primary/30">
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                <p className="font-semibold text-foreground">AASP Platform</p>
              </div>
              <p className="text-xs text-muted-foreground mt-1">Policy Evaluation • Audit • HITL</p>
            </div>
          </div>

          {/* Arrows to targets */}
          <div className="flex justify-center mb-4">
            <div className="flex items-center gap-16">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex flex-col items-center">
                  <div className="w-0.5 h-8 bg-gradient-to-b from-primary/50 to-green-500/50" />
                </div>
              ))}
            </div>
          </div>

          {/* Target Systems */}
          <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto">
            <div className="px-4 py-3 rounded-lg bg-secondary/30 border border-border/30 text-center">
              <Database className="h-5 w-5 text-green-400 mx-auto mb-1" />
              <p className="text-sm text-foreground">Databases</p>
            </div>
            <div className="px-4 py-3 rounded-lg bg-secondary/30 border border-border/30 text-center">
              <Cloud className="h-5 w-5 text-blue-400 mx-auto mb-1" />
              <p className="text-sm text-foreground">APIs</p>
            </div>
            <div className="px-4 py-3 rounded-lg bg-secondary/30 border border-border/30 text-center">
              <Code2 className="h-5 w-5 text-purple-400 mx-auto mb-1" />
              <p className="text-sm text-foreground">Services</p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
