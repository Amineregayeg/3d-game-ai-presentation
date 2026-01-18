'use client';

import { motion } from 'motion/react';
import { Zap, Globe, FileSearch, Shield, CheckCircle2, XCircle, Minus } from 'lucide-react';
import { competitiveData } from '@/lib/data/business-data';
import { FeatureCard } from '@/components/ui/feature-card';

const iconMap = {
  Zap,
  Globe,
  FileSearch,
  Shield,
};

export function CompetitiveSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {competitiveData.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {competitiveData.subtitle}
        </motion.p>
      </div>

      {/* Positioning Map */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-8 text-center">Market Positioning</h3>
        <div className="relative h-96 mx-auto max-w-3xl">
          {/* Axes */}
          <div className="absolute inset-0 border-l-2 border-b-2 border-border/30">
            {/* Y Axis Label */}
            <div className="absolute -left-20 top-1/2 -rotate-90 text-sm text-muted-foreground whitespace-nowrap">
              Action Control →
            </div>
            {/* X Axis Label */}
            <div className="absolute bottom-[-30px] left-1/2 -translate-x-1/2 text-sm text-muted-foreground">
              AI-Native →
            </div>
          </div>

          {/* Grid Lines */}
          <div className="absolute inset-0">
            <div className="absolute left-1/4 top-0 bottom-0 border-l border-dashed border-border/20" />
            <div className="absolute left-1/2 top-0 bottom-0 border-l border-dashed border-border/20" />
            <div className="absolute left-3/4 top-0 bottom-0 border-l border-dashed border-border/20" />
            <div className="absolute left-0 right-0 top-1/4 border-t border-dashed border-border/20" />
            <div className="absolute left-0 right-0 top-1/2 border-t border-dashed border-border/20" />
            <div className="absolute left-0 right-0 top-3/4 border-t border-dashed border-border/20" />
          </div>

          {/* Players */}
          {competitiveData.positioning.players.map((player) => (
            <motion.div
              key={player.name}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className={`absolute flex items-center justify-center transform -translate-x-1/2 -translate-y-1/2 ${
                player.highlight
                  ? 'z-10'
                  : 'z-0'
              }`}
              style={{
                left: `${player.x}%`,
                top: `${100 - player.y}%`,
              }}
            >
              <div
                className={`relative rounded-full flex items-center justify-center transition-all ${
                  player.highlight
                    ? 'bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/30 animate-pulse-glow'
                    : 'bg-secondary/80'
                } ${
                  player.size === 'large' ? 'h-16 w-16' :
                  player.size === 'medium' ? 'h-12 w-12' : 'h-8 w-8'
                }`}
              >
                {player.highlight && (
                  <Shield className="h-6 w-6 text-white" />
                )}
              </div>
              <span
                className={`absolute top-full mt-2 text-xs whitespace-nowrap ${
                  player.highlight ? 'text-primary font-semibold' : 'text-muted-foreground'
                }`}
              >
                {player.name}
              </span>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Competitor Comparison */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 backdrop-blur-sm overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/30">
                <th className="px-6 py-4 text-left text-sm font-semibold text-foreground">Category</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-foreground">Examples</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-foreground">Strengths</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-foreground">Weaknesses</th>
                <th className="px-6 py-4 text-center text-sm font-semibold text-foreground">Threat</th>
              </tr>
            </thead>
            <tbody>
              {competitiveData.competitors.map((competitor, index) => (
                <tr
                  key={competitor.name}
                  className={index < competitiveData.competitors.length - 1 ? 'border-b border-border/20' : ''}
                >
                  <td className="px-6 py-4">
                    <span className="font-medium text-foreground">{competitor.name}</span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-wrap gap-1">
                      {competitor.examples.map((ex) => (
                        <span key={ex} className="text-xs text-muted-foreground bg-secondary/50 px-2 py-1 rounded">
                          {ex}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <ul className="space-y-1">
                      {competitor.strengths.map((s) => (
                        <li key={s} className="flex items-center gap-2 text-sm text-muted-foreground">
                          <CheckCircle2 className="h-3 w-3 text-green-400 shrink-0" />
                          {s}
                        </li>
                      ))}
                    </ul>
                  </td>
                  <td className="px-6 py-4">
                    <ul className="space-y-1">
                      {competitor.weaknesses.map((w) => (
                        <li key={w} className="flex items-center gap-2 text-sm text-muted-foreground">
                          <XCircle className="h-3 w-3 text-red-400 shrink-0" />
                          {w}
                        </li>
                      ))}
                    </ul>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span
                      className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                        competitor.threat === 'high'
                          ? 'bg-red-500/20 text-red-400'
                          : competitor.threat === 'medium'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-green-500/20 text-green-400'
                      }`}
                    >
                      {competitor.threat}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Our Advantages */}
      <div>
        <h3 className="text-2xl font-semibold text-center mb-8">Our Competitive Advantages</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {competitiveData.advantages.map((advantage, index) => {
            const Icon = iconMap[advantage.icon.name as keyof typeof iconMap] || Shield;
            return (
              <FeatureCard
                key={advantage.title}
                title={advantage.title}
                description={advantage.description}
                icon={<Icon className="h-6 w-6" />}
                index={index}
                gradient
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
