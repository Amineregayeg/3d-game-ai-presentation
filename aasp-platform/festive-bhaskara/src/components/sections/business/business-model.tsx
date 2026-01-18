'use client';

import { motion } from 'motion/react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Check, Zap, Building2, Rocket } from 'lucide-react';
import { businessModelData } from '@/lib/data/business-data';

const tierIcons = {
  Starter: Zap,
  Professional: Building2,
  Enterprise: Rocket,
};

export function BusinessModelSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {businessModelData.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {businessModelData.subtitle}
        </motion.p>
      </div>

      {/* Pricing Tiers */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        {businessModelData.pricing.map((pricingTier, index) => {
          const Icon = tierIcons[pricingTier.tier as keyof typeof tierIcons] || Zap;
          const isPopular = pricingTier.tier === 'Professional';

          return (
            <div
              key={pricingTier.tier}
              className={`relative overflow-hidden rounded-2xl border p-6 ${
                isPopular
                  ? 'border-primary/50 bg-primary/5'
                  : 'border-border/50 bg-card/30'
              } backdrop-blur-sm`}
            >
              {isPopular && (
                <div className="absolute top-0 right-0 bg-primary text-primary-foreground text-xs font-semibold px-3 py-1 rounded-bl-lg">
                  Most Popular
                </div>
              )}

              <div className="flex items-center gap-3 mb-4">
                <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${
                  isPopular ? 'bg-primary/20' : 'bg-secondary'
                }`}>
                  <Icon className={`h-5 w-5 ${isPopular ? 'text-primary' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">{pricingTier.tier}</h3>
                  <p className="text-xs text-muted-foreground">{pricingTier.description}</p>
                </div>
              </div>

              <div className="mb-6">
                <span className="text-3xl font-bold text-foreground">{pricingTier.price}</span>
                {pricingTier.price !== 'Custom' && (
                  <span className="text-muted-foreground">{pricingTier.period}</span>
                )}
              </div>

              <ul className="space-y-3 mb-6">
                {pricingTier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-2">
                    <Check className={`h-4 w-4 mt-0.5 shrink-0 ${
                      isPopular ? 'text-primary' : 'text-green-400'
                    }`} />
                    <span className="text-sm text-muted-foreground">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                className={`w-full rounded-lg py-2.5 text-sm font-medium transition-all ${
                  isPopular
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'bg-secondary text-foreground hover:bg-secondary/80'
                }`}
              >
                {pricingTier.price === 'Custom' ? 'Contact Sales' : 'Get Started'}
              </button>
            </div>
          );
        })}
      </motion.div>

      {/* Revenue Projections */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6">Revenue Projection ($ Millions ARR)</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={businessModelData.revenueProjections}>
              <defs>
                <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="year" stroke="rgba(255,255,255,0.5)" />
              <YAxis stroke="rgba(255,255,255,0.5)" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                }}
                formatter={(value) => [`$${value}M`, 'ARR']}
              />
              <Area
                type="monotone"
                dataKey="arr"
                stroke="#3B82F6"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#revenueGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Key Metrics Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="grid grid-cols-1 md:grid-cols-2 gap-8"
      >
        {/* Unit Economics */}
        <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
          <h3 className="text-xl font-semibold mb-6">Unit Economics</h3>
          <div className="space-y-4">
            {businessModelData.unitEconomics.map((metric) => (
              <div
                key={metric.metric}
                className="flex items-center justify-between p-3 rounded-lg bg-secondary/30"
              >
                <span className="text-muted-foreground">{metric.metric}</span>
                <span className="font-semibold text-foreground">{metric.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Customer Projections */}
        <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
          <h3 className="text-xl font-semibold mb-6">Customer Growth</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={businessModelData.revenueProjections}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="year" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="customers" fill="#10B981" radius={[4, 4, 0, 0]} name="Customers" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Funding Ask */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        className="relative overflow-hidden rounded-2xl border border-primary/30 bg-primary/5 p-8"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-accent/10" />
        <div className="relative grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <p className="text-sm font-medium text-primary uppercase tracking-wider mb-2">
              Seed Round
            </p>
            <p className="text-4xl font-bold text-foreground mb-4">
              {businessModelData.funding.amount}
            </p>
            <p className="text-muted-foreground">
              {businessModelData.funding.use}
            </p>
          </div>
          <div className="space-y-3">
            <h4 className="font-semibold text-foreground mb-4">Use of Funds</h4>
            {businessModelData.funding.breakdown.map((item) => (
              <div key={item.category} className="flex items-center gap-3">
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">{item.category}</span>
                    <span className="text-foreground font-medium">{item.percentage}%</span>
                  </div>
                  <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
