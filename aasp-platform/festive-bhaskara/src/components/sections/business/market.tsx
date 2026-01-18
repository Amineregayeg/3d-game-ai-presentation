'use client';

import { motion } from 'motion/react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { TrendingUp, Users, FileWarning, DollarSign } from 'lucide-react';
import { marketData } from '@/lib/data/business-data';
import { StatsCard } from '@/components/ui/feature-card';

const COLORS = ['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B', '#6B7280'];

export function MarketSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          {marketData.title}
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          {marketData.subtitle}
        </motion.p>
      </div>

      {/* TAM/SAM/SOM Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        <div className="relative overflow-hidden rounded-2xl border border-border/50 bg-card/50 p-6 backdrop-blur-sm">
          <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl" />
          <div className="relative">
            <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-2">
              {marketData.tam.label}
            </p>
            <p className="text-4xl font-bold text-foreground mb-2">{marketData.tam.value}</p>
            <p className="text-sm text-muted-foreground">{marketData.tam.description}</p>
          </div>
        </div>

        <div className="relative overflow-hidden rounded-2xl border border-primary/30 bg-primary/5 p-6">
          <div className="absolute top-0 right-0 w-32 h-32 bg-primary/20 rounded-full blur-3xl" />
          <div className="relative">
            <p className="text-sm font-medium text-primary uppercase tracking-wider mb-2">
              {marketData.sam.label}
            </p>
            <p className="text-4xl font-bold text-foreground mb-2">{marketData.sam.value}</p>
            <p className="text-sm text-muted-foreground">{marketData.sam.description}</p>
          </div>
        </div>

        <div className="relative overflow-hidden rounded-2xl border border-accent/30 bg-accent/5 p-6">
          <div className="absolute top-0 right-0 w-32 h-32 bg-accent/20 rounded-full blur-3xl" />
          <div className="relative">
            <p className="text-sm font-medium text-accent uppercase tracking-wider mb-2">
              {marketData.som.label}
            </p>
            <p className="text-4xl font-bold text-foreground mb-2">{marketData.som.value}</p>
            <p className="text-sm text-muted-foreground">{marketData.som.description}</p>
          </div>
        </div>
      </motion.div>

      {/* Market Growth Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
      >
        <h3 className="text-xl font-semibold mb-6">Market Growth Projection ($ Billions)</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={marketData.growth}>
              <defs>
                <linearGradient id="tamGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="samGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="somGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
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
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="tam"
                name="TAM"
                stroke="#3B82F6"
                fillOpacity={1}
                fill="url(#tamGradient)"
              />
              <Area
                type="monotone"
                dataKey="sam"
                name="SAM"
                stroke="#8B5CF6"
                fillOpacity={1}
                fill="url(#samGradient)"
              />
              <Area
                type="monotone"
                dataKey="som"
                name="SOM"
                stroke="#10B981"
                fillOpacity={1}
                fill="url(#somGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Market Drivers */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {marketData.drivers.map((driver, index) => (
          <StatsCard
            key={driver.title}
            value={driver.value}
            label={driver.title}
            change={driver.description}
            changeType="positive"
            icon={
              index === 0 ? <Users className="h-5 w-5" /> :
              index === 1 ? <FileWarning className="h-5 w-5" /> :
              index === 2 ? <TrendingUp className="h-5 w-5" /> :
              <DollarSign className="h-5 w-5" />
            }
          />
        ))}
      </div>

      {/* Market Segments */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-8"
      >
        {/* Pie Chart */}
        <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
          <h3 className="text-xl font-semibold mb-6">Target Market Segments</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={marketData.segments}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="percentage"
                >
                  {marketData.segments.map((entry, index) => (
                    <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap justify-center gap-4 mt-4">
            {marketData.segments.map((segment, index) => (
              <div key={segment.name} className="flex items-center gap-2">
                <div
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="text-sm text-muted-foreground">
                  {segment.name} ({segment.percentage}%)
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Bar Chart */}
        <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
          <h3 className="text-xl font-semibold mb-6">Segment Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={marketData.segments} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" stroke="rgba(255,255,255,0.5)" />
                <YAxis dataKey="name" type="category" stroke="rgba(255,255,255,0.5)" width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="percentage" fill="#3B82F6" radius={[0, 4, 4, 0]}>
                  {marketData.segments.map((entry, index) => (
                    <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
