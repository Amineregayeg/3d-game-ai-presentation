'use client';

import { motion } from 'motion/react';
import {
  LineChart,
  Line,
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
  Cell,
} from 'recharts';
import { TrendingUp, Users, Shield, Zap, Target, CheckCircle2 } from 'lucide-react';
import { successMetrics } from '@/lib/data/mvp-data';

// Mock data for charts
const adoptionData = [
  { month: 'Jan', users: 10, actions: 50000 },
  { month: 'Feb', users: 25, actions: 150000 },
  { month: 'Mar', users: 50, actions: 400000 },
  { month: 'Apr', users: 85, actions: 800000 },
  { month: 'May', users: 130, actions: 1500000 },
  { month: 'Jun', users: 200, actions: 3000000 },
];

const performanceData = [
  { day: 'Mon', latency: 3.2, throughput: 95000 },
  { day: 'Tue', latency: 3.5, throughput: 98000 },
  { day: 'Wed', latency: 2.8, throughput: 102000 },
  { day: 'Thu', latency: 3.1, throughput: 99000 },
  { day: 'Fri', latency: 2.9, throughput: 105000 },
  { day: 'Sat', latency: 2.5, throughput: 85000 },
  { day: 'Sun', latency: 2.4, throughput: 75000 },
];

const categoryData = [
  { name: 'API Calls', value: 45, color: '#3B82F6' },
  { name: 'DB Queries', value: 25, color: '#10B981' },
  { name: 'File Access', value: 15, color: '#8B5CF6' },
  { name: 'External', value: 10, color: '#F59E0B' },
  { name: 'Other', value: 5, color: '#6B7280' },
];

export function MetricsSection() {
  return (
    <div className="space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          Success Metrics
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-lg text-muted-foreground"
        >
          Key performance indicators and targets for measuring MVP success
        </motion.p>
      </div>

      {/* KPI Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {successMetrics.kpis.map((kpi, index) => {
          const icons = [Users, Shield, Zap, TrendingUp];
          const Icon = icons[index] || Target;

          return (
            <motion.div
              key={kpi.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
              className="relative overflow-hidden rounded-xl border border-border/50 bg-card/30 p-5 backdrop-blur-sm"
            >
              <div className="absolute top-0 right-0 w-20 h-20 bg-primary/5 rounded-full blur-2xl" />
              <div className="relative">
                <div className="flex items-center justify-between mb-3">
                  <Icon className="h-6 w-6 text-primary" />
                  <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full">
                    {kpi.trend}
                  </span>
                </div>
                <p className="text-3xl font-bold text-foreground mb-1">{kpi.target}</p>
                <p className="text-sm text-muted-foreground">{kpi.name}</p>
                <p className="text-xs text-muted-foreground mt-1">{kpi.description}</p>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Adoption Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-lg font-semibold mb-6">Projected Adoption (6 Months)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={adoptionData}>
                <defs>
                  <linearGradient id="usersGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="month" stroke="rgba(255,255,255,0.5)" />
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
                  dataKey="users"
                  name="Customers"
                  stroke="#3B82F6"
                  fillOpacity={1}
                  fill="url(#usersGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Action Volume Chart */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-lg font-semibold mb-6">Actions Protected (Monthly)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={adoptionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="month" stroke="rgba(255,255,255,0.5)" />
                <YAxis stroke="rgba(255,255,255,0.5)" tickFormatter={(value) => `${value / 1000000}M`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                  }}
                  formatter={(value) => [`${(Number(value) / 1000000).toFixed(1)}M`, 'Actions']}
                />
                <Bar dataKey="actions" fill="#10B981" radius={[4, 4, 0, 0]} name="Actions" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Performance Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
      >
        <h3 className="text-lg font-semibold mb-6">System Performance (Last 7 Days)</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="day" stroke="rgba(255,255,255,0.5)" />
              <YAxis yAxisId="latency" stroke="#F59E0B" />
              <YAxis yAxisId="throughput" orientation="right" stroke="#8B5CF6" tickFormatter={(value) => `${value / 1000}K`} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Line
                yAxisId="latency"
                type="monotone"
                dataKey="latency"
                name="Latency (ms)"
                stroke="#F59E0B"
                strokeWidth={2}
                dot={{ fill: '#F59E0B' }}
              />
              <Line
                yAxisId="throughput"
                type="monotone"
                dataKey="throughput"
                name="Throughput (req/s)"
                stroke="#8B5CF6"
                strokeWidth={2}
                dot={{ fill: '#8B5CF6' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Action Categories */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Category Distribution */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-lg font-semibold mb-6">Action Category Distribution</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis type="number" stroke="rgba(255,255,255,0.5)" />
                <YAxis dataKey="name" type="category" stroke="rgba(255,255,255,0.5)" width={80} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                  }}
                  formatter={(value) => [`${value}%`, 'Share']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {categoryData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Success Targets */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-lg font-semibold mb-6">MVP Success Targets</h3>
          <div className="space-y-4">
            {successMetrics.targets.map((target) => (
              <div key={target.metric} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-foreground">{target.metric}</span>
                  <span className="text-sm font-medium text-primary">{target.target}</span>
                </div>
                <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${target.current}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1 }}
                    className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Current: {target.current}%</span>
                  <span>{target.status}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* North Star Metric */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        className="relative overflow-hidden rounded-2xl border border-primary/30 bg-gradient-to-br from-primary/10 via-transparent to-accent/10 p-8 text-center"
      >
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-primary/5 via-transparent to-transparent" />
        <div className="relative">
          <Target className="h-12 w-12 text-primary mx-auto mb-4" />
          <p className="text-sm font-medium text-primary uppercase tracking-wider mb-2">
            North Star Metric
          </p>
          <p className="text-4xl font-bold text-foreground mb-4">
            10M Actions Protected / Month
          </p>
          <p className="text-muted-foreground max-w-xl mx-auto">
            By the end of MVP phase, we aim to protect 10 million AI agent actions per month
            across our customer base, demonstrating product-market fit and platform reliability.
          </p>
          <div className="flex items-center justify-center gap-6 mt-6">
            {[
              { label: 'Current', value: '500K' },
              { label: 'Target', value: '10M' },
              { label: 'Growth', value: '20x' },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-2xl font-bold text-primary">{stat.value}</p>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
