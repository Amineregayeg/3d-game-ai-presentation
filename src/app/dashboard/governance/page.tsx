"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  Shield,
  ClipboardCheck,
  ScrollText,
  BookOpen,
  Layers,
  ArrowRight,
  AlertCircle,
  CheckCircle,
  TrendingUp
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

interface GovernanceStats {
  policies: { total: number; active: number };
  approvals: { pending: number; total: number };
  knowledge: { entries: number };
  patterns: { total: number; by_type: Record<string, number> };
  audit: { total_logs: number };
}

function StatCard({
  icon: Icon,
  title,
  value,
  subtitle,
  href,
  color,
  badge
}: {
  icon: React.ElementType;
  title: string;
  value: number | string;
  subtitle: string;
  href: string;
  color: string;
  badge?: { text: string; color: string };
}) {
  return (
    <Link href={href}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        whileHover={{ scale: 1.02 }}
        className="bg-white/5 border border-white/10 rounded-2xl p-6 cursor-pointer hover:bg-white/10 transition-all"
      >
        <div className="flex items-start justify-between mb-4">
          <div className={`p-3 rounded-xl bg-gradient-to-br ${color}`}>
            <Icon className="w-6 h-6 text-white" />
          </div>
          {badge && (
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${badge.color}`}>
              {badge.text}
            </span>
          )}
        </div>
        <h3 className="text-lg font-semibold text-white mb-1">{title}</h3>
        <p className="text-3xl font-bold text-white mb-2">{value}</p>
        <p className="text-slate-400 text-sm">{subtitle}</p>
        <div className="flex items-center gap-1 mt-4 text-blue-400 text-sm">
          <span>View details</span>
          <ArrowRight className="w-4 h-4" />
        </div>
      </motion.div>
    </Link>
  );
}

export default function GovernanceDashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState<GovernanceStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStats() {
      try {
        const token = localStorage.getItem("access_token");
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"}/api/governance/stats`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch (error) {
        console.error("Failed to fetch governance stats:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchStats();
  }, []);

  const userName = user?.name?.split(" ")[0] || user?.email?.split("@")[0] || "User";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-2">
          <Shield className="w-8 h-8 text-purple-400" />
          <h1 className="text-3xl font-bold text-white">Governance Center</h1>
        </div>
        <p className="text-slate-400">
          Manage policies, approvals, knowledge base, and patterns for your organization.
        </p>
      </motion.div>

      {/* Stats Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="bg-white/5 border border-white/10 rounded-2xl p-6 animate-pulse">
              <div className="w-12 h-12 bg-white/10 rounded-xl mb-4" />
              <div className="h-6 bg-white/10 rounded w-1/2 mb-2" />
              <div className="h-10 bg-white/10 rounded w-1/3" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <StatCard
            icon={Shield}
            title="Policies"
            value={stats?.policies.active || 0}
            subtitle={`${stats?.policies.total || 0} total policies`}
            href="/dashboard/governance/policies"
            color="from-purple-500 to-indigo-500"
          />
          <StatCard
            icon={ClipboardCheck}
            title="Approvals"
            value={stats?.approvals.pending || 0}
            subtitle="Pending requests"
            href="/dashboard/governance/approvals"
            color="from-amber-500 to-orange-500"
            badge={
              stats?.approvals.pending
                ? { text: `${stats.approvals.pending} pending`, color: "bg-amber-500/20 text-amber-400" }
                : undefined
            }
          />
          <StatCard
            icon={ScrollText}
            title="Audit Logs"
            value={stats?.audit.total_logs || 0}
            subtitle="Total logged actions"
            href="/dashboard/governance/audit"
            color="from-cyan-500 to-blue-500"
          />
          <StatCard
            icon={BookOpen}
            title="Knowledge Base"
            value={stats?.knowledge.entries || 0}
            subtitle="Active entries"
            href="/dashboard/governance/knowledge"
            color="from-emerald-500 to-teal-500"
          />
          <StatCard
            icon={Layers}
            title="Patterns"
            value={stats?.patterns.total || 0}
            subtitle={`${stats?.patterns.by_type?.allowed || 0} allowed, ${stats?.patterns.by_type?.forbidden || 0} forbidden`}
            href="/dashboard/governance/patterns"
            color="from-rose-500 to-pink-500"
          />
        </div>
      )}

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Pending Approvals Alert */}
        {stats?.approvals.pending ? (
          <div className="bg-gradient-to-br from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="w-6 h-6 text-amber-400" />
              <h3 className="text-lg font-semibold text-white">Action Required</h3>
            </div>
            <p className="text-slate-300 mb-4">
              You have {stats.approvals.pending} pending approval request{stats.approvals.pending > 1 ? "s" : ""} waiting for review.
            </p>
            <Link href="/dashboard/governance/approvals">
              <button className="flex items-center gap-2 px-4 py-2 bg-amber-500 rounded-lg font-medium text-white hover:bg-amber-600 transition-colors">
                Review Approvals
                <ArrowRight className="w-4 h-4" />
              </button>
            </Link>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <CheckCircle className="w-6 h-6 text-emerald-400" />
              <h3 className="text-lg font-semibold text-white">All Caught Up</h3>
            </div>
            <p className="text-slate-300">
              No pending approvals. All governance workflows are running smoothly.
            </p>
          </div>
        )}

        {/* Governance Health */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-6 h-6 text-blue-400" />
            <h3 className="text-lg font-semibold text-white">Governance Health</h3>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Active Policies</span>
              <span className="text-white font-medium">{stats?.policies.active || 0} / {stats?.policies.total || 0}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Knowledge Coverage</span>
              <span className="text-emerald-400 font-medium">{stats?.knowledge.entries || 0} entries</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Pattern Library</span>
              <span className="text-white font-medium">{stats?.patterns.total || 0} patterns defined</span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
