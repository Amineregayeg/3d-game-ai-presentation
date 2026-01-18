"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  MessageSquare,
  Clock,
  TrendingUp,
  Users,
  ArrowRight,
  Sparkles,
  History,
  Star,
  type LucideIcon
} from "lucide-react";
import { avatars, getAvatarsByLanguage } from "@/lib/avatars";
import { useAuth } from "@/contexts/AuthContext";

// Stats Card Component
function StatCard({
  icon: Icon,
  label,
  value,
  trend,
  color
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  trend?: string;
  color: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/5 border border-white/10 rounded-2xl p-6"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${color}`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        {trend && (
          <span className="text-emerald-400 text-sm font-medium flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            {trend}
          </span>
        )}
      </div>
      <p className="text-2xl font-bold text-white mb-1">{value}</p>
      <p className="text-slate-400 text-sm">{label}</p>
    </motion.div>
  );
}

// Recent Session Card
function SessionCard({
  avatar,
  topic,
  time,
  messages
}: {
  avatar: string;
  topic: string;
  time: string;
  messages: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-4 p-4 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors cursor-pointer"
    >
      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-medium">
        {avatar.charAt(0)}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white font-medium truncate">{topic}</p>
        <p className="text-slate-400 text-sm">with {avatar}</p>
      </div>
      <div className="text-right">
        <p className="text-slate-400 text-sm">{time}</p>
        <p className="text-slate-500 text-xs">{messages} messages</p>
      </div>
    </motion.div>
  );
}

// Avatar Quick Access Card
function AvatarQuickCard({ avatar }: { avatar: typeof avatars[0] }) {
  return (
    <Link href={`/dashboard/product?avatar=${avatar.id}`}>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.02 }}
        className={`p-4 rounded-xl bg-gradient-to-br ${avatar.accentColor} bg-opacity-10 border border-white/10 cursor-pointer`}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white font-bold">
            {avatar.name.charAt(0)}
          </div>
          <div>
            <p className="text-white font-medium">{avatar.name}</p>
            <p className="text-white/70 text-xs">{avatar.title}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {Array.from({ length: avatar.expertiseLevel === 'beginner' ? 1 : avatar.expertiseLevel === 'intermediate' ? 2 : 3 }).map((_, i) => (
            <Star key={i} className="w-3 h-3 text-yellow-400 fill-yellow-400" />
          ))}
          <span className="text-white/60 text-xs ml-1 capitalize">{avatar.expertiseLevel}</span>
        </div>
      </motion.div>
    </Link>
  );
}

export default function DashboardHome() {
  const { user } = useAuth();
  const [selectedLanguage, setSelectedLanguage] = useState<'en' | 'fr'>('en');
  const filteredAvatars = getAvatarsByLanguage(selectedLanguage);

  const userName = user?.name?.split(' ')[0] || user?.email?.split('@')[0] || 'User';

  const recentSessions = [
    { avatar: "Alex", topic: "Understanding SOQL Basics", time: "2h ago", messages: 12 },
    { avatar: "Jordan", topic: "Building Custom Reports", time: "Yesterday", messages: 28 },
    { avatar: "Morgan", topic: "Apex Trigger Best Practices", time: "2 days ago", messages: 45 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-white mb-2">Welcome back, {userName}</h1>
        <p className="text-slate-400">Here&apos;s what&apos;s happening with your Salesforce consulting sessions.</p>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          icon={MessageSquare}
          label="Total Conversations"
          value="127"
          trend="+12%"
          color="from-blue-500 to-indigo-500"
        />
        <StatCard
          icon={Clock}
          label="Hours Saved"
          value="42h"
          trend="+8%"
          color="from-emerald-500 to-teal-500"
        />
        <StatCard
          icon={Users}
          label="Consultants Used"
          value="4"
          color="from-purple-500 to-pink-500"
        />
        <StatCard
          icon={Sparkles}
          label="Questions Answered"
          value="523"
          trend="+15%"
          color="from-orange-500 to-red-500"
        />
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Recent Sessions */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <History className="w-5 h-5 text-slate-400" />
              Recent Sessions
            </h2>
            <Link href="/dashboard/history" className="text-blue-400 text-sm hover:text-blue-300 transition-colors flex items-center gap-1">
              View all
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="space-y-3">
            {recentSessions.map((session, index) => (
              <SessionCard key={index} {...session} />
            ))}
          </div>

          {/* Quick Start */}
          <div className="mt-8 p-6 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-2xl">
            <h3 className="text-lg font-semibold text-white mb-2">Start a New Conversation</h3>
            <p className="text-slate-400 mb-4">Choose a consultant from the marketplace or continue with your favorite.</p>
            <Link href="/dashboard/marketplace">
              <button className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium text-white hover:shadow-lg hover:shadow-blue-500/25 transition-all">
                Browse Consultants
                <ArrowRight className="w-4 h-4" />
              </button>
            </Link>
          </div>
        </div>

        {/* Quick Access Avatars */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Quick Access</h2>
            <div className="flex gap-1 bg-white/5 rounded-lg p-1">
              <button
                onClick={() => setSelectedLanguage('en')}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  selectedLanguage === 'en' ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                EN
              </button>
              <button
                onClick={() => setSelectedLanguage('fr')}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  selectedLanguage === 'fr' ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                FR
              </button>
            </div>
          </div>
          <div className="space-y-3">
            {filteredAvatars.map((avatar) => (
              <AvatarQuickCard key={avatar.id} avatar={avatar} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
