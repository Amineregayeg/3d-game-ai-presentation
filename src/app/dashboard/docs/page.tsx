"use client";

import { useState } from "react";
import { motion } from "motion/react";
import {
  Book,
  Code,
  MessageSquare,
  Settings,
  Zap,
  Search,
  ChevronRight,
  ExternalLink,
  FileText,
  Video,
  HelpCircle,
  type LucideIcon
} from "lucide-react";

interface DocSection {
  id: string;
  title: string;
  description: string;
  icon: LucideIcon;
  articles: { title: string; href: string }[];
}

const docSections: DocSection[] = [
  {
    id: "getting-started",
    title: "Getting Started",
    description: "Learn the basics of using SF Consultant AI",
    icon: Book,
    articles: [
      { title: "Introduction to SF Consultant AI", href: "#" },
      { title: "Choosing Your First Consultant", href: "#" },
      { title: "Understanding Expertise Levels", href: "#" },
      { title: "Your First Conversation", href: "#" },
    ],
  },
  {
    id: "conversations",
    title: "Conversations",
    description: "Master the art of consulting sessions",
    icon: MessageSquare,
    articles: [
      { title: "How to Ask Effective Questions", href: "#" },
      { title: "Using Voice Input", href: "#" },
      { title: "Conversation History", href: "#" },
      { title: "Exporting Transcripts", href: "#" },
    ],
  },
  {
    id: "consultants",
    title: "AI Consultants",
    description: "Deep dive into each consultant's capabilities",
    icon: Zap,
    articles: [
      { title: "Alex - Beginner Guide", href: "#" },
      { title: "Jordan - Power User Expert", href: "#" },
      { title: "Morgan - Enterprise Architect", href: "#" },
      { title: "French Consultants Overview", href: "#" },
    ],
  },
  {
    id: "integrations",
    title: "Integrations",
    description: "Connect with your Salesforce org",
    icon: Code,
    articles: [
      { title: "Salesforce MCP Setup", href: "#" },
      { title: "OAuth Configuration", href: "#" },
      { title: "API Reference", href: "#" },
      { title: "Webhooks", href: "#" },
    ],
  },
  {
    id: "settings",
    title: "Account & Settings",
    description: "Manage your account preferences",
    icon: Settings,
    articles: [
      { title: "Profile Settings", href: "#" },
      { title: "Billing & Plans", href: "#" },
      { title: "Team Management", href: "#" },
      { title: "Security Settings", href: "#" },
    ],
  },
];

function DocCard({ section }: { section: DocSection }) {
  const Icon = section.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/5 border border-white/10 rounded-2xl p-6 hover:border-white/20 transition-colors"
    >
      <div className="flex items-start gap-4 mb-4">
        <div className="p-3 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl">
          <Icon className="w-6 h-6 text-blue-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white mb-1">{section.title}</h3>
          <p className="text-slate-400 text-sm">{section.description}</p>
        </div>
      </div>

      <ul className="space-y-2">
        {section.articles.map((article, index) => (
          <li key={index}>
            <a
              href={article.href}
              className="flex items-center gap-2 py-2 px-3 rounded-lg text-slate-300 hover:bg-white/5 hover:text-white transition-colors group"
            >
              <FileText className="w-4 h-4 text-slate-500 group-hover:text-blue-400" />
              <span className="flex-1">{article.title}</span>
              <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-white group-hover:translate-x-1 transition-all" />
            </a>
          </li>
        ))}
      </ul>
    </motion.div>
  );
}

export default function DocsPage() {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredSections = docSections.filter(section =>
    section.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    section.articles.some(a => a.title.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl font-bold text-white mb-4">Documentation</h1>
        <p className="text-xl text-slate-400 max-w-2xl mx-auto">
          Everything you need to know about using SF Consultant AI effectively.
        </p>
      </motion.div>

      {/* Search */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="max-w-2xl mx-auto mb-12"
      >
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
          <input
            type="text"
            placeholder="Search documentation..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all text-lg"
          />
        </div>
      </motion.div>

      {/* Quick Links */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12"
      >
        <a
          href="#"
          className="flex items-center gap-4 p-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl hover:border-blue-500/40 transition-colors group"
        >
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Video className="w-5 h-5 text-blue-400" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">Video Tutorials</h4>
            <p className="text-slate-400 text-sm">Watch step-by-step guides</p>
          </div>
          <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-white transition-colors" />
        </a>

        <a
          href="#"
          className="flex items-center gap-4 p-4 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 rounded-xl hover:border-emerald-500/40 transition-colors group"
        >
          <div className="p-2 bg-emerald-500/20 rounded-lg">
            <Code className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">API Reference</h4>
            <p className="text-slate-400 text-sm">Integrate with your apps</p>
          </div>
          <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-white transition-colors" />
        </a>

        <a
          href="#"
          className="flex items-center gap-4 p-4 bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/20 rounded-xl hover:border-orange-500/40 transition-colors group"
        >
          <div className="p-2 bg-orange-500/20 rounded-lg">
            <HelpCircle className="w-5 h-5 text-orange-400" />
          </div>
          <div className="flex-1">
            <h4 className="text-white font-medium">Get Support</h4>
            <p className="text-slate-400 text-sm">Contact our team</p>
          </div>
          <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-white transition-colors" />
        </a>
      </motion.div>

      {/* Doc Sections Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredSections.map((section, index) => (
          <motion.div
            key={section.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index }}
          >
            <DocCard section={section} />
          </motion.div>
        ))}
      </div>

      {/* Empty State */}
      {filteredSections.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-16"
        >
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
            <Search className="w-8 h-8 text-slate-500" />
          </div>
          <h3 className="text-xl font-medium text-white mb-2">No results found</h3>
          <p className="text-slate-400">Try a different search term</p>
        </motion.div>
      )}
    </div>
  );
}
