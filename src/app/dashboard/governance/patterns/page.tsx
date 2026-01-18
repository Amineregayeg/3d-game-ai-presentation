"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  Layers,
  Plus,
  Trash2,
  ArrowLeft,
  Search,
  Filter,
  Check,
  X,
  AlertTriangle,
  MinusCircle
} from "lucide-react";

interface Pattern {
  id: number;
  name: string;
  type: string;
  platform: string;
  pattern: Record<string, unknown>;
  description: string;
  rationale: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export default function PatternsPage() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newPattern, setNewPattern] = useState({
    name: "",
    type: "recommended",
    platform: "salesforce",
    description: "",
    rationale: "",
    pattern: {}
  });

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  useEffect(() => {
    fetchPatterns();
  }, [typeFilter]);

  async function fetchPatterns() {
    try {
      const token = localStorage.getItem("access_token");
      const params = new URLSearchParams();
      if (typeFilter) params.append("type", typeFilter);

      const res = await fetch(`${apiUrl}/api/governance/patterns?${params}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setPatterns(data.patterns);
      }
    } catch (error) {
      console.error("Failed to fetch patterns:", error);
    } finally {
      setLoading(false);
    }
  }

  async function createPattern() {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${apiUrl}/api/governance/patterns`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newPattern)
      });
      if (res.ok) {
        setShowCreateModal(false);
        setNewPattern({
          name: "",
          type: "recommended",
          platform: "salesforce",
          description: "",
          rationale: "",
          pattern: {}
        });
        fetchPatterns();
      }
    } catch (error) {
      console.error("Failed to create pattern:", error);
    }
  }

  async function deletePattern(id: number) {
    if (!confirm("Are you sure you want to delete this pattern?")) return;
    try {
      const token = localStorage.getItem("access_token");
      await fetch(`${apiUrl}/api/governance/patterns/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchPatterns();
    } catch (error) {
      console.error("Failed to delete pattern:", error);
    }
  }

  const filteredPatterns = patterns.filter((p) =>
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const typeConfig: Record<string, { color: string; icon: React.ElementType; label: string }> = {
    allowed: { color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30", icon: Check, label: "Allowed" },
    forbidden: { color: "bg-red-500/20 text-red-400 border-red-500/30", icon: X, label: "Forbidden" },
    recommended: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", icon: Check, label: "Recommended" },
    deprecated: { color: "bg-amber-500/20 text-amber-400 border-amber-500/30", icon: AlertTriangle, label: "Deprecated" }
  };

  const patternTypes = ["allowed", "forbidden", "recommended", "deprecated"];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Link href="/dashboard/governance">
            <button className="p-2 text-slate-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <Layers className="w-7 h-7 text-rose-400" />
              Pattern Library
            </h1>
            <p className="text-slate-400 mt-1">Define allowed, forbidden, and recommended patterns</p>
          </div>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-rose-500 rounded-lg font-medium text-white hover:bg-rose-600 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Pattern
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
          <input
            type="text"
            placeholder="Search patterns..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-rose-500/50"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          {patternTypes.map((type) => {
            const config = typeConfig[type];
            return (
              <button
                key={type}
                onClick={() => setTypeFilter(typeFilter === type ? "" : type)}
                className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors border ${
                  typeFilter === type ? config.color : "border-white/10 text-slate-400 hover:text-white"
                }`}
              >
                <config.icon className="w-4 h-4" />
                {config.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Patterns List */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-6 animate-pulse">
              <div className="h-6 bg-white/10 rounded w-1/4 mb-3" />
              <div className="h-5 bg-white/10 rounded w-1/2 mb-2" />
              <div className="h-4 bg-white/10 rounded w-3/4" />
            </div>
          ))}
        </div>
      ) : filteredPatterns.length === 0 ? (
        <div className="text-center py-12">
          <Layers className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">No patterns found. Add your first pattern to get started.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredPatterns.map((pattern, index) => {
            const config = typeConfig[pattern.type] || typeConfig.recommended;
            return (
              <motion.div
                key={pattern.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`bg-white/5 border rounded-xl p-6 hover:bg-white/10 transition-all ${config.color.includes("border") ? config.color.split(" ").find((c) => c.startsWith("border-")) : "border-white/10"}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${config.color}`}>
                        <config.icon className="w-3 h-3" />
                        {config.label.toUpperCase()}
                      </span>
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-white/10 text-slate-300">
                        {pattern.platform}
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">{pattern.name}</h3>
                    {pattern.description && (
                      <p className="text-slate-400 text-sm mb-2">{pattern.description}</p>
                    )}
                    {pattern.rationale && (
                      <p className="text-slate-500 text-sm italic">
                        <span className="text-slate-400">Rationale:</span> {pattern.rationale}
                      </p>
                    )}
                    <p className="text-slate-500 text-xs mt-3">
                      Created {new Date(pattern.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={() => deletePattern(pattern.id)}
                    className="p-2 text-slate-500 hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-900 border border-white/10 rounded-2xl p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Add Pattern</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Pattern Name</label>
                <input
                  type="text"
                  value={newPattern.name}
                  onChange={(e) => setNewPattern({ ...newPattern, name: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                  placeholder="e.g., SOQL in Loops"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1">Type</label>
                  <select
                    value={newPattern.type}
                    onChange={(e) => setNewPattern({ ...newPattern, type: e.target.value })}
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                  >
                    <option value="allowed">Allowed</option>
                    <option value="forbidden">Forbidden</option>
                    <option value="recommended">Recommended</option>
                    <option value="deprecated">Deprecated</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1">Platform</label>
                  <select
                    value={newPattern.platform}
                    onChange={(e) => setNewPattern({ ...newPattern, platform: e.target.value })}
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                  >
                    <option value="salesforce">Salesforce</option>
                    <option value="general">General</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Description</label>
                <textarea
                  value={newPattern.description}
                  onChange={(e) => setNewPattern({ ...newPattern, description: e.target.value })}
                  rows={2}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-rose-500/50 resize-none"
                  placeholder="Describe the pattern"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Rationale</label>
                <textarea
                  value={newPattern.rationale}
                  onChange={(e) => setNewPattern({ ...newPattern, rationale: e.target.value })}
                  rows={2}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-rose-500/50 resize-none"
                  placeholder="Why is this pattern allowed/forbidden?"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={createPattern}
                disabled={!newPattern.name}
                className="flex-1 px-4 py-2 bg-rose-500 rounded-lg font-medium text-white hover:bg-rose-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Create Pattern
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
