"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  Shield,
  Plus,
  Edit,
  Trash2,
  ToggleLeft,
  ToggleRight,
  ArrowLeft,
  Search
} from "lucide-react";

interface Policy {
  id: number;
  name: string;
  type: string;
  rules: Record<string, unknown>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export default function PoliciesPage() {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newPolicy, setNewPolicy] = useState({ name: "", type: "type_a" });

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  useEffect(() => {
    fetchPolicies();
  }, []);

  async function fetchPolicies() {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${apiUrl}/api/governance/policies`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setPolicies(data.policies);
      }
    } catch (error) {
      console.error("Failed to fetch policies:", error);
    } finally {
      setLoading(false);
    }
  }

  async function createPolicy() {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${apiUrl}/api/governance/policies`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newPolicy)
      });
      if (res.ok) {
        setShowCreateModal(false);
        setNewPolicy({ name: "", type: "type_a" });
        fetchPolicies();
      }
    } catch (error) {
      console.error("Failed to create policy:", error);
    }
  }

  async function togglePolicy(id: number, isActive: boolean) {
    try {
      const token = localStorage.getItem("access_token");
      await fetch(`${apiUrl}/api/governance/policies/${id}`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ is_active: !isActive })
      });
      fetchPolicies();
    } catch (error) {
      console.error("Failed to toggle policy:", error);
    }
  }

  async function deletePolicy(id: number) {
    if (!confirm("Are you sure you want to delete this policy?")) return;
    try {
      const token = localStorage.getItem("access_token");
      await fetch(`${apiUrl}/api/governance/policies/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchPolicies();
    } catch (error) {
      console.error("Failed to delete policy:", error);
    }
  }

  const filteredPolicies = policies.filter((p) =>
    p.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const policyTypeColors: Record<string, string> = {
    type_a: "bg-purple-500/20 text-purple-400",
    type_b: "bg-blue-500/20 text-blue-400",
    type_c: "bg-emerald-500/20 text-emerald-400"
  };

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
              <Shield className="w-7 h-7 text-purple-400" />
              Governance Policies
            </h1>
            <p className="text-slate-400 mt-1">Manage policies that govern agent actions</p>
          </div>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-purple-500 rounded-lg font-medium text-white hover:bg-purple-600 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Policy
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
        <input
          type="text"
          placeholder="Search policies..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
        />
      </div>

      {/* Policies List */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-6 animate-pulse">
              <div className="h-6 bg-white/10 rounded w-1/3 mb-2" />
              <div className="h-4 bg-white/10 rounded w-1/4" />
            </div>
          ))}
        </div>
      ) : filteredPolicies.length === 0 ? (
        <div className="text-center py-12">
          <Shield className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">No policies found. Create your first policy to get started.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredPolicies.map((policy, index) => (
            <motion.div
              key={policy.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white/5 border border-white/10 rounded-xl p-6 hover:bg-white/10 transition-all"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`w-3 h-3 rounded-full ${policy.is_active ? "bg-emerald-400" : "bg-slate-500"}`} />
                  <div>
                    <h3 className="text-lg font-semibold text-white">{policy.name}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${policyTypeColors[policy.type] || "bg-slate-500/20 text-slate-400"}`}>
                        {policy.type?.replace("_", " ").toUpperCase()}
                      </span>
                      <span className="text-slate-500 text-sm">
                        Created {new Date(policy.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => togglePolicy(policy.id, policy.is_active)}
                    className="p-2 text-slate-400 hover:text-white transition-colors"
                    title={policy.is_active ? "Deactivate" : "Activate"}
                  >
                    {policy.is_active ? (
                      <ToggleRight className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <ToggleLeft className="w-5 h-5" />
                    )}
                  </button>
                  <button
                    onClick={() => deletePolicy(policy.id)}
                    className="p-2 text-slate-400 hover:text-red-400 transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-900 border border-white/10 rounded-2xl p-6 w-full max-w-md"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Create New Policy</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Policy Name</label>
                <input
                  type="text"
                  value={newPolicy.name}
                  onChange={(e) => setNewPolicy({ ...newPolicy, name: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  placeholder="Enter policy name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Policy Type</label>
                <select
                  value={newPolicy.type}
                  onChange={(e) => setNewPolicy({ ...newPolicy, type: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                >
                  <option value="type_a">Type A - Strict</option>
                  <option value="type_b">Type B - Moderate</option>
                  <option value="type_c">Type C - Relaxed</option>
                </select>
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
                onClick={createPolicy}
                disabled={!newPolicy.name}
                className="flex-1 px-4 py-2 bg-purple-500 rounded-lg font-medium text-white hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Create Policy
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
