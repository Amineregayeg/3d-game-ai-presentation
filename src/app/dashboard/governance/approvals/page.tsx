"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  ClipboardCheck,
  Check,
  X,
  Clock,
  ArrowLeft,
  Filter,
  User
} from "lucide-react";

interface Approval {
  id: number;
  action_type: string;
  target: string;
  context: Record<string, unknown>;
  status: string;
  requester: { id: number; name: string; email: string } | null;
  policy: { id: number; name: string } | null;
  created_at: string;
  decided_at: string | null;
  decision_reason: string | null;
}

export default function ApprovalsPage() {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("pending");
  const [decidingId, setDecidingId] = useState<number | null>(null);
  const [decisionReason, setDecisionReason] = useState("");

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  useEffect(() => {
    fetchApprovals();
  }, [statusFilter]);

  async function fetchApprovals() {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${apiUrl}/api/governance/approvals?status=${statusFilter}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setApprovals(data.approvals);
      }
    } catch (error) {
      console.error("Failed to fetch approvals:", error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDecision(id: number, decision: "approved" | "rejected") {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${apiUrl}/api/governance/approvals/${id}/decide`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ decision, reason: decisionReason })
      });
      if (res.ok) {
        setDecidingId(null);
        setDecisionReason("");
        fetchApprovals();
      }
    } catch (error) {
      console.error("Failed to decide approval:", error);
    }
  }

  const statusColors: Record<string, string> = {
    pending: "bg-amber-500/20 text-amber-400",
    approved: "bg-emerald-500/20 text-emerald-400",
    rejected: "bg-red-500/20 text-red-400"
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
              <ClipboardCheck className="w-7 h-7 text-amber-400" />
              Approval Inbox
            </h1>
            <p className="text-slate-400 mt-1">Review and decide on pending approval requests</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2 mb-6">
        <Filter className="w-4 h-4 text-slate-400" />
        {["pending", "approved", "rejected", "all"].map((status) => (
          <button
            key={status}
            onClick={() => setStatusFilter(status)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              statusFilter === status
                ? "bg-white/10 text-white"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        ))}
      </div>

      {/* Approvals List */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-6 animate-pulse">
              <div className="h-6 bg-white/10 rounded w-1/3 mb-2" />
              <div className="h-4 bg-white/10 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : approvals.length === 0 ? (
        <div className="text-center py-12">
          <ClipboardCheck className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">
            {statusFilter === "pending"
              ? "No pending approvals. All caught up!"
              : `No ${statusFilter} approvals found.`}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {approvals.map((approval, index) => (
            <motion.div
              key={approval.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white/5 border border-white/10 rounded-xl p-6"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColors[approval.status]}`}>
                      {approval.status.toUpperCase()}
                    </span>
                    <span className="text-slate-500 text-sm">
                      #{approval.id}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-1">
                    {approval.action_type.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                  </h3>
                  <p className="text-slate-300 mb-3">{approval.target}</p>
                  <div className="flex items-center gap-4 text-sm text-slate-400">
                    {approval.requester && (
                      <div className="flex items-center gap-1">
                        <User className="w-4 h-4" />
                        {approval.requester.name || approval.requester.email}
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {new Date(approval.created_at).toLocaleString()}
                    </div>
                  </div>
                  {approval.decision_reason && (
                    <p className="mt-3 text-sm text-slate-400 italic">
                      Reason: {approval.decision_reason}
                    </p>
                  )}
                </div>
                {approval.status === "pending" && (
                  <div className="flex items-center gap-2 ml-4">
                    {decidingId === approval.id ? (
                      <div className="flex flex-col gap-2">
                        <input
                          type="text"
                          placeholder="Reason (optional)"
                          value={decisionReason}
                          onChange={(e) => setDecisionReason(e.target.value)}
                          className="px-3 py-1 bg-white/5 border border-white/10 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-white/20"
                        />
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleDecision(approval.id, "approved")}
                            className="flex items-center gap-1 px-3 py-1 bg-emerald-500 rounded text-white text-sm hover:bg-emerald-600 transition-colors"
                          >
                            <Check className="w-4 h-4" />
                            Approve
                          </button>
                          <button
                            onClick={() => handleDecision(approval.id, "rejected")}
                            className="flex items-center gap-1 px-3 py-1 bg-red-500 rounded text-white text-sm hover:bg-red-600 transition-colors"
                          >
                            <X className="w-4 h-4" />
                            Reject
                          </button>
                          <button
                            onClick={() => {
                              setDecidingId(null);
                              setDecisionReason("");
                            }}
                            className="px-3 py-1 bg-white/5 rounded text-slate-400 text-sm hover:bg-white/10 transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDecidingId(approval.id)}
                        className="px-4 py-2 bg-amber-500 rounded-lg font-medium text-white hover:bg-amber-600 transition-colors"
                      >
                        Review
                      </button>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
