"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  ScrollText,
  ArrowLeft,
  Download,
  Filter,
  Search,
  ChevronLeft,
  ChevronRight,
  User,
  Clock
} from "lucide-react";

interface AuditLog {
  id: number;
  action: string;
  entity_type: string;
  entity_id: number | null;
  description: string | null;
  changes: Record<string, unknown> | null;
  metadata: Record<string, unknown> | null;
  ip_address: string | null;
  user: { id: number; name: string; email: string } | null;
  created_at: string;
}

export default function AuditPage() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [entityTypeFilter, setEntityTypeFilter] = useState("");
  const [actionFilter, setActionFilter] = useState("");

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  useEffect(() => {
    fetchLogs();
  }, [page, entityTypeFilter, actionFilter]);

  async function fetchLogs() {
    try {
      const token = localStorage.getItem("access_token");
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: "20"
      });
      if (entityTypeFilter) params.append("entity_type", entityTypeFilter);
      if (actionFilter) params.append("action", actionFilter);

      const res = await fetch(`${apiUrl}/api/governance/audit?${params}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs);
        setTotalPages(data.pages);
      }
    } catch (error) {
      console.error("Failed to fetch audit logs:", error);
    } finally {
      setLoading(false);
    }
  }

  async function exportLogs(format: "json" | "csv") {
    try {
      const token = localStorage.getItem("access_token");
      const params = new URLSearchParams({ format });
      if (entityTypeFilter) params.append("entity_type", entityTypeFilter);
      if (actionFilter) params.append("action", actionFilter);

      const res = await fetch(`${apiUrl}/api/governance/audit/export?${params}`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        if (format === "csv") {
          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "audit_logs.csv";
          a.click();
        } else {
          const data = await res.json();
          const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "audit_logs.json";
          a.click();
        }
      }
    } catch (error) {
      console.error("Failed to export logs:", error);
    }
  }

  const actionColors: Record<string, string> = {
    create: "bg-emerald-500/20 text-emerald-400",
    update: "bg-blue-500/20 text-blue-400",
    delete: "bg-red-500/20 text-red-400",
    approve: "bg-green-500/20 text-green-400",
    reject: "bg-orange-500/20 text-orange-400"
  };

  const entityTypes = ["policy", "approval", "pattern", "knowledge"];
  const actions = ["create", "update", "delete", "approve", "reject"];

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
              <ScrollText className="w-7 h-7 text-cyan-400" />
              Audit Logs
            </h1>
            <p className="text-slate-400 mt-1">View and export governance audit trail</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => exportLogs("csv")}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button
            onClick={() => exportLogs("json")}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <span className="text-slate-400 text-sm">Filters:</span>
        </div>
        <select
          value={entityTypeFilter}
          onChange={(e) => {
            setEntityTypeFilter(e.target.value);
            setPage(1);
          }}
          className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
        >
          <option value="">All Entities</option>
          {entityTypes.map((type) => (
            <option key={type} value={type}>
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </option>
          ))}
        </select>
        <select
          value={actionFilter}
          onChange={(e) => {
            setActionFilter(e.target.value);
            setPage(1);
          }}
          className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
        >
          <option value="">All Actions</option>
          {actions.map((action) => (
            <option key={action} value={action}>
              {action.charAt(0).toUpperCase() + action.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {/* Logs List */}
      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-4 animate-pulse">
              <div className="h-5 bg-white/10 rounded w-1/3 mb-2" />
              <div className="h-4 bg-white/10 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : logs.length === 0 ? (
        <div className="text-center py-12">
          <ScrollText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">No audit logs found.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {logs.map((log, index) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.03 }}
              className="bg-white/5 border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-all"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${actionColors[log.action] || "bg-slate-500/20 text-slate-400"}`}>
                      {log.action.toUpperCase()}
                    </span>
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-white/10 text-slate-300">
                      {log.entity_type}
                    </span>
                    {log.entity_id && (
                      <span className="text-slate-500 text-xs">ID: {log.entity_id}</span>
                    )}
                  </div>
                  <p className="text-white">{log.description || `${log.action} ${log.entity_type}`}</p>
                  <div className="flex items-center gap-4 mt-2 text-sm text-slate-400">
                    {log.user && (
                      <div className="flex items-center gap-1">
                        <User className="w-3 h-3" />
                        {log.user.name || log.user.email}
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(log.created_at).toLocaleString()}
                    </div>
                    {log.ip_address && (
                      <span className="text-slate-500">IP: {log.ip_address}</span>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 mt-8">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="flex items-center gap-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </button>
          <span className="text-slate-400">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="flex items-center gap-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
