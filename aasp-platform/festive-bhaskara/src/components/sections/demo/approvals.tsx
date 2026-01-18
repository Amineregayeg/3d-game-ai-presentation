'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Clock,
  CheckCircle,
  XCircle,
  Bell,
  RefreshCw,
  Zap,
  FileText,
  BarChart3,
  MessageSquare,
  Code,
  Shield,
  Globe,
  Database,
  ExternalLink,
  ChevronRight,
  User,
  Loader2,
} from 'lucide-react';
import { demoAgents, approvalStatusStyles } from '@/lib/data/demo-data';
import type { ApprovalRequest, ApprovalStatus } from '@/lib/demo/types';

const iconMap: Record<string, React.ElementType> = {
  FileText,
  BarChart3,
  MessageSquare,
  Code,
  Shield,
  Globe,
  Database,
  ExternalLink,
};

export function ApprovalsSection() {
  const [approvals, setApprovals] = React.useState<ApprovalRequest[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [filter, setFilter] = React.useState<'all' | ApprovalStatus>('all');
  const [processingId, setProcessingId] = React.useState<string | null>(null);
  const [selectedApproval, setSelectedApproval] = React.useState<ApprovalRequest | null>(null);
  const [rejectReason, setRejectReason] = React.useState('');

  // Fetch approvals
  const fetchApprovals = React.useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (filter !== 'all') params.append('status', filter);

      const response = await fetch(`/api/demo/approvals?${params}`);
      const data = await response.json();
      if (data.success) {
        setApprovals(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch approvals:', error);
    } finally {
      setIsLoading(false);
    }
  }, [filter]);

  React.useEffect(() => {
    fetchApprovals();
    const interval = setInterval(fetchApprovals, 5000);
    return () => clearInterval(interval);
  }, [fetchApprovals]);

  // Handle approval decision
  const handleDecision = async (id: string, decision: 'approved' | 'rejected', reason?: string) => {
    setProcessingId(id);
    try {
      const response = await fetch('/api/demo/approvals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id,
          decision,
          decidedBy: 'Demo User',
          reason,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setApprovals((prev) =>
          prev.map((a) => (a.id === id ? data.data : a))
        );
        setSelectedApproval(null);
        setRejectReason('');
      }
    } catch (error) {
      console.error('Failed to resolve approval:', error);
    } finally {
      setProcessingId(null);
    }
  };

  const pendingCount = approvals.filter((a) => a.status === 'pending').length;

  const formatTime = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          {
            label: 'Pending',
            value: pendingCount,
            color: 'text-yellow-400',
            bg: 'bg-yellow-500/10',
            icon: Clock,
          },
          {
            label: 'Approved Today',
            value: approvals.filter(
              (a) =>
                a.status === 'approved' &&
                new Date(a.decidedAt || '').toDateString() === new Date().toDateString()
            ).length,
            color: 'text-green-400',
            bg: 'bg-green-500/10',
            icon: CheckCircle,
          },
          {
            label: 'Rejected Today',
            value: approvals.filter(
              (a) =>
                a.status === 'rejected' &&
                new Date(a.decidedAt || '').toDateString() === new Date().toDateString()
            ).length,
            color: 'text-red-400',
            bg: 'bg-red-500/10',
            icon: XCircle,
          },
        ].map((stat) => (
          <div
            key={stat.label}
            className={`rounded-xl border border-border/50 ${stat.bg} p-4 backdrop-blur-sm`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-3xl font-bold ${stat.color}`}>{stat.value}</p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </div>
              <stat.icon className={`h-8 w-8 ${stat.color} opacity-50`} />
            </div>
          </div>
        ))}
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2">
        {(['all', 'pending', 'approved', 'rejected'] as const).map((status) => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === status
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary/50 text-muted-foreground hover:bg-secondary'
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
            {status === 'pending' && pendingCount > 0 && (
              <span className="ml-2 bg-yellow-500/30 text-yellow-400 px-1.5 py-0.5 rounded text-xs">
                {pendingCount}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Approvals List */}
      <div className="space-y-3">
        {approvals.length === 0 ? (
          <div className="text-center py-12 rounded-2xl border border-border/50 bg-card/30 backdrop-blur-sm">
            <Bell className="h-8 w-8 mx-auto mb-2 text-muted-foreground opacity-50" />
            <p className="text-sm text-muted-foreground">No approval requests</p>
            <p className="text-xs text-muted-foreground">
              Fire actions that require approval in the Simulator tab
            </p>
          </div>
        ) : (
          <AnimatePresence>
            {approvals.map((approval) => {
              const agent = demoAgents.find((a) => a.id === approval.action.agentId);
              const AgentIcon = agent ? iconMap[agent.icon] || Zap : Zap;
              const statusStyle = approvalStatusStyles[approval.status];
              const isPending = approval.status === 'pending';

              return (
                <motion.div
                  key={approval.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`rounded-xl border bg-card/30 backdrop-blur-sm overflow-hidden ${
                    isPending ? 'border-yellow-500/30' : 'border-border/50'
                  }`}
                >
                  <div className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      {/* Left: Action Details */}
                      <div className="flex items-start gap-4 flex-1 min-w-0">
                        <div
                          className="flex h-12 w-12 items-center justify-center rounded-xl shrink-0"
                          style={{ backgroundColor: `${agent?.color || '#666'}20` }}
                        >
                          <AgentIcon
                            className="h-6 w-6"
                            style={{ color: agent?.color || '#666' }}
                          />
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <p className="font-semibold text-foreground">
                              {approval.action.agentName}
                            </p>
                            <span
                              className={`px-2 py-0.5 rounded-full text-xs ${statusStyle.bg} ${statusStyle.text}`}
                            >
                              {statusStyle.label}
                            </span>
                          </div>

                          <p className="text-sm text-muted-foreground mb-2">
                            {approval.action.reason}
                          </p>

                          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                            <span className="bg-secondary px-2 py-0.5 rounded">
                              {approval.action.actionType}
                            </span>
                            <span className="font-mono truncate max-w-xs">
                              {approval.action.target}
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatTime(approval.requestedAt)}
                            </span>
                          </div>

                          {approval.decidedAt && (
                            <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                              <User className="h-3 w-3" />
                              <span>
                                {approval.status} by {approval.decidedBy}
                              </span>
                              {approval.reason && (
                                <>
                                  <span>-</span>
                                  <span className="italic">{approval.reason}</span>
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Right: Actions */}
                      {isPending && (
                        <div className="flex items-center gap-2 shrink-0">
                          <button
                            onClick={() => setSelectedApproval(approval)}
                            className="px-3 py-1.5 rounded-lg border border-red-500/50 text-red-400 text-sm font-medium hover:bg-red-500/10 transition-colors"
                          >
                            Reject
                          </button>
                          <button
                            onClick={() => handleDecision(approval.id, 'approved')}
                            disabled={processingId === approval.id}
                            className="px-3 py-1.5 rounded-lg bg-green-500 text-white text-sm font-medium hover:bg-green-600 transition-colors disabled:opacity-50 flex items-center gap-1"
                          >
                            {processingId === approval.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <CheckCircle className="h-4 w-4" />
                            )}
                            Approve
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Expandable Details */}
                  <button
                    onClick={() => setSelectedApproval(approval)}
                    className="w-full px-4 py-2 border-t border-border/30 bg-secondary/20 flex items-center justify-center gap-1 text-xs text-muted-foreground hover:bg-secondary/40 transition-colors"
                  >
                    View Details
                    <ChevronRight className="h-3 w-3" />
                  </button>
                </motion.div>
              );
            })}
          </AnimatePresence>
        )}
      </div>

      {/* Detail/Reject Modal */}
      <AnimatePresence>
        {selectedApproval && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => {
              setSelectedApproval(null);
              setRejectReason('');
            }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-lg rounded-2xl border border-border/50 bg-card p-6 shadow-xl"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Approval Request</h3>
                  <p className="text-sm text-muted-foreground">
                    From {selectedApproval.action.agentName}
                  </p>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-sm ${
                    approvalStatusStyles[selectedApproval.status].bg
                  } ${approvalStatusStyles[selectedApproval.status].text}`}
                >
                  {approvalStatusStyles[selectedApproval.status].label}
                </span>
              </div>

              <div className="space-y-4 mb-6">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Reason for Approval</p>
                  <p className="text-sm text-foreground">{selectedApproval.action.reason}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Action Type</p>
                    <p className="text-sm text-foreground">{selectedApproval.action.actionType}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Requested</p>
                    <p className="text-sm text-foreground">
                      {new Date(selectedApproval.requestedAt).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Target</p>
                  <p className="text-sm text-foreground font-mono bg-secondary/50 p-2 rounded break-all">
                    {selectedApproval.action.target}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Parameters</p>
                  <pre className="text-xs text-foreground font-mono bg-secondary/50 p-2 rounded overflow-auto max-h-32">
                    {JSON.stringify(selectedApproval.action.params, null, 2)}
                  </pre>
                </div>

                {selectedApproval.status === 'pending' && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Rejection Reason (Optional)</p>
                    <textarea
                      value={rejectReason}
                      onChange={(e) => setRejectReason(e.target.value)}
                      placeholder="Enter reason for rejection..."
                      rows={2}
                      className="w-full rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none resize-none"
                    />
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-3">
                <button
                  onClick={() => {
                    setSelectedApproval(null);
                    setRejectReason('');
                  }}
                  className="px-4 py-2 rounded-lg bg-secondary text-foreground text-sm font-medium hover:bg-secondary/80 transition-colors"
                >
                  Close
                </button>

                {selectedApproval.status === 'pending' && (
                  <>
                    <button
                      onClick={() =>
                        handleDecision(selectedApproval.id, 'rejected', rejectReason || undefined)
                      }
                      disabled={processingId === selectedApproval.id}
                      className="px-4 py-2 rounded-lg border border-red-500/50 text-red-400 text-sm font-medium hover:bg-red-500/10 transition-colors disabled:opacity-50"
                    >
                      Reject
                    </button>
                    <button
                      onClick={() => handleDecision(selectedApproval.id, 'approved')}
                      disabled={processingId === selectedApproval.id}
                      className="px-4 py-2 rounded-lg bg-green-500 text-white text-sm font-medium hover:bg-green-600 transition-colors disabled:opacity-50 flex items-center gap-1"
                    >
                      {processingId === selectedApproval.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <CheckCircle className="h-4 w-4" />
                      )}
                      Approve
                    </button>
                  </>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
