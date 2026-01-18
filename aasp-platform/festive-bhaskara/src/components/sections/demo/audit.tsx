'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  FileText,
  Search,
  Filter,
  CheckCircle,
  XCircle,
  Clock,
  ChevronDown,
  RefreshCw,
  Download,
  ExternalLink,
  Zap,
  BarChart3,
  MessageSquare,
  Code,
  Shield,
  Globe,
  Database,
} from 'lucide-react';
import { demoAgents, decisionStyles } from '@/lib/data/demo-data';
import type { Action, Decision } from '@/lib/demo/types';

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

const decisionIcons: Record<Decision, React.ElementType> = {
  allow: CheckCircle,
  block: XCircle,
  require_approval: Clock,
};

export function AuditSection() {
  const [actions, setActions] = React.useState<Action[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [filterAgent, setFilterAgent] = React.useState<string>('all');
  const [filterDecision, setFilterDecision] = React.useState<string>('all');
  const [showFilters, setShowFilters] = React.useState(false);
  const [selectedAction, setSelectedAction] = React.useState<Action | null>(null);
  const [isAutoRefresh, setIsAutoRefresh] = React.useState(true);

  // Fetch actions
  const fetchActions = React.useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: '50' });
      if (filterAgent !== 'all') params.append('agentId', filterAgent);
      if (filterDecision !== 'all') params.append('decision', filterDecision);

      const response = await fetch(`/api/demo/actions?${params}`);
      const data = await response.json();
      if (data.success) {
        setActions(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch actions:', error);
    } finally {
      setIsLoading(false);
    }
  }, [filterAgent, filterDecision]);

  // Initial fetch and auto-refresh
  React.useEffect(() => {
    fetchActions();

    if (isAutoRefresh) {
      const interval = setInterval(fetchActions, 3000);
      return () => clearInterval(interval);
    }
  }, [fetchActions, isAutoRefresh]);

  // Filter actions by search
  const filteredActions = React.useMemo(() => {
    if (!searchQuery) return actions;
    const query = searchQuery.toLowerCase();
    return actions.filter(
      (action) =>
        action.target.toLowerCase().includes(query) ||
        action.agentName.toLowerCase().includes(query) ||
        action.reason.toLowerCase().includes(query)
    );
  }, [actions, searchQuery]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search actions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-border/50 bg-secondary/30 pl-10 pr-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm transition-colors ${
              showFilters
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border/50 bg-secondary/30 text-muted-foreground hover:bg-secondary/50'
            }`}
          >
            <Filter className="h-4 w-4" />
            Filters
            <ChevronDown className={`h-4 w-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
          </button>

          <button
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm transition-colors ${
              isAutoRefresh
                ? 'border-green-500/50 bg-green-500/10 text-green-400'
                : 'border-border/50 bg-secondary/30 text-muted-foreground hover:bg-secondary/50'
            }`}
          >
            <RefreshCw className={`h-4 w-4 ${isAutoRefresh ? 'animate-spin' : ''}`} style={{ animationDuration: '3s' }} />
            Live
          </button>

          <button
            onClick={() => {
              const csv = actions.map(a =>
                `${a.timestamp},${a.agentName},${a.actionType},${a.target},${a.decision},${a.reason}`
              ).join('\n');
              const blob = new Blob([`Timestamp,Agent,Type,Target,Decision,Reason\n${csv}`], { type: 'text/csv' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = 'audit-log.csv';
              link.click();
            }}
            className="flex items-center gap-2 rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-muted-foreground hover:bg-secondary/50 transition-colors"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
        </div>
      </div>

      {/* Filters Panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="rounded-xl border border-border/50 bg-card/30 p-4 backdrop-blur-sm overflow-hidden"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Agent Filter */}
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-2">
                  Agent
                </label>
                <select
                  value={filterAgent}
                  onChange={(e) => setFilterAgent(e.target.value)}
                  className="w-full rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none"
                >
                  <option value="all">All Agents</option>
                  {demoAgents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Decision Filter */}
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-2">
                  Decision
                </label>
                <select
                  value={filterDecision}
                  onChange={(e) => setFilterDecision(e.target.value)}
                  className="w-full rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none"
                >
                  <option value="all">All Decisions</option>
                  <option value="allow">Allowed</option>
                  <option value="block">Blocked</option>
                  <option value="require_approval">Pending Approval</option>
                </select>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions List */}
      <div className="rounded-2xl border border-border/50 bg-card/30 backdrop-blur-sm overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-12 gap-4 p-4 border-b border-border/30 bg-secondary/20 text-xs font-medium text-muted-foreground uppercase tracking-wider">
          <div className="col-span-1">Time</div>
          <div className="col-span-2">Agent</div>
          <div className="col-span-2">Type</div>
          <div className="col-span-4">Target</div>
          <div className="col-span-2">Decision</div>
          <div className="col-span-1"></div>
        </div>

        {/* Body */}
        <div className="divide-y divide-border/30 max-h-[500px] overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-6 w-6 text-muted-foreground animate-spin" />
            </div>
          ) : filteredActions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <FileText className="h-8 w-8 mb-2 opacity-50" />
              <p className="text-sm">No actions found</p>
              <p className="text-xs">Fire some actions in the Simulator tab</p>
            </div>
          ) : (
            <AnimatePresence>
              {filteredActions.map((action, index) => {
                const agent = demoAgents.find((a) => a.id === action.agentId);
                const AgentIcon = agent ? iconMap[agent.icon] || Zap : Zap;
                const DecisionIcon = decisionIcons[action.decision];

                return (
                  <motion.div
                    key={action.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.02 }}
                    onClick={() => setSelectedAction(action)}
                    className="grid grid-cols-12 gap-4 p-4 hover:bg-secondary/30 cursor-pointer transition-colors items-center"
                  >
                    <div className="col-span-1 text-xs text-muted-foreground font-mono">
                      {formatTimestamp(action.timestamp)}
                    </div>
                    <div className="col-span-2 flex items-center gap-2">
                      <div
                        className="flex h-6 w-6 items-center justify-center rounded"
                        style={{ backgroundColor: `${agent?.color || '#666'}20` }}
                      >
                        <AgentIcon
                          className="h-3 w-3"
                          style={{ color: agent?.color || '#666' }}
                        />
                      </div>
                      <span className="text-sm text-foreground truncate">{action.agentName}</span>
                    </div>
                    <div className="col-span-2">
                      <span className="text-xs bg-secondary px-2 py-1 rounded text-muted-foreground">
                        {action.actionType}
                      </span>
                    </div>
                    <div className="col-span-4 text-sm text-muted-foreground truncate font-mono">
                      {action.target}
                    </div>
                    <div className="col-span-2">
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                          decisionStyles[action.decision].bg
                        } ${decisionStyles[action.decision].text}`}
                      >
                        <DecisionIcon className="h-3 w-3" />
                        {action.decision === 'require_approval' ? 'pending' : action.decision}
                      </span>
                    </div>
                    <div className="col-span-1 flex justify-end">
                      <ExternalLink className="h-4 w-4 text-muted-foreground opacity-50" />
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border/30 bg-secondary/20 flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Showing {filteredActions.length} of {actions.length} actions
          </p>
          <button
            onClick={fetchActions}
            className="flex items-center gap-1 text-xs text-primary hover:underline"
          >
            <RefreshCw className="h-3 w-3" />
            Refresh
          </button>
        </div>
      </div>

      {/* Action Detail Modal */}
      <AnimatePresence>
        {selectedAction && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedAction(null)}
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
                  <h3 className="text-lg font-semibold text-foreground">Action Details</h3>
                  <p className="text-sm text-muted-foreground font-mono">{selectedAction.id}</p>
                </div>
                <span
                  className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm ${
                    decisionStyles[selectedAction.decision].bg
                  } ${decisionStyles[selectedAction.decision].text}`}
                >
                  {React.createElement(decisionIcons[selectedAction.decision], { className: 'h-4 w-4' })}
                  {selectedAction.decision}
                </span>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Agent</p>
                    <p className="text-sm text-foreground">{selectedAction.agentName}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Action Type</p>
                    <p className="text-sm text-foreground">{selectedAction.actionType}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Timestamp</p>
                    <p className="text-sm text-foreground">{new Date(selectedAction.timestamp).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Duration</p>
                    <p className="text-sm text-foreground">{selectedAction.duration}ms</p>
                  </div>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Target</p>
                  <p className="text-sm text-foreground font-mono bg-secondary/50 p-2 rounded break-all">
                    {selectedAction.target}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Parameters</p>
                  <pre className="text-xs text-foreground font-mono bg-secondary/50 p-2 rounded overflow-auto max-h-32">
                    {JSON.stringify(selectedAction.params, null, 2)}
                  </pre>
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Reason</p>
                  <p className="text-sm text-foreground">{selectedAction.reason}</p>
                </div>

                {selectedAction.policyName && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Matched Policy</p>
                    <span className="text-sm bg-primary/20 text-primary px-2 py-1 rounded">
                      {selectedAction.policyName}
                    </span>
                  </div>
                )}
              </div>

              <div className="mt-6 flex justify-end">
                <button
                  onClick={() => setSelectedAction(null)}
                  className="px-4 py-2 rounded-lg bg-secondary text-foreground text-sm font-medium hover:bg-secondary/80 transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
