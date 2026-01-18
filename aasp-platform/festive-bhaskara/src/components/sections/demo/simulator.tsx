'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Zap,
  Send,
  CheckCircle,
  XCircle,
  Clock,
  ChevronDown,
  FileText,
  BarChart3,
  MessageSquare,
  Code,
  Shield,
  Globe,
  Database,
  ExternalLink,
  Loader2,
} from 'lucide-react';
import { demoAgents, actionTemplates, decisionStyles } from '@/lib/data/demo-data';
import type { Action, ActionType, Decision } from '@/lib/demo/types';

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

export function SimulatorSection() {
  const [selectedAgent, setSelectedAgent] = React.useState(demoAgents[0]);
  const [selectedActionType, setSelectedActionType] = React.useState<ActionType>('api_call');
  const [target, setTarget] = React.useState('');
  const [params, setParams] = React.useState('{}');
  const [isLoading, setIsLoading] = React.useState(false);
  const [result, setResult] = React.useState<Action | null>(null);
  const [showAgentDropdown, setShowAgentDropdown] = React.useState(false);
  const [recentActions, setRecentActions] = React.useState<Action[]>([]);

  // Load recent actions on mount
  React.useEffect(() => {
    fetchRecentActions();
  }, []);

  const fetchRecentActions = async () => {
    try {
      const response = await fetch('/api/demo/actions?limit=5');
      const data = await response.json();
      if (data.success) {
        setRecentActions(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch actions:', error);
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setResult(null);

    try {
      let parsedParams = {};
      try {
        parsedParams = JSON.parse(params);
      } catch {
        // Keep empty if invalid
      }

      const response = await fetch('/api/demo/actions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agentId: selectedAgent.id,
          actionType: selectedActionType,
          target: target || actionTemplates.find(t => t.type === selectedActionType)?.placeholder || '',
          params: parsedParams,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setResult(data.data.action);
        setRecentActions((prev) => [data.data.action, ...prev.slice(0, 4)]);
      }
    } catch (error) {
      console.error('Failed to fire action:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const selectedTemplate = actionTemplates.find(t => t.type === selectedActionType);
  const AgentIcon = iconMap[selectedAgent.icon] || Zap;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Left: Action Builder */}
      <div className="space-y-6">
        <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Fire an Action
          </h3>

          {/* Agent Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-muted-foreground mb-2">
              Select Agent
            </label>
            <div className="relative">
              <button
                onClick={() => setShowAgentDropdown(!showAgentDropdown)}
                className="w-full flex items-center justify-between gap-3 rounded-lg border border-border/50 bg-secondary/30 px-4 py-3 text-left hover:bg-secondary/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-lg"
                    style={{ backgroundColor: `${selectedAgent.color}20` }}
                  >
                    <AgentIcon className="h-5 w-5" style={{ color: selectedAgent.color }} />
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{selectedAgent.name}</p>
                    <p className="text-xs text-muted-foreground">{selectedAgent.description}</p>
                  </div>
                </div>
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
              </button>

              <AnimatePresence>
                {showAgentDropdown && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute z-10 w-full mt-2 rounded-lg border border-border/50 bg-card shadow-xl overflow-hidden"
                  >
                    {demoAgents.map((agent) => {
                      const Icon = iconMap[agent.icon] || Zap;
                      return (
                        <button
                          key={agent.id}
                          onClick={() => {
                            setSelectedAgent(agent);
                            setShowAgentDropdown(false);
                          }}
                          className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-secondary/50 transition-colors ${
                            agent.id === selectedAgent.id ? 'bg-primary/10' : ''
                          }`}
                        >
                          <div
                            className="flex h-8 w-8 items-center justify-center rounded-lg"
                            style={{ backgroundColor: `${agent.color}20` }}
                          >
                            <Icon className="h-4 w-4" style={{ color: agent.color }} />
                          </div>
                          <div>
                            <p className="font-medium text-foreground text-sm">{agent.name}</p>
                            <p className="text-xs text-muted-foreground">{agent.description}</p>
                          </div>
                        </button>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Action Type */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-muted-foreground mb-2">
              Action Type
            </label>
            <div className="grid grid-cols-2 gap-2">
              {actionTemplates.map((template) => {
                const Icon = iconMap[template.icon] || Globe;
                const isActive = template.type === selectedActionType;
                return (
                  <button
                    key={template.type}
                    onClick={() => {
                      setSelectedActionType(template.type);
                      setTarget(template.placeholder);
                      setParams(JSON.stringify(template.sampleParams, null, 2));
                    }}
                    className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm transition-colors ${
                      isActive
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border/50 bg-secondary/30 text-muted-foreground hover:bg-secondary/50'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {template.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Target */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-muted-foreground mb-2">
              Target
            </label>
            <input
              type="text"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder={selectedTemplate?.placeholder}
              className="w-full rounded-lg border border-border/50 bg-secondary/30 px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>

          {/* Parameters */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-muted-foreground mb-2">
              Parameters (JSON)
            </label>
            <textarea
              value={params}
              onChange={(e) => setParams(e.target.value)}
              rows={4}
              className="w-full rounded-lg border border-border/50 bg-secondary/30 px-4 py-2.5 text-sm text-foreground font-mono focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Evaluating...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Fire Action
              </>
            )}
          </button>
        </div>
      </div>

      {/* Right: Result & Recent Actions */}
      <div className="space-y-6">
        {/* Result Display */}
        <AnimatePresence mode="wait">
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`rounded-2xl border p-6 backdrop-blur-sm ${
                decisionStyles[result.decision].border
              } ${decisionStyles[result.decision].bg}`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  {React.createElement(decisionIcons[result.decision], {
                    className: `h-8 w-8 ${decisionStyles[result.decision].text}`,
                  })}
                  <div>
                    <p className={`text-lg font-bold ${decisionStyles[result.decision].text}`}>
                      {result.decision === 'allow'
                        ? 'Action Allowed'
                        : result.decision === 'block'
                        ? 'Action Blocked'
                        : 'Approval Required'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Evaluated in {result.duration}ms
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-muted-foreground shrink-0">Reason:</span>
                  <span className="text-foreground">{result.reason}</span>
                </div>
                {result.policyName && (
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">Policy:</span>
                    <span className="bg-secondary px-2 py-1 rounded text-foreground">
                      {result.policyName}
                    </span>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">Action ID:</span>
                  <code className="text-xs bg-secondary px-2 py-1 rounded text-muted-foreground font-mono">
                    {result.id}
                  </code>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Recent Actions */}
        <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Clock className="h-5 w-5 text-muted-foreground" />
            Recent Actions
          </h3>

          {recentActions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Zap className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No actions yet. Fire your first action above!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {recentActions.map((action) => {
                const DecisionIcon = decisionIcons[action.decision];
                const agent = demoAgents.find((a) => a.id === action.agentId);
                const AgentIcon = agent ? iconMap[agent.icon] || Zap : Zap;

                return (
                  <motion.div
                    key={action.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div
                      className="flex h-8 w-8 items-center justify-center rounded-lg shrink-0"
                      style={{ backgroundColor: `${agent?.color || '#666'}20` }}
                    >
                      <AgentIcon
                        className="h-4 w-4"
                        style={{ color: agent?.color || '#666' }}
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {action.agentName}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {action.actionType}: {action.target}
                      </p>
                    </div>
                    <div
                      className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                        decisionStyles[action.decision].bg
                      } ${decisionStyles[action.decision].text}`}
                    >
                      <DecisionIcon className="h-3 w-3" />
                      {action.decision === 'require_approval' ? 'pending' : action.decision}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'Total Actions', value: recentActions.length, color: 'text-primary' },
            {
              label: 'Blocked',
              value: recentActions.filter((a) => a.decision === 'block').length,
              color: 'text-red-400',
            },
            {
              label: 'Pending',
              value: recentActions.filter((a) => a.decision === 'require_approval').length,
              color: 'text-yellow-400',
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="rounded-xl border border-border/50 bg-card/30 p-4 text-center backdrop-blur-sm"
            >
              <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
              <p className="text-xs text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
