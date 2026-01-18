'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Shield,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  ToggleLeft,
  ToggleRight,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  FileText,
  DollarSign,
  ExternalLink,
  Loader2,
} from 'lucide-react';
import { policyTemplates, conditionOperators } from '@/lib/data/demo-data';
import type { Policy, PolicyRule, Condition, ActionType, Decision } from '@/lib/demo/types';

const actionTypes: ActionType[] = ['api_call', 'db_query', 'file_access', 'external'];
const effects: Decision[] = ['allow', 'block', 'require_approval'];

const templateIcons: Record<string, React.ElementType> = {
  DollarSign,
  Shield,
  FileX: FileText,
  ExternalLink,
};

export function PoliciesSection() {
  const [policies, setPolicies] = React.useState<Policy[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [expandedPolicy, setExpandedPolicy] = React.useState<string | null>(null);
  const [isCreating, setIsCreating] = React.useState(false);
  const [editingPolicy, setEditingPolicy] = React.useState<Policy | null>(null);

  // New policy form state
  const [newPolicy, setNewPolicy] = React.useState({
    name: '',
    description: '',
    actionType: 'api_call' as ActionType,
    targetPattern: '',
    effect: 'block' as Decision,
    conditions: [] as Condition[],
  });

  // Fetch policies
  const fetchPolicies = React.useCallback(async () => {
    try {
      const response = await fetch('/api/demo/policies');
      const data = await response.json();
      if (data.success) {
        setPolicies(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch policies:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchPolicies();
  }, [fetchPolicies]);

  // Toggle policy enabled state
  const togglePolicy = async (id: string) => {
    try {
      const response = await fetch(`/api/demo/policies?id=${id}`, { method: 'PATCH' });
      const data = await response.json();
      if (data.success) {
        setPolicies((prev) =>
          prev.map((p) => (p.id === id ? { ...p, enabled: data.data.enabled } : p))
        );
      }
    } catch (error) {
      console.error('Failed to toggle policy:', error);
    }
  };

  // Delete policy
  const deletePolicy = async (id: string) => {
    try {
      const response = await fetch(`/api/demo/policies?id=${id}`, { method: 'DELETE' });
      const data = await response.json();
      if (data.success) {
        setPolicies((prev) => prev.filter((p) => p.id !== id));
      }
    } catch (error) {
      console.error('Failed to delete policy:', error);
    }
  };

  // Create policy from template
  const createFromTemplate = (template: typeof policyTemplates[0]) => {
    setNewPolicy({
      name: template.name,
      description: template.description,
      actionType: template.baseRule.actionType,
      targetPattern: template.baseRule.targetPattern || '',
      effect: template.baseRule.effect,
      conditions: template.baseRule.conditions || [],
    });
    setIsCreating(true);
  };

  // Add condition
  const addCondition = () => {
    setNewPolicy((prev) => ({
      ...prev,
      conditions: [
        ...prev.conditions,
        { field: 'amount', operator: 'gt', value: 0 },
      ],
    }));
  };

  // Remove condition
  const removeCondition = (index: number) => {
    setNewPolicy((prev) => ({
      ...prev,
      conditions: prev.conditions.filter((_, i) => i !== index),
    }));
  };

  // Update condition
  const updateCondition = (index: number, field: keyof Condition, value: unknown) => {
    setNewPolicy((prev) => ({
      ...prev,
      conditions: prev.conditions.map((c, i) =>
        i === index ? { ...c, [field]: value } : c
      ),
    }));
  };

  // Submit new policy
  const submitPolicy = async () => {
    if (!newPolicy.name) return;

    const rule: PolicyRule = {
      id: crypto.randomUUID(),
      actionType: newPolicy.actionType,
      targetPattern: newPolicy.targetPattern || undefined,
      conditions: newPolicy.conditions.length > 0 ? newPolicy.conditions : undefined,
      effect: newPolicy.effect,
    };

    try {
      const response = await fetch('/api/demo/policies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newPolicy.name,
          description: newPolicy.description,
          rules: [rule],
        }),
      });

      const data = await response.json();
      if (data.success) {
        setPolicies((prev) => [...prev, data.data]);
        setIsCreating(false);
        setNewPolicy({
          name: '',
          description: '',
          actionType: 'api_call',
          targetPattern: '',
          effect: 'block',
          conditions: [],
        });
      }
    } catch (error) {
      console.error('Failed to create policy:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Policy Templates */}
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="h-5 w-5 text-primary" />
          Quick Start Templates
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {policyTemplates.map((template) => {
            const Icon = templateIcons[template.icon] || Shield;
            return (
              <motion.button
                key={template.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => createFromTemplate(template)}
                className="text-left rounded-xl border border-border/50 bg-card/30 p-4 backdrop-blur-sm hover:bg-secondary/50 transition-colors"
              >
                <div className="flex items-center gap-3 mb-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20">
                    <Icon className="h-4 w-4 text-primary" />
                  </div>
                  <span className="font-medium text-foreground">{template.name}</span>
                </div>
                <p className="text-xs text-muted-foreground">{template.description}</p>
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Create New Policy Form */}
      <AnimatePresence>
        {isCreating && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="rounded-2xl border border-primary/30 bg-primary/5 p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Plus className="h-5 w-5 text-primary" />
                  Create New Policy
                </h3>
                <button
                  onClick={() => setIsCreating(false)}
                  className="p-2 rounded-lg hover:bg-secondary/50 transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left Column */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                      Policy Name
                    </label>
                    <input
                      type="text"
                      value={newPolicy.name}
                      onChange={(e) => setNewPolicy((p) => ({ ...p, name: e.target.value }))}
                      placeholder="e.g., Payment Threshold"
                      className="w-full rounded-lg border border-border/50 bg-secondary/30 px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                      Description
                    </label>
                    <input
                      type="text"
                      value={newPolicy.description}
                      onChange={(e) => setNewPolicy((p) => ({ ...p, description: e.target.value }))}
                      placeholder="Describe what this policy does"
                      className="w-full rounded-lg border border-border/50 bg-secondary/30 px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                      Action Type
                    </label>
                    <select
                      value={newPolicy.actionType}
                      onChange={(e) => setNewPolicy((p) => ({ ...p, actionType: e.target.value as ActionType }))}
                      className="w-full rounded-lg border border-border/50 bg-secondary/30 px-4 py-2.5 text-sm text-foreground focus:border-primary focus:outline-none"
                    >
                      {actionTypes.map((type) => (
                        <option key={type} value={type}>
                          {type.replace('_', ' ')}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                      Target Pattern (Regex)
                    </label>
                    <input
                      type="text"
                      value={newPolicy.targetPattern}
                      onChange={(e) => setNewPolicy((p) => ({ ...p, targetPattern: e.target.value }))}
                      placeholder="e.g., .*stripe.*"
                      className="w-full rounded-lg border border-border/50 bg-secondary/30 px-4 py-2.5 text-sm text-foreground font-mono placeholder:text-muted-foreground focus:border-primary focus:outline-none"
                    />
                  </div>
                </div>

                {/* Right Column */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-2">
                      Effect
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {effects.map((effect) => (
                        <button
                          key={effect}
                          onClick={() => setNewPolicy((p) => ({ ...p, effect }))}
                          className={`rounded-lg border px-3 py-2 text-sm transition-colors ${
                            newPolicy.effect === effect
                              ? effect === 'allow'
                                ? 'border-green-500 bg-green-500/20 text-green-400'
                                : effect === 'block'
                                ? 'border-red-500 bg-red-500/20 text-red-400'
                                : 'border-yellow-500 bg-yellow-500/20 text-yellow-400'
                              : 'border-border/50 bg-secondary/30 text-muted-foreground'
                          }`}
                        >
                          {effect.replace('_', ' ')}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium text-muted-foreground">
                        Conditions
                      </label>
                      <button
                        onClick={addCondition}
                        className="text-xs text-primary hover:underline flex items-center gap-1"
                      >
                        <Plus className="h-3 w-3" />
                        Add Condition
                      </button>
                    </div>

                    <div className="space-y-2">
                      {newPolicy.conditions.map((condition, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <input
                            type="text"
                            value={condition.field}
                            onChange={(e) => updateCondition(index, 'field', e.target.value)}
                            placeholder="field"
                            className="flex-1 rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none"
                          />
                          <select
                            value={condition.operator}
                            onChange={(e) => updateCondition(index, 'operator', e.target.value)}
                            className="rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none"
                          >
                            {conditionOperators.map((op) => (
                              <option key={op.value} value={op.value}>
                                {op.symbol}
                              </option>
                            ))}
                          </select>
                          <input
                            type="text"
                            value={String(condition.value)}
                            onChange={(e) => {
                              const val = isNaN(Number(e.target.value))
                                ? e.target.value
                                : Number(e.target.value);
                              updateCondition(index, 'value', val);
                            }}
                            placeholder="value"
                            className="w-24 rounded-lg border border-border/50 bg-secondary/30 px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none"
                          />
                          <button
                            onClick={() => removeCondition(index)}
                            className="p-2 rounded-lg hover:bg-red-500/20 text-red-400 transition-colors"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      ))}

                      {newPolicy.conditions.length === 0 && (
                        <p className="text-xs text-muted-foreground italic">
                          No conditions - policy will match all actions of the selected type
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-3">
                <button
                  onClick={() => setIsCreating(false)}
                  className="px-4 py-2 rounded-lg border border-border/50 text-muted-foreground hover:bg-secondary/50 transition-colors text-sm"
                >
                  Cancel
                </button>
                <button
                  onClick={submitPolicy}
                  disabled={!newPolicy.name}
                  className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  Create Policy
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Existing Policies */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Active Policies</h3>
          <button
            onClick={() => setIsCreating(true)}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <Plus className="h-4 w-4" />
            New Policy
          </button>
        </div>

        <div className="space-y-3">
          {policies.map((policy) => (
            <motion.div
              key={policy.id}
              layout
              className={`rounded-xl border bg-card/30 backdrop-blur-sm overflow-hidden ${
                policy.enabled ? 'border-border/50' : 'border-border/30 opacity-60'
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between p-4">
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setExpandedPolicy(expandedPolicy === policy.id ? null : policy.id)}
                    className="p-1 rounded hover:bg-secondary/50 transition-colors"
                  >
                    {expandedPolicy === policy.id ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>
                  <div>
                    <p className="font-medium text-foreground">{policy.name}</p>
                    <p className="text-xs text-muted-foreground">{policy.description}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-xs text-muted-foreground">
                    {policy.rules.length} rule{policy.rules.length !== 1 ? 's' : ''}
                  </span>
                  <button
                    onClick={() => togglePolicy(policy.id)}
                    className="p-1 rounded hover:bg-secondary/50 transition-colors"
                  >
                    {policy.enabled ? (
                      <ToggleRight className="h-6 w-6 text-green-400" />
                    ) : (
                      <ToggleLeft className="h-6 w-6 text-muted-foreground" />
                    )}
                  </button>
                  <button
                    onClick={() => deletePolicy(policy.id)}
                    className="p-1 rounded hover:bg-red-500/20 text-red-400 transition-colors"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {/* Expanded Details */}
              <AnimatePresence>
                {expandedPolicy === policy.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-border/30 bg-secondary/20 overflow-hidden"
                  >
                    <div className="p-4 space-y-3">
                      {policy.rules.map((rule, index) => (
                        <div
                          key={rule.id}
                          className="rounded-lg border border-border/30 bg-card/50 p-3"
                        >
                          <div className="flex items-start justify-between">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <span className="text-xs bg-secondary px-2 py-0.5 rounded text-muted-foreground">
                                  Rule {index + 1}
                                </span>
                                {rule.actionType && (
                                  <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">
                                    {rule.actionType}
                                  </span>
                                )}
                              </div>
                              {rule.targetPattern && (
                                <p className="text-xs text-muted-foreground font-mono">
                                  Target: {rule.targetPattern}
                                </p>
                              )}
                              {rule.conditions && rule.conditions.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {rule.conditions.map((c, i) => (
                                    <span
                                      key={i}
                                      className="text-xs bg-secondary px-2 py-0.5 rounded text-muted-foreground"
                                    >
                                      {c.field} {conditionOperators.find((o) => o.value === c.operator)?.symbol} {c.value}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                            <span
                              className={`text-xs px-2 py-1 rounded-full ${
                                rule.effect === 'allow'
                                  ? 'bg-green-500/20 text-green-400'
                                  : rule.effect === 'block'
                                  ? 'bg-red-500/20 text-red-400'
                                  : 'bg-yellow-500/20 text-yellow-400'
                              }`}
                            >
                              {rule.effect}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}

          {policies.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No policies yet</p>
              <p className="text-xs">Create one from a template above or click New Policy</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
