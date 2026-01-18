// Policy Evaluation Engine
import type {
  ActionRequest,
  Policy,
  PolicyRule,
  Condition,
  EvaluationResult,
  Decision,
} from './types';
import { demoStore } from './store';

// Check if a condition matches
function evaluateCondition(condition: Condition, params: Record<string, unknown>): boolean {
  const value = params[condition.field];

  // If field doesn't exist in params, condition doesn't match
  if (value === undefined) return false;

  switch (condition.operator) {
    case 'eq':
      return value === condition.value;
    case 'neq':
      return value !== condition.value;
    case 'gt':
      return typeof value === 'number' && value > (condition.value as number);
    case 'lt':
      return typeof value === 'number' && value < (condition.value as number);
    case 'gte':
      return typeof value === 'number' && value >= (condition.value as number);
    case 'lte':
      return typeof value === 'number' && value <= (condition.value as number);
    case 'contains':
      return typeof value === 'string' && value.includes(condition.value as string);
    case 'matches':
      try {
        const regex = new RegExp(condition.value as string, 'i');
        return typeof value === 'string' && regex.test(value);
      } catch {
        return false;
      }
    default:
      return false;
  }
}

// Check if a rule matches an action
function ruleMatches(rule: PolicyRule, action: ActionRequest): boolean {
  // Check action type
  if (rule.actionType && rule.actionType !== action.actionType) {
    return false;
  }

  // Check target pattern
  if (rule.targetPattern) {
    try {
      const regex = new RegExp(rule.targetPattern, 'i');
      if (!regex.test(action.target)) {
        return false;
      }
    } catch {
      // Invalid regex, skip pattern matching
    }
  }

  // Check conditions
  if (rule.conditions && rule.conditions.length > 0) {
    // All conditions must match (AND logic)
    const allMatch = rule.conditions.every((condition) =>
      evaluateCondition(condition, action.params)
    );
    if (!allMatch) {
      return false;
    }
  }

  return true;
}

// Main evaluation function
export function evaluateAction(action: ActionRequest): EvaluationResult {
  const policies = demoStore.getPolicies();

  // Check each enabled policy in order
  for (const policy of policies) {
    if (!policy.enabled) continue;

    // Check each rule in the policy
    for (const rule of policy.rules) {
      if (ruleMatches(rule, action)) {
        return {
          decision: rule.effect,
          reason: generateReason(rule.effect, policy, rule, action),
          matchedPolicy: policy,
          matchedRule: rule,
        };
      }
    }
  }

  // Default: allow if no policy matches
  return {
    decision: 'allow',
    reason: 'No matching policy found - action allowed by default',
  };
}

// Generate human-readable reason
function generateReason(
  decision: Decision,
  policy: Policy,
  rule: PolicyRule,
  action: ActionRequest
): string {
  const actionDesc = `${action.actionType} to ${truncate(action.target, 50)}`;

  switch (decision) {
    case 'allow':
      return `Allowed by policy "${policy.name}"`;
    case 'block':
      if (rule.targetPattern) {
        return `Blocked by policy "${policy.name}" - target matches restricted pattern`;
      }
      if (rule.conditions?.length) {
        const conditionDesc = rule.conditions
          .map((c) => `${c.field} ${c.operator} ${c.value}`)
          .join(', ');
        return `Blocked by policy "${policy.name}" - ${conditionDesc}`;
      }
      return `Blocked by policy "${policy.name}"`;
    case 'require_approval':
      if (rule.conditions?.length) {
        const conditionDesc = rule.conditions
          .map((c) => `${c.field} ${c.operator} ${c.value}`)
          .join(', ');
        return `Requires approval - ${conditionDesc} (Policy: ${policy.name})`;
      }
      return `Requires approval per policy "${policy.name}"`;
    default:
      return 'Unknown decision';
  }
}

// Helper to truncate strings
function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength - 3) + '...';
}

// Dry-run evaluation (for policy testing)
export function dryRunEvaluation(
  action: ActionRequest,
  customPolicies?: Policy[]
): EvaluationResult {
  const originalPolicies = demoStore.getPolicies();

  // Temporarily use custom policies if provided
  if (customPolicies) {
    // Create a mock evaluation
    for (const policy of customPolicies) {
      if (!policy.enabled) continue;

      for (const rule of policy.rules) {
        if (ruleMatches(rule, action)) {
          return {
            decision: rule.effect,
            reason: generateReason(rule.effect, policy, rule, action),
            matchedPolicy: policy,
            matchedRule: rule,
          };
        }
      }
    }

    return {
      decision: 'allow',
      reason: 'No matching policy found - action allowed by default',
    };
  }

  // Use existing policies
  return evaluateAction(action);
}

// Get policy coverage stats
export function getPolicyCoverage(): Record<string, number> {
  const policies = demoStore.getPolicies();
  const coverage: Record<string, number> = {
    api_call: 0,
    db_query: 0,
    file_access: 0,
    external: 0,
  };

  for (const policy of policies) {
    if (!policy.enabled) continue;

    for (const rule of policy.rules) {
      if (rule.actionType) {
        coverage[rule.actionType]++;
      } else {
        // Rule applies to all action types
        Object.keys(coverage).forEach((key) => coverage[key]++);
      }
    }
  }

  return coverage;
}
