// Harbyx @governed_action Decorator

import { getHarbyxClient } from './client';
import { harbyxConfig, evaluateLocally } from './config';
import {
  ActionType,
  Decision,
  IngestResponse,
  HarbyxError,
} from './types';

export interface GovernedActionResult<T> {
  success: boolean;
  data?: T;
  pending?: boolean;
  approvalId?: string;
  blocked?: boolean;
  reason?: string;
}

export interface GovernedActionOptions {
  actionType: ActionType;
  target?: string;
  extractTarget?: (...args: unknown[]) => string;
  onBlock?: (reason: string) => void;
  onApprovalRequired?: (approvalId: string) => void;
  timeout?: number;
}

/**
 * Decorator factory for governed actions
 *
 * Usage:
 * ```typescript
 * class SalesforceService {
 *   @governedAction({ actionType: 'api_call', target: 'salesforce:insert' })
 *   async createRecord(sobject: string, data: object) {
 *     // Implementation
 *   }
 * }
 * ```
 */
export function governedAction(options: GovernedActionOptions) {
  return function <T extends (...args: unknown[]) => Promise<unknown>>(
    target: object,
    propertyKey: string,
    descriptor: TypedPropertyDescriptor<T>
  ): TypedPropertyDescriptor<T> | void {
    const originalMethod = descriptor.value;

    if (!originalMethod) {
      return descriptor;
    }

    descriptor.value = async function (this: unknown, ...args: unknown[]) {
      // Determine the target for this specific call
      const actionTarget = options.extractTarget
        ? options.extractTarget(...args)
        : options.target || propertyKey;

      try {
        let decision: Decision;
        let approvalId: string | undefined;
        let reason: string | undefined;

        // Check if we should use mock mode or local evaluation
        if (harbyxConfig.mockMode || !harbyxConfig.apiKey) {
          decision = evaluateLocally(options.actionType, actionTarget);
          reason = `Local policy evaluation: ${decision}`;
        } else {
          // Call Harbyx API
          try {
            const result: IngestResponse = await getHarbyxClient().evaluateAction({
              agentId: harbyxConfig.defaultAgentId,
              actionType: options.actionType,
              target: actionTarget,
              params: { args: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg) },
              metadata: {
                method: propertyKey,
                timestamp: new Date().toISOString(),
              },
            });

            decision = result.decision;
            approvalId = result.approval_id;
            reason = result.reason;
          } catch (error) {
            // Fallback to local evaluation if API fails
            if (harbyxConfig.enableLocalFallback) {
              console.warn('[Harbyx] API unavailable, using local evaluation');
              decision = evaluateLocally(options.actionType, actionTarget);
              reason = 'Local fallback evaluation';
            } else {
              throw error;
            }
          }
        }

        // Handle the decision
        switch (decision) {
          case 'allow':
            // Execute the original method
            return originalMethod.apply(this, args);

          case 'block':
            options.onBlock?.(reason || 'Action blocked by policy');
            throw new HarbyxError(
              `Action blocked: ${reason || 'Policy violation'}`,
              403,
              'ACTION_BLOCKED'
            );

          case 'require_approval':
            options.onApprovalRequired?.(approvalId || 'unknown');
            return {
              pending: true,
              approvalId,
              reason: reason || 'Approval required',
            } as GovernedActionResult<unknown>;

          default:
            throw new HarbyxError(
              `Unknown decision: ${decision}`,
              500,
              'UNKNOWN_DECISION'
            );
        }
      } catch (error) {
        if (error instanceof HarbyxError) throw error;
        throw new HarbyxError(
          `Governance check failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          500,
          'GOVERNANCE_ERROR'
        );
      }
    } as T;

    return descriptor;
  };
}

/**
 * Higher-order function wrapper for governed actions (for non-class methods)
 *
 * Usage:
 * ```typescript
 * const createSalesforceRecord = withGovernance(
 *   { actionType: 'api_call', target: 'salesforce:insert' },
 *   async (sobject: string, data: object) => {
 *     // Implementation
 *   }
 * );
 * ```
 */
export function withGovernance<T extends (...args: unknown[]) => Promise<unknown>>(
  options: GovernedActionOptions,
  fn: T
): (...args: Parameters<T>) => Promise<GovernedActionResult<Awaited<ReturnType<T>>>> {
  return async function (...args: Parameters<T>): Promise<GovernedActionResult<Awaited<ReturnType<T>>>> {
    const actionTarget = options.extractTarget
      ? options.extractTarget(...args)
      : options.target || fn.name || 'anonymous';

    try {
      let decision: Decision;
      let approvalId: string | undefined;
      let reason: string | undefined;

      if (harbyxConfig.mockMode || !harbyxConfig.apiKey) {
        decision = evaluateLocally(options.actionType, actionTarget);
        reason = `Local policy evaluation: ${decision}`;
      } else {
        try {
          const result = await getHarbyxClient().evaluateAction({
            agentId: harbyxConfig.defaultAgentId,
            actionType: options.actionType,
            target: actionTarget,
            params: { args: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg) },
          });

          decision = result.decision;
          approvalId = result.approval_id;
          reason = result.reason;
        } catch (error) {
          if (harbyxConfig.enableLocalFallback) {
            decision = evaluateLocally(options.actionType, actionTarget);
            reason = 'Local fallback evaluation';
          } else {
            throw error;
          }
        }
      }

      switch (decision) {
        case 'allow':
          const data = await fn(...args);
          return { success: true, data: data as Awaited<ReturnType<T>> };

        case 'block':
          options.onBlock?.(reason || 'Action blocked');
          return {
            success: false,
            blocked: true,
            reason: reason || 'Action blocked by policy',
          };

        case 'require_approval':
          options.onApprovalRequired?.(approvalId || 'unknown');
          return {
            success: false,
            pending: true,
            approvalId,
            reason: reason || 'Approval required',
          };

        default:
          return {
            success: false,
            reason: `Unknown decision: ${decision}`,
          };
      }
    } catch (error) {
      return {
        success: false,
        reason: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  };
}

/**
 * Check if a result indicates the action is pending approval
 */
export function isPendingApproval<T>(
  result: GovernedActionResult<T>
): result is GovernedActionResult<T> & { pending: true; approvalId: string } {
  return result.pending === true && !!result.approvalId;
}

/**
 * Check if a result indicates the action was blocked
 */
export function isBlocked<T>(
  result: GovernedActionResult<T>
): result is GovernedActionResult<T> & { blocked: true; reason: string } {
  return result.blocked === true;
}

/**
 * Check if a result indicates success
 */
export function isSuccess<T>(
  result: GovernedActionResult<T>
): result is GovernedActionResult<T> & { success: true; data: T } {
  return result.success === true && 'data' in result;
}
