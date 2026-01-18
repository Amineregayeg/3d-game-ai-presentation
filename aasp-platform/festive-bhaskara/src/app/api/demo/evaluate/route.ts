import { NextRequest, NextResponse } from 'next/server';
import { dryRunEvaluation } from '@/lib/demo/policy-engine';
import type { ActionRequest } from '@/lib/demo/types';

// POST - Dry-run evaluate an action against policies
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.action || !body.action.agentId || !body.action.actionType || !body.action.target) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields in action: agentId, actionType, target' },
        { status: 400 }
      );
    }

    const action: ActionRequest = {
      agentId: body.action.agentId,
      actionType: body.action.actionType,
      target: body.action.target,
      params: body.action.params || {},
    };

    // Evaluate against existing policies or custom policies if provided
    const result = dryRunEvaluation(action, body.policies);

    return NextResponse.json({
      success: true,
      data: {
        decision: result.decision,
        reason: result.reason,
        matchedPolicy: result.matchedPolicy
          ? {
              id: result.matchedPolicy.id,
              name: result.matchedPolicy.name,
            }
          : null,
        matchedRule: result.matchedRule
          ? {
              id: result.matchedRule.id,
              effect: result.matchedRule.effect,
              conditions: result.matchedRule.conditions,
            }
          : null,
      },
    });
  } catch (error) {
    console.error('Error evaluating action:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to evaluate action' },
      { status: 500 }
    );
  }
}
