import { NextRequest, NextResponse } from 'next/server';
import { demoStore } from '@/lib/demo/store';
import { evaluateAction } from '@/lib/demo/policy-engine';
import type { ActionRequest, Action, ApprovalRequest } from '@/lib/demo/types';

function generateId(): string {
  return crypto.randomUUID();
}

// GET - List actions
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const limit = parseInt(searchParams.get('limit') || '50', 10);
  const agentId = searchParams.get('agentId') || undefined;
  const decision = searchParams.get('decision') || undefined;

  const actions = demoStore.getActions(limit, { agentId, decision });

  return NextResponse.json({
    success: true,
    data: actions,
    total: actions.length,
  });
}

// POST - Fire a new action
export async function POST(request: NextRequest) {
  try {
    const body: ActionRequest = await request.json();

    // Validate required fields
    if (!body.agentId || !body.actionType || !body.target) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: agentId, actionType, target' },
        { status: 400 }
      );
    }

    // Get agent info
    const agent = demoStore.getAgent(body.agentId);
    if (!agent) {
      return NextResponse.json(
        { success: false, error: `Agent not found: ${body.agentId}` },
        { status: 404 }
      );
    }

    // Evaluate the action against policies
    const startTime = Date.now();
    const evaluation = evaluateAction(body);
    const duration = Date.now() - startTime;

    // Create action record
    const action: Action = {
      id: generateId(),
      agentId: body.agentId,
      agentName: agent.name,
      actionType: body.actionType,
      target: body.target,
      params: body.params || {},
      decision: evaluation.decision,
      reason: evaluation.reason,
      policyId: evaluation.matchedPolicy?.id,
      policyName: evaluation.matchedPolicy?.name,
      timestamp: new Date().toISOString(),
      duration,
    };

    // Store the action
    demoStore.addAction(action);

    // If requires approval, create approval request
    if (evaluation.decision === 'require_approval') {
      const approval: ApprovalRequest = {
        id: generateId(),
        actionId: action.id,
        action,
        requestedAt: new Date().toISOString(),
        status: 'pending',
      };
      demoStore.addApproval(approval);
    }

    return NextResponse.json({
      success: true,
      data: {
        action,
        evaluation: {
          decision: evaluation.decision,
          reason: evaluation.reason,
          policyMatched: evaluation.matchedPolicy?.name,
        },
      },
    });
  } catch (error) {
    console.error('Error processing action:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to process action' },
      { status: 500 }
    );
  }
}
