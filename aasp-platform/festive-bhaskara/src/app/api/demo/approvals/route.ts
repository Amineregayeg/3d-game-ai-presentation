import { NextRequest, NextResponse } from 'next/server';
import { demoStore } from '@/lib/demo/store';

// GET - List approvals
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const status = searchParams.get('status') as 'pending' | 'approved' | 'rejected' | null;

  const approvals = demoStore.getApprovals(status || undefined);

  return NextResponse.json({
    success: true,
    data: approvals,
    total: approvals.length,
    pending: demoStore.getApprovals('pending').length,
  });
}

// POST - Resolve an approval (approve or reject)
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.id || !body.decision) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: id, decision' },
        { status: 400 }
      );
    }

    if (!['approved', 'rejected'].includes(body.decision)) {
      return NextResponse.json(
        { success: false, error: 'Decision must be "approved" or "rejected"' },
        { status: 400 }
      );
    }

    const decidedBy = body.decidedBy || 'Demo User';
    const reason = body.reason || undefined;

    const resolved = demoStore.resolveApproval(body.id, body.decision, decidedBy, reason);

    if (!resolved) {
      return NextResponse.json(
        { success: false, error: `Approval not found or already resolved: ${body.id}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: resolved,
    });
  } catch (error) {
    console.error('Error resolving approval:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to resolve approval' },
      { status: 500 }
    );
  }
}
