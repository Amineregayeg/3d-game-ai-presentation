// Harbyx Policy Sync API
// Syncs local Salesforce Consultant policies to Harbyx dashboard

import { NextResponse } from 'next/server';
import { syncPoliciesToHarbyx, getSyncStatus, clearHarbyxPolicies } from '@/lib/harbyx/sync';

export async function POST() {
  try {
    // In production, add proper authentication check here
    // For now, we allow sync from the dashboard
    const result = await syncPoliciesToHarbyx();
    return NextResponse.json(result);
  } catch (error) {
    console.error('[Harbyx Sync API] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Sync failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const status = await getSyncStatus();
    return NextResponse.json(status);
  } catch (error) {
    console.error('[Harbyx Sync API] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to get status' },
      { status: 500 }
    );
  }
}

export async function DELETE() {
  try {
    // In production, add proper authentication check here
    const result = await clearHarbyxPolicies();
    return NextResponse.json(result);
  } catch (error) {
    console.error('[Harbyx Sync API] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Clear failed' },
      { status: 500 }
    );
  }
}
