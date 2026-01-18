import { NextResponse } from 'next/server';
import { demoStore } from '@/lib/demo/store';
import { getPolicyCoverage } from '@/lib/demo/policy-engine';

// GET - Get demo stats
export async function GET() {
  const stats = demoStore.getStats();
  const coverage = getPolicyCoverage();

  return NextResponse.json({
    success: true,
    data: {
      ...stats,
      policyCoverage: coverage,
    },
  });
}
