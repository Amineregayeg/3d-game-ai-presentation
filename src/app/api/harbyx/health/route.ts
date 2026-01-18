import { NextResponse } from 'next/server';
import { getHarbyxClient, HarbyxError } from '@/lib/harbyx';

export async function GET() {
  try {
    const client = getHarbyxClient();
    const health = await client.healthCheck();

    return NextResponse.json({
      status: health.connected ? 'connected' : 'disconnected',
      harbyx: health,
      agentId: 'salesforce-consultant-agent',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    if (error instanceof HarbyxError && error.code === 'MISSING_API_KEY') {
      return NextResponse.json(
        {
          status: 'not_configured',
          message: 'HARBYX_API_KEY environment variable is not set',
          timestamp: new Date().toISOString(),
        },
        { status: 503 }
      );
    }

    return NextResponse.json(
      {
        status: 'error',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
