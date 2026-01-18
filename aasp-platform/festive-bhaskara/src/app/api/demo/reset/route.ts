import { NextResponse } from 'next/server';
import { demoStore } from '@/lib/demo/store';

// POST - Reset demo state
export async function POST() {
  demoStore.reset();

  return NextResponse.json({
    success: true,
    message: 'Demo state reset successfully',
  });
}
