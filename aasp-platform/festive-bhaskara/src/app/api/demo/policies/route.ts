import { NextRequest, NextResponse } from 'next/server';
import { demoStore } from '@/lib/demo/store';
import type { Policy, PolicyRule } from '@/lib/demo/types';

function generateId(): string {
  return crypto.randomUUID();
}

// GET - List all policies
export async function GET() {
  const policies = demoStore.getPolicies();

  return NextResponse.json({
    success: true,
    data: policies,
    total: policies.length,
  });
}

// POST - Create a new policy
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.name || !body.rules || !Array.isArray(body.rules)) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: name, rules (array)' },
        { status: 400 }
      );
    }

    // Generate IDs for rules
    const rules: PolicyRule[] = body.rules.map((rule: Partial<PolicyRule>) => ({
      ...rule,
      id: rule.id || generateId(),
    }));

    // Create policy
    const policy: Policy = {
      id: generateId(),
      name: body.name,
      description: body.description || '',
      enabled: body.enabled !== false, // Default to enabled
      rules,
      createdAt: new Date().toISOString(),
    };

    demoStore.addPolicy(policy);

    return NextResponse.json({
      success: true,
      data: policy,
    });
  } catch (error) {
    console.error('Error creating policy:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to create policy' },
      { status: 500 }
    );
  }
}

// PUT - Update a policy
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();

    if (!body.id) {
      return NextResponse.json(
        { success: false, error: 'Missing required field: id' },
        { status: 400 }
      );
    }

    // If rules are provided, ensure they have IDs
    if (body.rules && Array.isArray(body.rules)) {
      body.rules = body.rules.map((rule: Partial<PolicyRule>) => ({
        ...rule,
        id: rule.id || generateId(),
      }));
    }

    const updated = demoStore.updatePolicy(body.id, body);

    if (!updated) {
      return NextResponse.json(
        { success: false, error: `Policy not found: ${body.id}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: updated,
    });
  } catch (error) {
    console.error('Error updating policy:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to update policy' },
      { status: 500 }
    );
  }
}

// DELETE - Delete a policy
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json(
        { success: false, error: 'Missing required query param: id' },
        { status: 400 }
      );
    }

    const deleted = demoStore.deletePolicy(id);

    if (!deleted) {
      return NextResponse.json(
        { success: false, error: `Policy not found: ${id}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Policy deleted successfully',
    });
  } catch (error) {
    console.error('Error deleting policy:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to delete policy' },
      { status: 500 }
    );
  }
}

// PATCH - Toggle policy enabled state
export async function PATCH(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json(
        { success: false, error: 'Missing required query param: id' },
        { status: 400 }
      );
    }

    const toggled = demoStore.togglePolicy(id);

    if (!toggled) {
      return NextResponse.json(
        { success: false, error: `Policy not found: ${id}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: toggled,
    });
  } catch (error) {
    console.error('Error toggling policy:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to toggle policy' },
      { status: 500 }
    );
  }
}
