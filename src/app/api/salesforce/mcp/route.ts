import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

// Proxy MCP operations to the backend
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Forward to backend MCP API
    const response = await fetch(`${BACKEND_URL}/api/salesforce-mcp/execute`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("MCP proxy error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "MCP operation failed" },
      { status: 500 }
    );
  }
}

// Get MCP status
export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/salesforce-mcp/status`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("MCP status error:", error);
    return NextResponse.json(
      { connected: false, error: "Backend not available" },
      { status: 500 }
    );
  }
}
