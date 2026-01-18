import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET() {
  try {
    // Get status from backend
    const response = await fetch(`${BACKEND_URL}/api/salesforce-mcp/status`);
    const data = await response.json();

    if (data.connected) {
      return NextResponse.json({
        connected: true,
        org: {
          id: data.org_id,
          name: data.org_name,
          instanceUrl: data.instance_url,
          username: data.username,
          orgType: data.org_type?.toLowerCase() || "developer",
          connectedAt: new Date().toISOString(),
        },
      });
    }

    return NextResponse.json({
      connected: false,
      message: data.error || "Not connected to Salesforce",
    });
  } catch (error) {
    console.error("Status check error:", error);
    return NextResponse.json({
      connected: false,
      error: "Backend not available",
    });
  }
}

export async function DELETE() {
  // Since we use OAuth Client Credentials, there's no session to clear
  // The connection is persistent
  return NextResponse.json({
    success: true,
    message: "Connection status cleared",
  });
}
