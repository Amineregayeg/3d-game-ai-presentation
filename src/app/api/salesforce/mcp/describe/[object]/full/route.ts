import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ object: string }> }
) {
  try {
    const { object } = await params;

    const response = await fetch(
      `${BACKEND_URL}/api/salesforce-mcp/describe/${object}/full`,
      { cache: "no-store" }
    );
    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Describe error:", error);
    return NextResponse.json(
      { success: false, error: "Backend not available" },
      { status: 500 }
    );
  }
}
