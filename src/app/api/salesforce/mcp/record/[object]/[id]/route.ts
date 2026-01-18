import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ object: string; id: string }> }
) {
  try {
    const { object, id } = await params;

    const response = await fetch(
      `${BACKEND_URL}/api/salesforce-mcp/record/${object}/${id}`,
      { cache: "no-store" }
    );
    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Record fetch error:", error);
    return NextResponse.json(
      { success: false, error: "Backend not available" },
      { status: 500 }
    );
  }
}
