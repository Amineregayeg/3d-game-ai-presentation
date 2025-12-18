import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/gpu/health`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "GPU API unavailable", status: response.status },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("GPU health check failed:", error);
    return NextResponse.json(
      { error: "Failed to connect to GPU API", status: "error" },
      { status: 503 }
    );
  }
}
