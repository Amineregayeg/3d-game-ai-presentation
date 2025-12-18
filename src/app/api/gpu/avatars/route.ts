import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/gpu/avatars`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json({ avatars: [] }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to fetch avatars:", error);
    return NextResponse.json({ avatars: [] }, { status: 503 });
  }
}
