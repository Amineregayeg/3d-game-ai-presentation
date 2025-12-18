import { NextResponse } from "next/server";

const VPS_API = process.env.VPS_API_URL || "http://5.249.161.66:5000";

export async function GET() {
  try {
    const res = await fetch(`${VPS_API}/api/avatar/voices`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });

    if (!res.ok) {
      return NextResponse.json(
        { error: "Failed to fetch voices", voices: [] },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching voices:", error);
    return NextResponse.json(
      { error: "Failed to connect to backend", voices: [] },
      { status: 500 }
    );
  }
}
