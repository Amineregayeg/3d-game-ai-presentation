import { NextResponse } from "next/server";

const VPS_API = process.env.VPS_API_URL || "http://5.249.161.66:5000";

export async function GET() {
  try {
    const res = await fetch(`${VPS_API}/api/avatar/status`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });

    if (!res.ok) {
      return NextResponse.json(
        {
          elevenlabs: false,
          musetalk: false,
          gpu_server: "offline"
        },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching status:", error);
    return NextResponse.json(
      {
        elevenlabs: false,
        musetalk: false,
        gpu_server: "offline",
        error: "Failed to connect to backend"
      },
      { status: 500 }
    );
  }
}
