import { NextResponse } from "next/server";

const VPS_API = process.env.VPS_API_URL || "http://5.249.161.66:5000";

export async function GET() {
  try {
    const res = await fetch(`${VPS_API}/api/avatar/avatars`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });

    if (!res.ok) {
      return NextResponse.json(
        {
          error: "Failed to fetch avatars",
          avatars: [{ id: "default", name: "Default Avatar", url: "/3dgameassistant/api/avatar/static/avatars/default.png" }]
        },
        { status: res.status }
      );
    }

    const data = await res.json();

    // Rewrite avatar URLs to use our proxy with basePath
    if (data.avatars) {
      data.avatars = data.avatars.map((avatar: { id: string; name: string; url: string }) => ({
        ...avatar,
        url: "/3dgameassistant" + avatar.url.replace("/static/", "/api/avatar/static/")
      }));
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching avatars:", error);
    return NextResponse.json(
      {
        error: "Failed to connect to backend",
        avatars: [{ id: "default", name: "Default Avatar", url: "/3dgameassistant/api/avatar/static/avatars/default.png" }]
      },
      { status: 500 }
    );
  }
}
