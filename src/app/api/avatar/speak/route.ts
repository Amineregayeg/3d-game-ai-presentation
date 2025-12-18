import { NextResponse } from "next/server";

const VPS_API = process.env.VPS_API_URL || "http://5.249.161.66:5000";

// Increase timeout for long video generation (up to 5 minutes)
export const maxDuration = 300;

export async function POST(request: Request) {
  try {
    const body = await request.json();

    // Use AbortController for fetch timeout (4 minutes)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 240000);

    const res = await fetch(`${VPS_API}/api/avatar/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ error: "Unknown error" }));
      return NextResponse.json(
        { error: errorData.error || "Failed to generate speech" },
        { status: res.status }
      );
    }

    const data = await res.json();

    // Rewrite URLs to proxy through our API with basePath
    // The VPS returns paths like /static/avatar/output/xxx.mp3
    // We need to serve these through our proxy
    if (data.audio_url) {
      data.audio_url = `/3dgameassistant/api/avatar/media?url=${encodeURIComponent(data.audio_url)}`;
    }
    if (data.video_url) {
      data.video_url = `/3dgameassistant/api/avatar/media?url=${encodeURIComponent(data.video_url)}`;
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error generating speech:", error);
    return NextResponse.json(
      { error: "Failed to connect to backend" },
      { status: 500 }
    );
  }
}
