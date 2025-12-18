import { NextRequest, NextResponse } from "next/server";

const VPS_API = process.env.VPS_API_URL || "http://5.249.161.66:5000";

export async function GET(request: NextRequest) {
  try {
    const url = request.nextUrl.searchParams.get("url");

    if (!url) {
      return NextResponse.json({ error: "Missing url parameter" }, { status: 400 });
    }

    // Construct full URL to VPS
    const fullUrl = url.startsWith("http") ? url : `${VPS_API}${url}`;

    const res = await fetch(fullUrl);

    if (!res.ok) {
      return NextResponse.json(
        { error: "Failed to fetch media" },
        { status: res.status }
      );
    }

    // Get the content type from the response
    const contentType = res.headers.get("content-type") || "application/octet-stream";

    // Stream the response
    const blob = await res.blob();

    return new NextResponse(blob, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error) {
    console.error("Error fetching media:", error);
    return NextResponse.json(
      { error: "Failed to fetch media" },
      { status: 500 }
    );
  }
}
