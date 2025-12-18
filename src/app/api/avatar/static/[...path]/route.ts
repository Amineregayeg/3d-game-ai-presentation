import { NextRequest, NextResponse } from "next/server";

const VPS_API = process.env.VPS_API_URL || "http://5.249.161.66:5000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const { path } = await params;
    const filePath = path.join("/");
    const fullUrl = `${VPS_API}/static/${filePath}`;

    const res = await fetch(fullUrl);

    if (!res.ok) {
      return NextResponse.json(
        { error: "File not found" },
        { status: res.status }
      );
    }

    const contentType = res.headers.get("content-type") || "application/octet-stream";
    const blob = await res.blob();

    return new NextResponse(blob, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch (error) {
    console.error("Error fetching static file:", error);
    return NextResponse.json(
      { error: "Failed to fetch file" },
      { status: 500 }
    );
  }
}
