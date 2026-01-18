import { NextRequest, NextResponse } from "next/server";

// Salesforce OAuth Configuration
const SALESFORCE_CLIENT_ID = process.env.SALESFORCE_CLIENT_ID;
const SALESFORCE_CLIENT_SECRET = process.env.SALESFORCE_CLIENT_SECRET;
const SALESFORCE_REDIRECT_URI = process.env.SALESFORCE_REDIRECT_URI || "http://localhost:3000/api/salesforce/callback";
const SALESFORCE_LOGIN_URL = process.env.SALESFORCE_LOGIN_URL || "https://login.salesforce.com";

// In-memory token storage (use Redis/DB in production)
const tokenStore = new Map<string, {
  accessToken: string;
  refreshToken: string;
  instanceUrl: string;
  userId: string;
  orgId: string;
  expiresAt: number;
}>();

export async function POST(request: NextRequest) {
  try {
    const { code } = await request.json();

    if (!code) {
      return NextResponse.json(
        { error: "Authorization code is required" },
        { status: 400 }
      );
    }

    if (!SALESFORCE_CLIENT_ID || !SALESFORCE_CLIENT_SECRET) {
      return NextResponse.json(
        { error: "Salesforce credentials not configured" },
        { status: 500 }
      );
    }

    // Exchange authorization code for tokens
    const tokenUrl = `${SALESFORCE_LOGIN_URL}/services/oauth2/token`;
    const tokenParams = new URLSearchParams({
      grant_type: "authorization_code",
      code,
      client_id: SALESFORCE_CLIENT_ID,
      client_secret: SALESFORCE_CLIENT_SECRET,
      redirect_uri: SALESFORCE_REDIRECT_URI,
    });

    const tokenResponse = await fetch(tokenUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: tokenParams.toString(),
    });

    if (!tokenResponse.ok) {
      const error = await tokenResponse.json();
      console.error("Token exchange error:", error);
      return NextResponse.json(
        { error: error.error_description || "Token exchange failed" },
        { status: 400 }
      );
    }

    const tokens = await tokenResponse.json();

    // Get org info
    const userInfoResponse = await fetch(`${tokens.instance_url}/services/oauth2/userinfo`, {
      headers: {
        Authorization: `Bearer ${tokens.access_token}`,
      },
    });

    let orgInfo: {
      name: string;
      username: string;
      orgType: "production" | "sandbox" | "developer";
    } = {
      name: "Unknown Org",
      username: tokens.id?.split("/").pop() || "unknown",
      orgType: "production",
    };

    if (userInfoResponse.ok) {
      const userInfo = await userInfoResponse.json();
      orgInfo = {
        name: userInfo.organization_id || "Salesforce Org",
        username: userInfo.preferred_username || userInfo.email,
        orgType: tokens.instance_url.includes("sandbox") ? "sandbox" :
                 tokens.instance_url.includes("dev-ed") ? "developer" : "production",
      };
    }

    // Store tokens (use session ID as key)
    const sessionId = crypto.randomUUID();
    tokenStore.set(sessionId, {
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      instanceUrl: tokens.instance_url,
      userId: tokens.id,
      orgId: tokens.id?.split("/")[4] || "",
      expiresAt: Date.now() + (tokens.issued_at ? parseInt(tokens.issued_at) : 3600000),
    });

    // Set session cookie
    const response = NextResponse.json({
      success: true,
      org: {
        id: tokens.id?.split("/")[4] || sessionId,
        name: orgInfo.name,
        instanceUrl: tokens.instance_url,
        username: orgInfo.username,
        orgType: orgInfo.orgType,
        connectedAt: new Date().toISOString(),
      },
    });

    response.cookies.set("sf_session", sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 3600, // 1 hour
    });

    return response;
  } catch (error) {
    console.error("Salesforce callback error:", error);
    return NextResponse.json(
      { error: "Failed to complete authentication" },
      { status: 500 }
    );
  }
}

// Handle GET request (OAuth redirect)
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get("code");
  const error = searchParams.get("error");

  if (error) {
    // Redirect to demo page with error
    return NextResponse.redirect(
      new URL(`/salesforce_demo?error=${error}`, request.url)
    );
  }

  if (code) {
    // Redirect to demo page with code (client will POST to exchange)
    return NextResponse.redirect(
      new URL(`/salesforce_demo?code=${code}`, request.url)
    );
  }

  return NextResponse.redirect(new URL("/salesforce_demo", request.url));
}

// Export token store for use in other routes
export { tokenStore };
