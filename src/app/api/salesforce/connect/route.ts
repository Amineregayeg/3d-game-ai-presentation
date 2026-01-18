import { NextResponse } from "next/server";

// Salesforce OAuth Configuration
const SALESFORCE_CLIENT_ID = process.env.SALESFORCE_CLIENT_ID;
const SALESFORCE_CLIENT_SECRET = process.env.SALESFORCE_CLIENT_SECRET;
const SALESFORCE_REDIRECT_URI = process.env.SALESFORCE_REDIRECT_URI || "http://localhost:3000/api/salesforce/callback";
const SALESFORCE_LOGIN_URL = process.env.SALESFORCE_LOGIN_URL || "https://login.salesforce.com";

export async function POST() {
  try {
    // Check if credentials are configured
    if (!SALESFORCE_CLIENT_ID || !SALESFORCE_CLIENT_SECRET) {
      // Return demo mode response
      return NextResponse.json({
        demo: true,
        message: "Salesforce credentials not configured. Running in demo mode.",
      });
    }

    // Build OAuth authorization URL
    const authUrl = new URL(`${SALESFORCE_LOGIN_URL}/services/oauth2/authorize`);
    authUrl.searchParams.set("response_type", "code");
    authUrl.searchParams.set("client_id", SALESFORCE_CLIENT_ID);
    authUrl.searchParams.set("redirect_uri", SALESFORCE_REDIRECT_URI);
    authUrl.searchParams.set("scope", "api refresh_token web");
    authUrl.searchParams.set("prompt", "login consent");

    return NextResponse.json({
      authUrl: authUrl.toString(),
    });
  } catch (error) {
    console.error("Salesforce connect error:", error);
    return NextResponse.json(
      { error: "Failed to initiate Salesforce connection" },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Return connection status
  return NextResponse.json({
    configured: !!(SALESFORCE_CLIENT_ID && SALESFORCE_CLIENT_SECRET),
    loginUrl: SALESFORCE_LOGIN_URL,
  });
}
