import { NextRequest, NextResponse } from "next/server";

// ElevenLabs Configuration
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1";

// Store active conversations (use Redis in production)
const activeConversations = new Map<string, {
  agentId: string;
  startedAt: number;
  status: "active" | "ended";
}>();

// Start a conversation
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, agentId, voiceId, conversationId } = body;

    if (!ELEVENLABS_API_KEY) {
      // Demo mode - return simulated WebSocket URL
      const demoConversationId = `demo-conv-${Date.now()}`;
      return NextResponse.json({
        _demo: true,
        conversationId: demoConversationId,
        wsUrl: `wss://demo.elevenlabs.io/conversation/${demoConversationId}`,
        message: "ElevenLabs API key not configured. Running in demo mode.",
      });
    }

    if (action === "start") {
      // Use provided agentId or default
      const targetAgentId = agentId || "agent_7001kdqegdr4eyct05t0cawfwxtf";

      // Get signed URL for WebSocket connection
      const response = await fetch(
        `${ELEVENLABS_API_URL}/convai/conversation/get_signed_url?agent_id=${targetAgentId}`,
        {
          method: "GET",
          headers: {
            "xi-api-key": ELEVENLABS_API_KEY,
          },
        }
      );

      if (!response.ok) {
        const error = await response.json();
        console.error("ElevenLabs conversation start error:", error);
        return NextResponse.json(
          { error: error.detail || "Failed to start conversation" },
          { status: response.status }
        );
      }

      const { signed_url } = await response.json();
      const newConversationId = `conv-${Date.now()}`;

      activeConversations.set(newConversationId, {
        agentId,
        startedAt: Date.now(),
        status: "active",
      });

      return NextResponse.json({
        conversationId: newConversationId,
        wsUrl: signed_url,
      });
    }

    if (action === "end") {
      if (conversationId && activeConversations.has(conversationId)) {
        activeConversations.get(conversationId)!.status = "ended";
      }

      return NextResponse.json({
        success: true,
        message: "Conversation ended",
      });
    }

    if (action === "status") {
      const conversation = conversationId
        ? activeConversations.get(conversationId)
        : null;

      return NextResponse.json({
        conversationId,
        status: conversation?.status || "unknown",
        startedAt: conversation?.startedAt,
      });
    }

    return NextResponse.json(
      { error: "Invalid action. Use 'start', 'end', or 'status'." },
      { status: 400 }
    );
  } catch (error) {
    console.error("Conversation API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Conversation operation failed" },
      { status: 500 }
    );
  }
}

// Get conversation history
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const conversationId = searchParams.get("conversationId");

    if (!ELEVENLABS_API_KEY) {
      return NextResponse.json({
        _demo: true,
        conversations: Array.from(activeConversations.entries()).map(([id, data]) => ({
          conversationId: id,
          ...data,
        })),
      });
    }

    if (conversationId) {
      // Get specific conversation
      const response = await fetch(
        `${ELEVENLABS_API_URL}/convai/conversations/${conversationId}`,
        {
          headers: {
            "xi-api-key": ELEVENLABS_API_KEY,
          },
        }
      );

      if (!response.ok) {
        const error = await response.json();
        return NextResponse.json(
          { error: error.detail || "Failed to get conversation" },
          { status: response.status }
        );
      }

      return NextResponse.json(await response.json());
    }

    // List recent conversations
    const response = await fetch(`${ELEVENLABS_API_URL}/convai/conversations`, {
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || "Failed to list conversations" },
        { status: response.status }
      );
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    console.error("Conversation GET error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to get conversation" },
      { status: 500 }
    );
  }
}

// Get audio for a conversation turn (for lip-sync)
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { conversationId, messageId } = body;

    if (!ELEVENLABS_API_KEY) {
      return NextResponse.json({
        _demo: true,
        audioUrl: null,
        message: "Demo mode - no audio available",
      });
    }

    // Get audio for specific message
    const response = await fetch(
      `${ELEVENLABS_API_URL}/convai/conversations/${conversationId}/audio/${messageId}`,
      {
        headers: {
          "xi-api-key": ELEVENLABS_API_KEY,
        },
      }
    );

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || "Failed to get audio" },
        { status: response.status }
      );
    }

    // Return audio as base64
    const audioBuffer = await response.arrayBuffer();
    const base64Audio = Buffer.from(audioBuffer).toString("base64");

    return NextResponse.json({
      audioUrl: `data:audio/mpeg;base64,${base64Audio}`,
    });
  } catch (error) {
    console.error("Audio GET error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to get audio" },
      { status: 500 }
    );
  }
}
