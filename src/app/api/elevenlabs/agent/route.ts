import { NextRequest, NextResponse } from "next/server";

// ElevenLabs Configuration
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1";

// Default agent configuration for Salesforce Consultant
const DEFAULT_AGENT_CONFIG = {
  name: "Alex - Salesforce Consultant",
  first_message: "Hello! I'm Alex, your Salesforce consultant. I have over 15 years of experience and I'm here to help you with anything Salesforce - from creating custom fields and objects, to building automation with Flows, to troubleshooting issues. What can I help you with today?",
  system_prompt: `You are Alex, a senior Salesforce consultant with 15+ years of experience and 10x certifications. You are confident, proactive, and supportive.

Key behaviors:
- Take initiative and ownership - don't wait for guidance
- Provide clear, actionable recommendations
- Explain your reasoning in a friendly, educational way
- When executing changes, explain what you're doing and why
- Use Salesforce best practices and naming conventions
- Be concise but thorough

You have access to the following Salesforce MCP tools:
- query: Execute SOQL queries
- search: Execute SOSL searches
- insert/update/delete: Manage records
- describe: Get object metadata
- apex: Execute anonymous Apex
- createField/createObject: Create schema elements

When the user asks you to do something, analyze the request, explain your approach, and execute the necessary operations. Always confirm before making changes.`,
  voice_id: "pqHfZKP75CvOlQylNhV4", // Bill voice
  language: "en",
  model: "eleven_turbo_v2_5",
  temperature: 0.7,
  max_tokens: 1000,
};

// Create or update agent
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, config } = body;

    if (!ELEVENLABS_API_KEY) {
      // Return demo mode response
      return NextResponse.json({
        _demo: true,
        agentId: "demo-agent-" + Date.now(),
        name: DEFAULT_AGENT_CONFIG.name,
        message: "ElevenLabs API key not configured. Running in demo mode.",
      });
    }

    if (action === "create") {
      // Create new conversational AI agent
      const agentConfig = {
        ...DEFAULT_AGENT_CONFIG,
        ...config,
      };

      const response = await fetch(`${ELEVENLABS_API_URL}/convai/agents`, {
        method: "POST",
        headers: {
          "xi-api-key": ELEVENLABS_API_KEY,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: agentConfig.name,
          conversation_config: {
            agent: {
              first_message: agentConfig.first_message,
              prompt: {
                prompt: agentConfig.system_prompt,
              },
              language: agentConfig.language,
            },
            tts: {
              voice_id: agentConfig.voice_id,
              model_id: agentConfig.model,
            },
            stt: {
              provider: "elevenlabs",
            },
          },
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        console.error("ElevenLabs agent creation error:", error);
        return NextResponse.json(
          { error: error.detail || "Failed to create agent" },
          { status: response.status }
        );
      }

      const agent = await response.json();
      return NextResponse.json({
        agentId: agent.agent_id,
        name: agent.name,
      });
    }

    if (action === "update") {
      const { agentId, ...updates } = config;

      const response = await fetch(`${ELEVENLABS_API_URL}/convai/agents/${agentId}`, {
        method: "PATCH",
        headers: {
          "xi-api-key": ELEVENLABS_API_KEY,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        const error = await response.json();
        return NextResponse.json(
          { error: error.detail || "Failed to update agent" },
          { status: response.status }
        );
      }

      return NextResponse.json({ success: true });
    }

    return NextResponse.json(
      { error: "Invalid action. Use 'create' or 'update'." },
      { status: 400 }
    );
  } catch (error) {
    console.error("Agent API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Agent operation failed" },
      { status: 500 }
    );
  }
}

// Get agent info
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const agentId = searchParams.get("agentId");

    if (!ELEVENLABS_API_KEY) {
      return NextResponse.json({
        _demo: true,
        agent: DEFAULT_AGENT_CONFIG,
        message: "ElevenLabs API key not configured",
      });
    }

    if (agentId) {
      // Get specific agent
      const response = await fetch(`${ELEVENLABS_API_URL}/convai/agents/${agentId}`, {
        headers: {
          "xi-api-key": ELEVENLABS_API_KEY,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        return NextResponse.json(
          { error: error.detail || "Failed to get agent" },
          { status: response.status }
        );
      }

      return NextResponse.json(await response.json());
    }

    // List all agents
    const response = await fetch(`${ELEVENLABS_API_URL}/convai/agents`, {
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || "Failed to list agents" },
        { status: response.status }
      );
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    console.error("Agent GET error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to get agent info" },
      { status: 500 }
    );
  }
}

// Delete agent
export async function DELETE(request: NextRequest) {
  try {
    const { agentId } = await request.json();

    if (!ELEVENLABS_API_KEY) {
      return NextResponse.json({ _demo: true, success: true });
    }

    const response = await fetch(`${ELEVENLABS_API_URL}/convai/agents/${agentId}`, {
      method: "DELETE",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || "Failed to delete agent" },
        { status: response.status }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Agent DELETE error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to delete agent" },
      { status: 500 }
    );
  }
}
