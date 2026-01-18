import { NextRequest, NextResponse } from "next/server";

// ElevenLabs Configuration
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1";

// Default voice ID (Rachel - professional female voice)
const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM";

interface TTSRequest {
  text: string;
  voiceId?: string;
  modelId?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as TTSRequest;

    if (!body.text || typeof body.text !== "string") {
      return NextResponse.json(
        { error: "Text is required" },
        { status: 400 }
      );
    }

    if (!ELEVENLABS_API_KEY) {
      // Demo mode - return null audioUrl
      return NextResponse.json({
        _demo: true,
        audioUrl: null,
        message: "ElevenLabs API key not configured. Running in demo mode.",
      });
    }

    const voiceId = body.voiceId || DEFAULT_VOICE_ID;
    const modelId = body.modelId || "eleven_monolingual_v1";

    // Call ElevenLabs TTS API
    const response = await fetch(
      `${ELEVENLABS_API_URL}/text-to-speech/${voiceId}`,
      {
        method: "POST",
        headers: {
          "Accept": "audio/mpeg",
          "Content-Type": "application/json",
          "xi-api-key": ELEVENLABS_API_KEY,
        },
        body: JSON.stringify({
          text: body.text,
          model_id: modelId,
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.5,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "TTS generation failed" }));
      console.error("ElevenLabs TTS error:", error);

      // Return graceful fallback for quota exceeded
      if (error.status === "quota_exceeded" || response.status === 429) {
        return NextResponse.json({
          audioUrl: null,
          quotaExceeded: true,
          message: "TTS quota exceeded. Voice will be available in conversations.",
        });
      }

      return NextResponse.json(
        { error: error.detail || "TTS generation failed" },
        { status: response.status }
      );
    }

    // Get audio as buffer and convert to base64 data URL
    const audioBuffer = await response.arrayBuffer();
    const base64Audio = Buffer.from(audioBuffer).toString("base64");
    const audioUrl = `data:audio/mpeg;base64,${base64Audio}`;

    return NextResponse.json({
      audioUrl,
      voiceId,
      textLength: body.text.length,
    });
  } catch (error) {
    console.error("TTS API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "TTS generation failed" },
      { status: 500 }
    );
  }
}
