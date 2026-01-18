"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// Types for ElevenLabs Conversational AI
export interface ConversationState {
  status: "idle" | "connecting" | "connected" | "error";
  isUserSpeaking: boolean;
  isAgentSpeaking: boolean;
  userTranscript: string;
  agentTranscript: string;
  error?: string;
}

export interface ConversationCallbacks {
  onUserTranscript?: (transcript: string, isFinal: boolean) => void;
  onAgentTranscript?: (transcript: string) => void;
  onAgentAudioStart?: () => void;
  onAgentAudioEnd?: () => void;
  onError?: (error: string) => void;
  onStatusChange?: (status: ConversationState["status"]) => void;
}

const SAMPLE_RATE = 16000;

// Helper to convert ArrayBuffer to base64
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export function useElevenLabsConversation(callbacks?: ConversationCallbacks) {
  const [state, setState] = useState<ConversationState>({
    status: "idle",
    isUserSpeaking: false,
    isAgentSpeaking: false,
    userTranscript: "",
    agentTranscript: "",
  });

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const isPlayingRef = useRef(false);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const nextPlayTimeRef = useRef(0);

  // Store callbacks in ref to avoid dependency issues
  const callbacksRef = useRef(callbacks);
  useEffect(() => {
    callbacksRef.current = callbacks;
  }, [callbacks]);

  // Update state helper
  const updateState = useCallback((updates: Partial<ConversationState>) => {
    setState((prev) => {
      const newState = { ...prev, ...updates };
      if (updates.status && callbacksRef.current?.onStatusChange) {
        callbacksRef.current.onStatusChange(updates.status);
      }
      return newState;
    });
  }, []);

  // Convert Float32 to Int16 PCM
  const floatTo16BitPCM = (float32Array: Float32Array): ArrayBuffer => {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  };

  // Play received audio
  const playAudio = useCallback(async (audioData: ArrayBuffer) => {
    try {
      if (!playbackContextRef.current) {
        playbackContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
      }
      const ctx = playbackContextRef.current;

      // ElevenLabs sends PCM 16-bit mono at 16kHz
      const pcmData = new Int16Array(audioData);
      const floatData = new Float32Array(pcmData.length);
      for (let i = 0; i < pcmData.length; i++) {
        floatData[i] = pcmData[i] / 32768;
      }

      const audioBuffer = ctx.createBuffer(1, floatData.length, SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(floatData);

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);

      // Schedule playback
      const currentTime = ctx.currentTime;
      const startTime = Math.max(currentTime, nextPlayTimeRef.current);
      source.start(startTime);
      nextPlayTimeRef.current = startTime + audioBuffer.duration;

      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        updateState({ isAgentSpeaking: true });
        callbacksRef.current?.onAgentAudioStart?.();
      }

      source.onended = () => {
        if (ctx.currentTime >= nextPlayTimeRef.current - 0.1) {
          isPlayingRef.current = false;
          updateState({ isAgentSpeaking: false });
          callbacksRef.current?.onAgentAudioEnd?.();
        }
      };
    } catch (e) {
      console.error("Audio playback error:", e);
    }
  }, [updateState]);

  // Handle WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    // Binary data = audio
    if (event.data instanceof Blob) {
      event.data.arrayBuffer().then((buffer) => {
        playAudio(buffer);
      });
      return;
    }

    if (event.data instanceof ArrayBuffer) {
      playAudio(event.data);
      return;
    }

    // Text data = JSON control messages
    try {
      const data = JSON.parse(event.data);
      console.log("ElevenLabs message:", data);

      switch (data.type) {
        case "conversation_initiation_metadata":
          console.log("Conversation initialized:", data);
          updateState({ status: "connected" });
          break;

        case "user_transcript":
          if (data.user_transcript) {
            updateState({
              userTranscript: data.user_transcript,
              isUserSpeaking: true,
            });
            callbacksRef.current?.onUserTranscript?.(data.user_transcript, false);
          }
          break;

        case "user_transcript_final":
          if (data.user_transcript) {
            updateState({
              userTranscript: data.user_transcript,
              isUserSpeaking: false,
            });
            callbacksRef.current?.onUserTranscript?.(data.user_transcript, true);
          }
          break;

        case "agent_response":
          if (data.agent_response) {
            updateState({ agentTranscript: data.agent_response });
            callbacksRef.current?.onAgentTranscript?.(data.agent_response);
          }
          break;

        case "interruption":
          // User interrupted
          audioQueueRef.current = [];
          nextPlayTimeRef.current = 0;
          isPlayingRef.current = false;
          updateState({ isAgentSpeaking: false });
          break;

        case "ping":
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "pong", event_id: data.ping_event?.event_id }));
          }
          break;

        case "error":
          console.error("ElevenLabs error:", data);
          updateState({ error: data.message || "Unknown error" });
          callbacksRef.current?.onError?.(data.message || "Unknown error");
          break;
      }
    } catch (e) {
      console.warn("Failed to parse message:", e);
    }
  }, [playAudio, updateState]);

  // Setup microphone capture
  const setupMicrophone = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);

        // Simple VAD
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        const speaking = rms > 0.01;

        setState((prev) => {
          if (prev.isUserSpeaking !== speaking) {
            return { ...prev, isUserSpeaking: speaking };
          }
          return prev;
        });

        // Convert to 16-bit PCM and send as base64-encoded JSON
        const pcmBuffer = floatTo16BitPCM(inputData);
        const base64Audio = arrayBufferToBase64(pcmBuffer);
        wsRef.current?.send(JSON.stringify({
          user_audio_chunk: base64Audio
        }));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log("Microphone setup complete");
      return true;
    } catch (error) {
      console.error("Microphone setup failed:", error);
      updateState({ status: "error", error: "Microphone access denied" });
      callbacksRef.current?.onError?.("Microphone access denied");
      return false;
    }
  }, [updateState]);

  // Cleanup resources
  const cleanup = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (playbackContextRef.current) {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    nextPlayTimeRef.current = 0;
  }, []);

  // Start conversation
  const startConversation = useCallback(async (wsUrl: string, options?: {
    systemPrompt?: string;
    voiceId?: string;
    language?: 'en' | 'fr';
    avatarName?: string;
  }) => {
    if (state.status === "connecting" || state.status === "connected") {
      return;
    }

    updateState({
      status: "connecting",
      userTranscript: "",
      agentTranscript: "",
      error: undefined,
    });

    try {
      // Setup microphone first
      const micReady = await setupMicrophone();
      if (!micReady) {
        throw new Error("Failed to setup microphone");
      }

      // Connect WebSocket
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected");
        // Build conversation config with avatar's voice and system prompt
        const conversationOverride: Record<string, unknown> = {};

        // Set voice if provided (and model for non-English languages)
        // ElevenLabs uses camelCase for overrides
        if (options?.voiceId) {
          conversationOverride.tts = {
            voiceId: options.voiceId,
            // French requires multilingual model, not turbo_v2
            modelId: options?.language === 'fr' ? 'eleven_multilingual_v2' : 'eleven_turbo_v2'
          };
        }

        // Set system prompt, language, and first message if provided
        if (options?.systemPrompt || options?.language) {
          // Add voice-friendly instructions to the prompt
          const voicePrompt = options.systemPrompt
            ? `${options.systemPrompt}\n\n# IMPORTANT - Voice Conversation Rules:\n- Keep responses SHORT and conversational (2-3 sentences max)\n- Speak naturally as if having a phone call\n- Avoid lists, code blocks, or long explanations\n- Ask one question at a time\n- Be warm and engaging`
            : undefined;

          conversationOverride.agent = {
            ...(voicePrompt && {
              prompt: {
                prompt: voicePrompt
              }
            }),
            // Set first message based on avatar name
            ...(options.avatarName && {
              firstMessage: options.language === 'fr'
                ? `Bonjour! Je suis ${options.avatarName}, votre consultant Salesforce. Comment puis-je vous aider aujourd'hui?`
                : `Hi there! I'm ${options.avatarName}, your Salesforce consultant. How can I help you today?`
            }),
            // Set language for proper speech recognition
            language: options?.language === 'fr' ? 'fr' : 'en'
          };
        }

        const config: Record<string, unknown> = {
          type: "conversation_initiation_client_data",
        };

        // Add overrides if we have settings
        // Try both formats - SDK uses 'overrides', raw WebSocket may use 'conversation_config_override'
        if (Object.keys(conversationOverride).length > 0) {
          config.conversation_config_override = conversationOverride;
        }

        console.log("Sending conversation config:", JSON.stringify(config, null, 2));
        ws.send(JSON.stringify(config));
      };

      ws.onmessage = handleMessage;

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        updateState({ status: "error", error: "Connection error" });
        callbacksRef.current?.onError?.("Connection error");
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        cleanup();
        updateState({ status: "idle" });
      };
    } catch (error) {
      console.error("Failed to start conversation:", error);
      cleanup();
      updateState({
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      });
      callbacksRef.current?.onError?.(
        error instanceof Error ? error.message : "Unknown error"
      );
    }
  }, [state.status, updateState, setupMicrophone, handleMessage, cleanup]);

  // End conversation
  const endConversation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    cleanup();
    updateState({
      status: "idle",
      isUserSpeaking: false,
      isAgentSpeaking: false,
    });
  }, [cleanup, updateState]);

  // Cleanup on unmount
  const endConversationRef = useRef(endConversation);
  useEffect(() => {
    endConversationRef.current = endConversation;
  }, [endConversation]);

  useEffect(() => {
    return () => {
      endConversationRef.current();
    };
  }, []);

  return {
    state,
    startConversation,
    endConversation,
    isConnected: state.status === "connected",
    isConnecting: state.status === "connecting",
  };
}
