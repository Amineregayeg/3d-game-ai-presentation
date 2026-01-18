// Real-time Harbyx events client
// Uses SSE for real-time updates with polling fallback

export type HarbyxEventType = "approval.created" | "approval.decided" | "policy.updated";

export interface HarbyxRealtimeEvent {
  type: "connected" | "event" | "heartbeat";
  event?: HarbyxEventType;
  data?: {
    approval_id?: string;
    action_type?: string;
    target?: string;
    agent_id?: string;
    status?: string;
    decision?: string;
    reason?: string;
    policy_id?: string;
    policy_name?: string;
    [key: string]: unknown;
  };
  timestamp: string;
}

type EventCallback = (event: HarbyxRealtimeEvent) => void;

export class HarbyxRealtimeClient {
  private eventSource: EventSource | null = null;
  private listeners: Map<HarbyxEventType | "*", Set<EventCallback>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnected = false;

  constructor(private baseUrl: string = "/api/harbyx/events") {
    this.listeners.set("*", new Set());
    this.listeners.set("approval.created", new Set());
    this.listeners.set("approval.decided", new Set());
    this.listeners.set("policy.updated", new Set());
  }

  connect(): void {
    if (typeof window === "undefined") return;
    if (this.eventSource?.readyState === EventSource.OPEN) return;

    this.eventSource = new EventSource(this.baseUrl);

    this.eventSource.onopen = () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      console.log("[Harbyx RT] Connected to SSE");
    };

    this.eventSource.onmessage = (event) => {
      try {
        const data: HarbyxRealtimeEvent = JSON.parse(event.data);
        this.handleEvent(data);
      } catch (error) {
        console.error("[Harbyx RT] Failed to parse event:", error);
      }
    };

    this.eventSource.onerror = () => {
      this.isConnected = false;
      console.error("[Harbyx RT] Connection error");
      this.eventSource?.close();
      this.attemptReconnect();
    };
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("[Harbyx RT] Max reconnection attempts reached");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`[Harbyx RT] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => this.connect(), delay);
  }

  private handleEvent(event: HarbyxRealtimeEvent): void {
    if (event.type === "heartbeat") {
      return; // Ignore heartbeats
    }

    // Notify wildcard listeners
    this.listeners.get("*")?.forEach((callback) => callback(event));

    // Notify specific event listeners
    if (event.event && this.listeners.has(event.event)) {
      this.listeners.get(event.event)?.forEach((callback) => callback(event));
    }
  }

  on(event: HarbyxEventType | "*", callback: EventCallback): () => void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.add(callback);
    }

    // Return unsubscribe function
    return () => {
      listeners?.delete(callback);
    };
  }

  disconnect(): void {
    this.eventSource?.close();
    this.eventSource = null;
    this.isConnected = false;
    console.log("[Harbyx RT] Disconnected");
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }
}

// Singleton instance
let realtimeClient: HarbyxRealtimeClient | null = null;

export function getRealtimeClient(): HarbyxRealtimeClient {
  if (!realtimeClient) {
    realtimeClient = new HarbyxRealtimeClient();
  }
  return realtimeClient;
}
