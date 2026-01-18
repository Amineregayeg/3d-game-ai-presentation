import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

// Harbyx Webhook Handler for Salesforce Consultant Agent
// Receives events from Harbyx: approval.created, approval.decided, policy.updated

// In-memory event store for SSE (in production, use Redis or database)
// This is shared with the events endpoint
export const eventStore = {
  events: [] as WebhookEvent[],
  maxEvents: 100,
  listeners: new Set<(event: WebhookEvent) => void>(),

  addEvent(event: WebhookEvent) {
    this.events.unshift(event);
    if (this.events.length > this.maxEvents) {
      this.events.pop();
    }
    // Notify all SSE listeners
    this.listeners.forEach((listener) => listener(event));
  },

  subscribe(listener: (event: WebhookEvent) => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  },

  getRecent(since?: string): WebhookEvent[] {
    if (!since) return this.events.slice(0, 10);
    return this.events.filter((e) => new Date(e.timestamp) > new Date(since));
  },
};

export interface WebhookEvent {
  id: string;
  event: "approval.created" | "approval.decided" | "policy.updated";
  data: {
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

/**
 * Verify webhook signature from Harbyx
 */
function verifySignature(
  payload: string,
  signature: string | null,
  secret: string
): boolean {
  if (!signature || !secret) return false;

  const expectedSignature = crypto
    .createHmac("sha256", secret)
    .update(payload)
    .digest("hex");

  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(`sha256=${expectedSignature}`)
  );
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.text();
    const signature = request.headers.get("x-harbyx-signature");
    const webhookSecret = process.env.HARBYX_WEBHOOK_SECRET;

    // Verify signature if secret is configured
    if (webhookSecret && !verifySignature(body, signature, webhookSecret)) {
      console.error("[Webhook] Invalid signature");
      return NextResponse.json({ error: "Invalid signature" }, { status: 401 });
    }

    const payload = JSON.parse(body) as {
      event: WebhookEvent["event"];
      data: WebhookEvent["data"];
    };

    // Validate event type
    const validEvents = ["approval.created", "approval.decided", "policy.updated"];
    if (!validEvents.includes(payload.event)) {
      return NextResponse.json(
        { error: "Unknown event type" },
        { status: 400 }
      );
    }

    // Create webhook event
    const webhookEvent: WebhookEvent = {
      id: crypto.randomUUID(),
      event: payload.event,
      data: payload.data,
      timestamp: new Date().toISOString(),
    };

    // Store event for SSE delivery
    eventStore.addEvent(webhookEvent);

    console.log(`[Webhook] Received ${payload.event}:`, payload.data);

    // Log to Flask backend for audit persistence
    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
      await fetch(`${backendUrl}/api/governance/audit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entity_type: payload.event === "policy.updated" ? "policy" : "approval",
          entity_id: payload.data.approval_id || payload.data.policy_id,
          action: payload.event,
          description: `Harbyx webhook: ${payload.event}`,
          metadata: payload.data,
          harbyx_event_id: webhookEvent.id,
        }),
      });
    } catch (auditError) {
      console.error("[Webhook] Failed to log to backend:", auditError);
      // Don't fail the webhook if audit logging fails
    }

    // Handle specific events
    switch (payload.event) {
      case "approval.created":
        console.log(
          `[Webhook] New approval request: ${payload.data.approval_id}`
        );
        break;

      case "approval.decided":
        console.log(
          `[Webhook] Approval ${payload.data.approval_id} ${payload.data.decision}`
        );
        break;

      case "policy.updated":
        console.log(`[Webhook] Policy updated: ${payload.data.policy_name}`);
        break;
    }

    return NextResponse.json({
      success: true,
      event_id: webhookEvent.id,
      received_at: webhookEvent.timestamp,
    });
  } catch (error) {
    console.error("[Webhook] Error processing webhook:", error);
    return NextResponse.json(
      { error: "Failed to process webhook" },
      { status: 500 }
    );
  }
}

// GET endpoint to list recent events (for debugging/monitoring)
export async function GET(request: NextRequest) {
  const since = request.nextUrl.searchParams.get("since");
  const events = eventStore.getRecent(since ?? undefined);

  return NextResponse.json({
    events,
    count: events.length,
    timestamp: new Date().toISOString(),
  });
}
