import { NextRequest } from "next/server";
import { eventStore, WebhookEvent } from "../webhook/route";

// Server-Sent Events endpoint for real-time approval updates

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();

  // Create a readable stream for SSE
  const stream = new ReadableStream({
    start(controller) {
      // Send initial connection message
      const connectMessage = `data: ${JSON.stringify({
        type: "connected",
        agentId: "salesforce-consultant-agent",
        timestamp: new Date().toISOString(),
      })}\n\n`;
      controller.enqueue(encoder.encode(connectMessage));

      // Subscribe to new events
      const unsubscribe = eventStore.subscribe((event: WebhookEvent) => {
        const message = `data: ${JSON.stringify({
          type: "event",
          event: event.event,
          data: event.data,
          timestamp: event.timestamp,
        })}\n\n`;
        controller.enqueue(encoder.encode(message));
      });

      // Send heartbeat every 30 seconds to keep connection alive
      const heartbeat = setInterval(() => {
        const ping = `data: ${JSON.stringify({
          type: "heartbeat",
          timestamp: new Date().toISOString(),
        })}\n\n`;
        controller.enqueue(encoder.encode(ping));
      }, 30000);

      // Clean up on connection close
      request.signal.addEventListener("abort", () => {
        unsubscribe();
        clearInterval(heartbeat);
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no", // Disable buffering for nginx
    },
  });
}
