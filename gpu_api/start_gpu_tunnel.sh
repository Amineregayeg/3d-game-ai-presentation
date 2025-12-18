#!/bin/bash
# SSH Tunnel from VPS to GPU API
# Forwards GPU port 5001 to localhost:5001 on VPS

GPU_HOST="80.188.223.202"
GPU_PORT="17757"
LOCAL_PORT="5001"
REMOTE_PORT="5001"

echo "Starting SSH tunnel to GPU server..."
echo "  GPU: $GPU_HOST:$GPU_PORT"
echo "  Forwarding: localhost:$LOCAL_PORT -> GPU:$REMOTE_PORT"

# Kill existing tunnel if running
pkill -f "ssh.*$GPU_PORT.*$LOCAL_PORT:localhost:$REMOTE_PORT" 2>/dev/null

# Start tunnel in background
ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -N -L $LOCAL_PORT:localhost:$REMOTE_PORT \
    -p $GPU_PORT root@$GPU_HOST &

TUNNEL_PID=$!
echo "Tunnel started with PID: $TUNNEL_PID"

# Wait a moment and check
sleep 2
if kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "Tunnel is running successfully"
    echo "Test with: curl http://localhost:5001/health"
else
    echo "Tunnel failed to start!"
    exit 1
fi
