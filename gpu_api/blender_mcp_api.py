#!/usr/bin/env python3
"""
Blender MCP API Service
Uses Claude + Blender MCP to generate 3D models from natural language
"""

import os
import sys
import json
import subprocess
import threading
import queue
import time
import tempfile
import base64
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BLENDER_MCP_CMD = os.environ.get("BLENDER_MCP_CMD", "blender-mcp")
EXPORTS_PATH = "/tmp/blender_exports"
os.makedirs(EXPORTS_PATH, exist_ok=True)

class BlenderMCPClient:
    """Client for Blender MCP server with Claude integration"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.mcp_process = None
        self.tools = []
        self.output_queue = queue.Queue()
        self.reader_thread = None
        self.request_id = 1000
        self.is_connected = False

    def _read_output(self):
        """Background thread to read MCP output"""
        while self.mcp_process and self.mcp_process.poll() is None:
            try:
                line = self.mcp_process.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
            except Exception:
                break

    def _get_response(self, timeout=30):
        """Get response from queue with timeout"""
        try:
            response = self.output_queue.get(timeout=timeout)
            return json.loads(response)
        except queue.Empty:
            return None
        except json.JSONDecodeError:
            return None

    def _send_request(self, req):
        """Send request to MCP server"""
        if self.mcp_process and self.mcp_process.poll() is None:
            self.mcp_process.stdin.write(json.dumps(req) + '\n')
            self.mcp_process.stdin.flush()

    def start(self):
        """Start the Blender MCP server"""
        try:
            # Use xvfb for headless Blender
            env = os.environ.copy()
            env["DISPLAY"] = ":99"

            self.mcp_process = subprocess.Popen(
                [BLENDER_MCP_CMD],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env
            )

            # Start background reader thread
            self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self.reader_thread.start()

            time.sleep(2)
            self.discover_tools()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to start Blender MCP: {e}")
            return False

    def discover_tools(self):
        """Discover available tools from Blender MCP"""
        init_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "blender-mcp-api",
                    "version": "1.0.0"
                }
            }
        }
        self.request_id += 1

        try:
            self._send_request(init_request)
            init_response = self._get_response(timeout=10)

            if not init_response:
                print("No response from MCP initialization")
                return

            tools_request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/list"
            }
            self.request_id += 1

            self._send_request(tools_request)
            tools_response = self._get_response(timeout=10)

            if tools_response and 'result' in tools_response and 'tools' in tools_response['result']:
                self.tools = self._convert_tools(tools_response['result']['tools'])
                print(f"Loaded {len(self.tools)} Blender tools")
        except Exception as e:
            print(f"Failed to discover tools: {e}")

    def _convert_tools(self, mcp_tools):
        """Convert MCP tool format to Claude API format"""
        claude_tools = []
        for tool in mcp_tools:
            claude_tools.append({
                "name": tool['name'],
                "description": tool.get('description', ''),
                "input_schema": tool.get('inputSchema', {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            })
        return claude_tools

    def call_tool(self, tool_name: str, tool_input: dict) -> str:
        """Call a tool on the Blender MCP server"""
        tool_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input
            }
        }
        self.request_id += 1

        try:
            self._send_request(tool_request)
            response = self._get_response(timeout=60)

            if not response:
                return f"Timeout: No response from tool '{tool_name}'"

            if 'result' in response:
                content = response['result'].get('content', [])
                if content and len(content) > 0:
                    return content[0].get('text', str(response['result']))
                return str(response['result'])
            elif 'error' in response:
                return f"Error: {response['error']}"
            else:
                return f"Unexpected response: {response}"
        except Exception as e:
            return f"Tool execution error: {e}"

    def generate(self, user_query: str) -> dict:
        """Generate 3D content based on user query using Claude + Blender MCP"""
        if not self.is_connected:
            return {"error": "Blender MCP not connected"}

        if not self.tools:
            return {"error": "No Blender tools available"}

        conversation = [{
            "role": "user",
            "content": f"""You are a 3D modeling assistant using Blender.

The user wants: {user_query}

Create the 3D model using the available Blender tools. After creating the model:
1. Apply appropriate materials/colors if mentioned
2. Position the object nicely in the scene
3. Export the result as GLB format to /tmp/blender_exports/model.glb

Be creative but practical. Focus on creating what the user asked for."""
        }]

        tool_calls_log = []

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=conversation
            )

            # Handle tool calls
            while response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append(block)
                    elif block.type == "tool_use":
                        tool_calls_log.append({
                            "name": block.name,
                            "input": block.input
                        })

                        result = self.call_tool(block.name, block.input)

                        assistant_content.append(block)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                conversation.append({
                    "role": "assistant",
                    "content": assistant_content
                })

                conversation.append({
                    "role": "user",
                    "content": tool_results
                })

                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    tools=self.tools,
                    messages=conversation
                )

            # Extract final text response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text

            # Check if GLB was created
            glb_path = Path(EXPORTS_PATH) / "model.glb"
            has_model = glb_path.exists()

            return {
                "success": True,
                "response": final_response,
                "tool_calls": tool_calls_log,
                "has_model": has_model,
                "model_path": str(glb_path) if has_model else None
            }

        except Exception as e:
            return {"error": str(e), "tool_calls": tool_calls_log}

    def cleanup(self):
        """Clean up MCP server process"""
        if self.mcp_process:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
            except:
                self.mcp_process.kill()
            self.mcp_process = None
            self.is_connected = False


# Global client instance
mcp_client = None

def get_client():
    global mcp_client
    if mcp_client is None and ANTHROPIC_API_KEY:
        mcp_client = BlenderMCPClient(ANTHROPIC_API_KEY)
        mcp_client.start()
    return mcp_client


@app.route("/health", methods=["GET"])
def health():
    client = get_client()
    return jsonify({
        "status": "healthy",
        "blender_mcp": client.is_connected if client else False,
        "tools_count": len(client.tools) if client else 0,
        "api_key_set": bool(ANTHROPIC_API_KEY)
    })


@app.route("/api/blender/generate", methods=["POST"])
def generate_3d():
    """Generate 3D model from natural language query"""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data["query"]

        client = get_client()
        if not client:
            return jsonify({"error": "Blender MCP not initialized. Check ANTHROPIC_API_KEY"}), 500

        result = client.generate(query)

        if "error" in result:
            return jsonify(result), 500

        # If model was generated, read it as base64
        model_base64 = None
        if result.get("has_model") and result.get("model_path"):
            try:
                with open(result["model_path"], "rb") as f:
                    model_base64 = base64.b64encode(f.read()).decode()
            except Exception as e:
                print(f"Failed to read model file: {e}")

        processing_time = time.time() - start_time

        return jsonify({
            "success": True,
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "has_model": result.get("has_model", False),
            "model_base64": model_base64,
            "processing_time": round(processing_time, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/blender/model/<filename>", methods=["GET"])
def get_model(filename):
    """Serve a generated model file"""
    file_path = Path(EXPORTS_PATH) / filename
    if file_path.exists():
        return send_file(file_path, mimetype="model/gltf-binary")
    return jsonify({"error": "Model not found"}), 404


@app.route("/api/blender/tools", methods=["GET"])
def list_tools():
    """List available Blender MCP tools"""
    client = get_client()
    if not client:
        return jsonify({"error": "Not connected"}), 500

    return jsonify({
        "tools": [{"name": t["name"], "description": t["description"]} for t in client.tools]
    })


if __name__ == "__main__":
    print("=" * 50)
    print("Blender MCP API Server")
    print("=" * 50)
    print(f"API Key set: {bool(ANTHROPIC_API_KEY)}")
    print(f"Blender MCP command: {BLENDER_MCP_CMD}")
    print("=" * 50)

    # Start xvfb for headless Blender
    try:
        subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["DISPLAY"] = ":99"
        print("Xvfb started on :99")
    except:
        print("Warning: Xvfb not available, Blender may not work headlessly")

    # Pre-initialize client
    print("Initializing Blender MCP client...")
    client = get_client()
    if client and client.is_connected:
        print(f"Connected with {len(client.tools)} tools")
    else:
        print("Warning: Failed to connect to Blender MCP")

    app.run(host="0.0.0.0", port=5002, debug=False)
