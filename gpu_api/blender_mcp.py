"""
Blender MCP Blueprint for Flask Backend
Uses Claude + Blender MCP to generate 3D models from natural language
"""

import os
import json
import subprocess
import threading
import queue
import time
import tempfile
import base64
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file

blender_mcp = Blueprint('blender_mcp', __name__)

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BLENDER_MCP_CMD = "/root/.local/bin/blender-mcp"
EXPORTS_PATH = "/tmp/blender_exports"
os.makedirs(EXPORTS_PATH, exist_ok=True)

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic module not available")


class BlenderMCPClient:
    """Client for Blender MCP server with Claude integration"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.mcp_process = None
        self.tools = []
        self.output_queue = queue.Queue()
        self.reader_thread = None
        self.request_id = 1000
        self.is_connected = False
        self.connection_error = None

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

            time.sleep(3)

            if self.mcp_process.poll() is not None:
                stderr = self.mcp_process.stderr.read()
                self.connection_error = f"MCP process died: {stderr}"
                return False

            self.discover_tools()
            self.is_connected = True
            return True
        except Exception as e:
            self.connection_error = str(e)
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
        if not self.client:
            return {"error": "Anthropic client not initialized"}

        if not self.is_connected:
            return {"error": f"Blender MCP not connected: {self.connection_error}"}

        if not self.tools:
            return {"error": "No Blender tools available"}

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        export_filename = f"model_{timestamp}.glb"
        export_path = os.path.join(EXPORTS_PATH, export_filename)

        conversation = [{
            "role": "user",
            "content": f"""You are a 3D modeling assistant using Blender. The user wants: {user_query}

Create the 3D model using the available Blender tools:
1. First clear the scene or create a new one
2. Create the requested geometry (cube, sphere, cylinder, etc)
3. Apply materials/colors if mentioned
4. Position objects appropriately
5. Export the result as GLB to: {export_path}

Keep it simple and practical. Focus on creating what the user asked for."""
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
            max_iterations = 20
            iteration = 0

            while response.stop_reason == "tool_use" and iteration < max_iterations:
                iteration += 1
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
            has_model = os.path.exists(export_path)

            return {
                "success": True,
                "response": final_response,
                "tool_calls": tool_calls_log,
                "has_model": has_model,
                "model_path": export_path if has_model else None,
                "model_filename": export_filename if has_model else None
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
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
_mcp_client = None
_xvfb_process = None

def ensure_xvfb():
    """Ensure Xvfb is running for headless Blender"""
    global _xvfb_process
    if _xvfb_process is None or _xvfb_process.poll() is not None:
        try:
            _xvfb_process = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1024x768x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            os.environ["DISPLAY"] = ":99"
            time.sleep(1)
            print("Xvfb started on :99")
        except Exception as e:
            print(f"Warning: Xvfb not available: {e}")

def get_client():
    global _mcp_client
    if _mcp_client is None and ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
        ensure_xvfb()
        _mcp_client = BlenderMCPClient(ANTHROPIC_API_KEY)
        _mcp_client.start()
    return _mcp_client


@blender_mcp.route("/api/blender/health", methods=["GET"])
def health():
    client = get_client()
    return jsonify({
        "status": "healthy",
        "anthropic_available": ANTHROPIC_AVAILABLE,
        "blender_mcp_connected": client.is_connected if client else False,
        "tools_count": len(client.tools) if client else 0,
        "api_key_set": bool(ANTHROPIC_API_KEY),
        "connection_error": client.connection_error if client else "Client not initialized"
    })


@blender_mcp.route("/api/blender/generate", methods=["POST"])
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
            return jsonify({
                "error": "Blender MCP not initialized",
                "details": "Check ANTHROPIC_API_KEY environment variable"
            }), 500

        if not client.is_connected:
            return jsonify({
                "error": "Blender MCP not connected",
                "details": client.connection_error
            }), 500

        result = client.generate(query)

        if "error" in result and not result.get("success"):
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
            "model_filename": result.get("model_filename"),
            "processing_time": round(processing_time, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@blender_mcp.route("/api/blender/model/<filename>", methods=["GET"])
def get_model(filename):
    """Serve a generated model file"""
    # Sanitize filename
    filename = os.path.basename(filename)
    file_path = os.path.join(EXPORTS_PATH, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="model/gltf-binary")
    return jsonify({"error": "Model not found"}), 404


@blender_mcp.route("/api/blender/tools", methods=["GET"])
def list_tools():
    """List available Blender MCP tools"""
    client = get_client()
    if not client or not client.is_connected:
        return jsonify({"error": "Not connected", "tools": []}), 200

    return jsonify({
        "tools": [{"name": t["name"], "description": t["description"]} for t in client.tools]
    })
