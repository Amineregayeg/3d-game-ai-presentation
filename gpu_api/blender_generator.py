"""
Blender MCP Generator API
Connects to running Blender MCP server and uses Claude to generate 3D models
Flow: User Query + RAG Response → Claude → MCP Tools → Blender → GLB Export
"""

import os
import json
import socket
import time
import base64
import re
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
import anthropic

blender_gen = Blueprint('blender_gen', __name__)

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BLENDER_MCP_HOST = os.environ.get("BLENDER_MCP_HOST", "localhost")
BLENDER_MCP_PORT = int(os.environ.get("BLENDER_MCP_PORT", "9876"))
EXPORTS_PATH = "/tmp/blender_exports"
os.makedirs(EXPORTS_PATH, exist_ok=True)

# Initialize Anthropic client
client = None
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


class BlenderMCPConnection:
    """Direct connection to Blender MCP socket server"""

    def __init__(self, host: str = "localhost", port: int = 9876):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(30)
            self.sock.connect((self.host, self.port))
            return True
        except Exception as e:
            print(f"[MCP] Connection failed: {e}")
            self.sock = None
            return False

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

    def send_command(self, command: dict) -> dict:
        """Send JSON command to Blender and get response"""
        if not self.sock:
            if not self.connect():
                return {"error": "Not connected to Blender"}

        try:
            # Send command
            data = json.dumps(command).encode('utf-8')
            self.sock.sendall(data + b'\n')

            # Receive response
            response = b""
            while True:
                chunk = self.sock.recv(8192)
                if not chunk:
                    break
                response += chunk
                if b'\n' in chunk:
                    break

            return json.loads(response.decode('utf-8'))
        except Exception as e:
            return {"error": str(e)}

    def execute_code(self, code: str) -> dict:
        """Execute Python code in Blender"""
        return self.send_command({
            "type": "execute_code",
            "params": {"code": code}
        })

    def get_scene_info(self) -> dict:
        """Get current Blender scene information"""
        return self.send_command({
            "type": "get_scene_info",
            "params": {}
        })


# MCP Tools definition for Claude
MCP_TOOLS = [
    {
        "name": "execute_blender_code",
        "description": "Execute Python code in Blender to create or modify 3D objects. Use bpy module.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in Blender (must use bpy module)"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "get_scene_info",
        "description": "Get information about the current Blender scene including objects, materials, and lights",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "export_model",
        "description": "Export the current scene to GLB format",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to export the GLB file"
                }
            },
            "required": ["filepath"]
        }
    }
]


def call_mcp_tool(conn: BlenderMCPConnection, tool_name: str, tool_input: dict) -> str:
    """Execute an MCP tool call on Blender"""

    def parse_result(result: dict) -> str:
        """Parse Blender MCP response format"""
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("status") == "error":
            return f"Error: {result.get('message', 'Unknown error')}"
        if result.get("status") == "success":
            inner_result = result.get("result", {})
            if isinstance(inner_result, dict):
                return json.dumps(inner_result, indent=2)
            return str(inner_result) if inner_result else "Success"
        return json.dumps(result, indent=2)

    if tool_name == "execute_blender_code":
        code = tool_input.get("code", "")
        result = conn.execute_code(code)
        return parse_result(result)

    elif tool_name == "get_scene_info":
        result = conn.get_scene_info()
        return parse_result(result)

    elif tool_name == "export_model":
        filepath = tool_input.get("filepath", "/tmp/model.glb")
        export_code = f'''
import bpy
bpy.ops.export_scene.gltf(
    filepath="{filepath}",
    export_format='GLB',
    use_selection=False
)
print("Exported to {filepath}")
'''
        result = conn.execute_code(export_code)
        parsed = parse_result(result)
        if "Error" in parsed:
            return parsed
        return f"Model exported to {filepath}"

    return f"Unknown tool: {tool_name}"


SYSTEM_PROMPT = """You are a 3D modeling assistant that creates objects in Blender using Python code.

You have access to a running Blender instance through MCP tools. Use these tools to:
1. execute_blender_code - Run Python code in Blender
2. get_scene_info - Check current scene state
3. export_model - Export the final model to GLB

When creating 3D models:
- Always clear the scene first (delete default objects)
- Create geometry using bpy.ops.mesh.primitive_* functions
- Apply materials using bpy.data.materials and Principled BSDF
- Add appropriate lighting
- Export to GLB when done

The user's request and context from the RAG system are provided. Use the RAG context to inform your modeling decisions."""


def generate_with_mcp(user_query: str, rag_response: str, export_path: str) -> dict:
    """Use Claude + MCP to generate 3D model based on query and RAG response"""

    if not client:
        return {"error": "Anthropic client not initialized"}

    # Connect to Blender
    conn = BlenderMCPConnection(BLENDER_MCP_HOST, BLENDER_MCP_PORT)
    if not conn.connect():
        return {"error": f"Cannot connect to Blender MCP at {BLENDER_MCP_HOST}:{BLENDER_MCP_PORT}"}

    try:
        # Build the prompt with both query and RAG context
        user_message = f"""User Request: {user_query}

RAG Context (use this to inform your 3D modeling):
{rag_response}

Please create the 3D model requested by the user. After creating it, export to: {export_path}

Start by clearing the scene, then create the geometry, apply materials, add lighting, and export."""

        messages = [{"role": "user", "content": user_message}]
        tool_calls_log = []

        # Initial Claude call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,
            messages=messages
        )

        # Process tool calls in a loop
        max_iterations = 15
        iteration = 0

        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            tool_results = []
            assistant_content = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append(block)
                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"[MCP] Tool call: {tool_name}")
                    tool_calls_log.append({
                        "name": tool_name,
                        "input": tool_input
                    })

                    # Execute tool on Blender
                    result = call_mcp_tool(conn, tool_name, tool_input)

                    assistant_content.append(block)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            # Continue conversation
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=MCP_TOOLS,
                messages=messages
            )

        # Extract final response
        final_response = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text

        # Check if model was exported
        has_model = os.path.exists(export_path)

        return {
            "success": True,
            "response": final_response,
            "tool_calls": tool_calls_log,
            "has_model": has_model,
            "model_path": export_path if has_model else None,
            "iterations": iteration
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "tool_calls": tool_calls_log if 'tool_calls_log' in dir() else []}

    finally:
        conn.disconnect()


@blender_gen.route("/api/blender/health", methods=["GET"])
def health():
    """Health check"""
    # Test connection to Blender
    conn = BlenderMCPConnection(BLENDER_MCP_HOST, BLENDER_MCP_PORT)
    blender_connected = conn.connect()
    conn.disconnect()

    return jsonify({
        "status": "healthy",
        "blender_mcp_host": BLENDER_MCP_HOST,
        "blender_mcp_port": BLENDER_MCP_PORT,
        "blender_connected": blender_connected,
        "anthropic_available": client is not None,
        "api_key_set": bool(ANTHROPIC_API_KEY)
    })


@blender_gen.route("/api/blender/generate", methods=["POST"])
def generate_3d():
    """Generate 3D model from user query + RAG response"""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data["query"]
        rag_response = data.get("rag_response", "No additional context available.")

        if not client:
            return jsonify({
                "error": "Anthropic client not initialized",
                "details": "Check ANTHROPIC_API_KEY"
            }), 500

        # Generate unique export path
        timestamp = int(time.time() * 1000)
        export_filename = f"model_{timestamp}.glb"
        export_path = os.path.join(EXPORTS_PATH, export_filename)

        # Generate with MCP
        print(f"[Blender] Generating: {query[:50]}...")
        result = generate_with_mcp(query, rag_response, export_path)

        if "error" in result and not result.get("success"):
            return jsonify(result), 500

        # Read model as base64 if it exists
        model_base64 = None
        if result.get("has_model") and os.path.exists(export_path):
            try:
                with open(export_path, "rb") as f:
                    model_base64 = base64.b64encode(f.read()).decode()
                print(f"[Blender] Model exported: {export_filename}")
            except Exception as e:
                print(f"[Blender] Failed to read model: {e}")

        processing_time = time.time() - start_time

        return jsonify({
            "success": result.get("success", False),
            "has_model": result.get("has_model", False),
            "model_base64": model_base64,
            "model_filename": export_filename if model_base64 else None,
            "model_url": f"/api/blender/model/{export_filename}" if model_base64 else None,
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "processing_time": round(processing_time, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@blender_gen.route("/api/blender/model/<filename>", methods=["GET"])
def get_model(filename):
    """Serve exported model file"""
    filename = os.path.basename(filename)
    file_path = os.path.join(EXPORTS_PATH, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="model/gltf-binary")
    return jsonify({"error": "Model not found"}), 404


@blender_gen.route("/api/blender/tools", methods=["GET"])
def list_tools():
    """List available MCP tools"""
    return jsonify({
        "tools": [{"name": t["name"], "description": t["description"]} for t in MCP_TOOLS],
        "mode": "mcp_direct"
    })
