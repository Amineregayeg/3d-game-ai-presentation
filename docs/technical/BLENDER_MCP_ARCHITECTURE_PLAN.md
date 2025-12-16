# Component 4: Blender MCP Integration Architecture
## AI-Powered 3D Asset Generation via Model Context Protocol

**Technical Planning Document v1.0**
**Date:** December 4, 2025
**Status:** Production-Ready
**MCP Server:** BlenderMCP by Siddharth Ahuja
**Protocol:** Model Context Protocol (MCP) by Anthropic

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Context Protocol (MCP) Overview](#model-context-protocol-mcp-overview)
   - 2.1 What is MCP?
   - 2.2 MCP Architecture
   - 2.3 Core Primitives
   - 2.4 Transport Mechanisms
3. [BlenderMCP Architecture](#blendermcp-architecture)
   - 3.1 System Components
   - 3.2 Socket Communication Protocol
   - 3.3 Message Format & JSON-RPC
   - 3.4 Connection Management
4. [MCP Tools Reference](#mcp-tools-reference)
   - 4.1 Scene Operations
   - 4.2 Object Manipulation
   - 4.3 Material & Texture Control
   - 4.4 Code Execution
   - 4.5 Asset Integrations
5. [External Asset Integrations](#external-asset-integrations)
   - 5.1 Poly Haven (HDRI, Textures, Models)
   - 5.2 Sketchfab (3D Model Marketplace)
   - 5.3 Hyper3D Rodin (AI 3D Generation)
   - 5.4 Hunyuan3D (Tencent AI Generation)
6. [Blender Addon Implementation](#blender-addon-implementation)
   - 6.1 Socket Server Architecture
   - 6.2 Command Handlers
   - 6.3 Material Node Graph Construction
   - 6.4 Asset Import Pipeline
7. [Game Engine Integration](#game-engine-integration)
   - 7.1 Asset Export Formats
   - 7.2 Unity Import Workflow
   - 7.3 Unreal Engine 5 Import Workflow
   - 7.4 Texture & Material Conversion
8. [Security Considerations](#security-considerations)
   - 8.1 Code Execution Risks
   - 8.2 API Key Management
   - 8.3 File System Access
   - 8.4 Network Security
9. [Performance Optimization](#performance-optimization)
   - 9.1 Asset Caching Strategy
   - 9.2 Batch Operations
   - 9.3 Background Processing
   - 9.4 Memory Management
10. [3D Game AI Assistant Integration](#3d-game-ai-assistant-integration)
    - 10.1 RAG Context for Blender Commands
    - 10.2 Natural Language → MCP Tool Mapping
    - 10.3 Asset Request Workflow
    - 10.4 Iterative Refinement Loop
11. [Installation & Configuration](#installation--configuration)
    - 11.1 Prerequisites
    - 11.2 Blender Addon Setup
    - 11.3 MCP Server Configuration
    - 11.4 Claude Desktop Integration
12. [Error Handling & Resilience](#error-handling--resilience)
    - 12.1 Connection Recovery
    - 12.2 Timeout Management
    - 12.3 Fallback Strategies
    - 12.4 Logging & Monitoring
13. [Implementation Roadmap](#implementation-roadmap)
    - 13.1 Phase 1: Basic MCP Setup
    - 13.2 Phase 2: Asset Integration
    - 13.3 Phase 3: Game Engine Bridge
    - 13.4 Phase 4: Production Hardening
14. [API Reference](#api-reference)
    - 14.1 MCP Server Endpoints
    - 14.2 Blender Addon Commands
    - 14.3 Configuration Options
15. [Appendix](#appendix)
    - 15.1 Supported Blender Versions
    - 15.2 Asset Source Comparison
    - 15.3 Troubleshooting Guide

---

## Executive Summary

Component 4 enables AI-powered 3D asset generation through the **Model Context Protocol (MCP)**, connecting Claude AI directly to Blender for real-time 3D modeling, scene creation, and asset management.

### Key Capabilities

**BlenderMCP Integration:**
- **24 MCP Tools** for comprehensive Blender control
- **Socket-based communication** (localhost:9876)
- **JSON-RPC protocol** for reliable message passing
- **Two-way interaction** between Claude AI and Blender

**Asset Sources (Priority Order):**
1. **Sketchfab** - 3D model marketplace (downloadable models)
2. **Poly Haven** - Free HDRIs, textures, and models
3. **Hyper3D Rodin** - AI-generated 3D from text/images
4. **Hunyuan3D** - Tencent's AI 3D generation
5. **Python Scripting** - Custom procedural generation

**Performance Targets:**

| Metric | Target | Notes |
|--------|--------|-------|
| **Socket Response** | <100ms | Local connection |
| **Scene Info Retrieval** | <50ms | Cached metadata |
| **Asset Import** | <30s | Depends on asset size |
| **AI Model Generation** | 30-120s | External API dependent |
| **Export to Game Engine** | <10s | FBX/GLTF format |

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    3D GAME AI ASSISTANT - COMPONENT 4                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input: Natural language request from user                               │
│    "Create a low-poly medieval sword with metallic material"             │
│    ↓                                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  CLAUDE AI (LLM)                                                    │  │
│  │  - Parse user intent                                                │  │
│  │  - Query RAG for Blender patterns                                   │  │
│  │  - Select appropriate MCP tools                                     │  │
│  │  - Generate tool call parameters                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│    ↓                                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  MCP SERVER (Python)                                                │  │
│  │  - Receive tool calls from Claude                                   │  │
│  │  - Validate parameters                                              │  │
│  │  - Forward to Blender via socket                                    │  │
│  │  - Return results to Claude                                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│    ↓ TCP Socket (localhost:9876)                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  BLENDER ADDON                                                      │  │
│  │  - Socket server listener                                           │  │
│  │  - Command execution in main thread                                 │  │
│  │  - Python API access (bpy)                                          │  │
│  │  - Response formatting                                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│    ↓                                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  BLENDER 3D                                                         │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │  │
│  │  │ Create Mesh  │ │ Apply Mats   │ │ Add Mods     │                │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │  │
│  │  │ Import Asset │ │ Scene Setup  │ │ Export FBX   │                │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│    ↓                                                                      │
│  Output: 3D asset ready for game engine import (FBX, GLTF, OBJ)          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Model Context Protocol (MCP) Overview

### 2.1 What is MCP?

The **Model Context Protocol (MCP)** is an open standard introduced by Anthropic in November 2024 to standardize how AI systems integrate with external tools, data sources, and services.

**Key Benefits:**
- **Universal Standard** - Single protocol replaces fragmented integrations
- **Modular Architecture** - Servers and clients operate independently
- **Tool Invocation** - LLMs can call functions and receive structured results
- **Context Sharing** - Bidirectional data exchange between AI and tools

**Adoption (2025):**
- Thousands of community-built MCP servers
- SDKs for Python, TypeScript, Kotlin, Java
- Adopted by OpenAI, Google DeepMind, and major AI providers
- De facto standard for AI-to-tool communication

### 2.2 MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │   MCP CLIENT     │◄────────►│   MCP SERVER     │             │
│  │  (AI Application)│  JSON-RPC│  (Tool Provider) │             │
│  │                  │          │                  │             │
│  │  - Claude Desktop│          │  - BlenderMCP    │             │
│  │  - Cursor IDE    │          │  - GitHub MCP    │             │
│  │  - VS Code       │          │  - Postgres MCP  │             │
│  │  - Custom Apps   │          │  - Custom MCPs   │             │
│  └──────────────────┘          └──────────────────┘             │
│           │                            │                         │
│           │                            │                         │
│           ▼                            ▼                         │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │   LLM (Claude)   │          │  External System │             │
│  │                  │          │                  │             │
│  │  - Tool calling  │          │  - Blender       │             │
│  │  - Context mgmt  │          │  - Databases     │             │
│  │  - Response gen  │          │  - APIs          │             │
│  └──────────────────┘          └──────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Core Primitives

MCP defines five core message types (primitives):

| Primitive | Description | BlenderMCP Example |
|-----------|-------------|-------------------|
| **Prompts** | Prepared instruction templates | Asset creation strategy |
| **Resources** | Structured data (documents, code) | Scene information |
| **Tools** | Executable functions | `create_mesh`, `apply_material` |
| **Roots** | File system entry points | Blender project directories |
| **Sampling** | Request AI completion | Iterative refinement |

### 2.4 Transport Mechanisms

**Supported Transports:**
- **stdio** - Standard input/output (process-based)
- **HTTP** - REST API with optional SSE (Server-Sent Events)
- **TCP Sockets** - Direct socket connection (BlenderMCP uses this)

**BlenderMCP Transport:**
```
Protocol: TCP/JSON-RPC over sockets
Host: localhost (configurable via BLENDER_HOST)
Port: 9876 (configurable via BLENDER_PORT)
Timeout: 180 seconds
```

---

## BlenderMCP Architecture

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLENDERMCP SYSTEM COMPONENTS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Component 1: MCP Server (Python)                               │
│  ├── Location: src/blender_mcp/server.py                        │
│  ├── Framework: MCP Python SDK                                  │
│  ├── Purpose: Implement MCP protocol, expose tools              │
│  └── Connects to: Claude AI (client), Blender Addon (server)    │
│                                                                  │
│  Component 2: Blender Addon (Python)                            │
│  ├── Location: addon.py                                         │
│  ├── Framework: Blender Python API (bpy)                        │
│  ├── Purpose: Socket server, command execution                  │
│  └── Runs within: Blender 3.6+                                  │
│                                                                  │
│  Component 3: External Integrations                             │
│  ├── Poly Haven API (HDRIs, textures, models)                   │
│  ├── Sketchfab API (3D model marketplace)                       │
│  ├── Hyper3D Rodin API (AI 3D generation)                       │
│  └── Hunyuan3D API (Tencent AI generation)                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Socket Communication Protocol

**Connection Flow:**

```
MCP Server                    Blender Addon
    │                              │
    │──── TCP Connect ────────────►│ Port 9876
    │                              │
    │◄─── Connection ACK ──────────│
    │                              │
    │──── JSON Command ───────────►│
    │     {                        │
    │       "type": "get_scene_info",
    │       "params": {}           │
    │     }                        │
    │                              │
    │◄─── JSON Response ───────────│
    │     {                        │
    │       "status": "success",   │
    │       "result": {...}        │
    │     }                        │
    │                              │
    │──── Keep-Alive ─────────────►│ Persistent connection
    │                              │
```

### 3.3 Message Format & JSON-RPC

**Command Message:**
```json
{
  "type": "execute_blender_code",
  "params": {
    "code": "import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)"
  }
}
```

**Response Message (Success):**
```json
{
  "status": "success",
  "result": {
    "output": "Cube created at origin",
    "object_name": "Cube"
  }
}
```

**Response Message (Error):**
```json
{
  "status": "error",
  "message": "Object 'NonExistent' not found in scene"
}
```

### 3.4 Connection Management

```python
# Global connection state (MCP Server)
blender_connection = None

def get_blender_connection():
    """
    Establish or reuse TCP connection to Blender addon.
    Implements automatic reconnection on failure.
    """
    global blender_connection

    if blender_connection is None:
        host = os.environ.get("BLENDER_HOST", "localhost")
        port = int(os.environ.get("BLENDER_PORT", "9876"))

        blender_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blender_connection.settimeout(180)  # 3 minute timeout
        blender_connection.connect((host, port))

    return blender_connection

def send_command(command_type: str, params: dict) -> dict:
    """
    Send command to Blender and receive response.
    Handles chunked responses and JSON validation.
    """
    conn = get_blender_connection()

    # Send command
    message = json.dumps({"type": command_type, "params": params})
    conn.sendall(message.encode('utf-8'))

    # Receive response (chunked)
    response_data = b""
    while True:
        chunk = conn.recv(8192)
        response_data += chunk

        # Validate complete JSON
        try:
            return json.loads(response_data.decode('utf-8'))
        except json.JSONDecodeError:
            continue  # More data expected
```

---

## MCP Tools Reference

### 4.1 Scene Operations

| Tool | Parameters | Description |
|------|------------|-------------|
| `get_scene_info` | None | Returns scene name, object count, object list (max 10) |
| `get_object_info` | `object_name: str` | Detailed object properties (location, rotation, scale, materials, mesh stats) |
| `get_viewport_screenshot` | `max_size: int = 800` | Captures 3D viewport as PNG, returns base64 |

**Example: get_scene_info Response**
```json
{
  "status": "success",
  "result": {
    "scene_name": "Scene",
    "object_count": 5,
    "objects": [
      {"name": "Camera", "type": "CAMERA"},
      {"name": "Light", "type": "LIGHT"},
      {"name": "Cube", "type": "MESH"},
      {"name": "Sphere", "type": "MESH"},
      {"name": "Sword", "type": "MESH"}
    ]
  }
}
```

### 4.2 Object Manipulation

**Primary Tool: `execute_blender_code`**

This powerful tool allows running arbitrary Python code in Blender's namespace.

**Parameters:**
- `code: str` - Python code to execute

**Example: Create a Low-Poly Sword**
```python
code = """
import bpy

# Clear selection
bpy.ops.object.select_all(action='DESELECT')

# Create blade (elongated cube)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1))
blade = bpy.context.active_object
blade.name = "Sword_Blade"
blade.scale = (0.1, 0.05, 1.5)

# Create handle (cylinder)
bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.5, location=(0, 0, -0.25))
handle = bpy.context.active_object
handle.name = "Sword_Handle"

# Create crossguard (cube)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.2))
guard = bpy.context.active_object
guard.name = "Sword_Guard"
guard.scale = (0.3, 0.05, 0.05)

# Select all sword parts
bpy.ops.object.select_all(action='DESELECT')
blade.select_set(True)
handle.select_set(True)
guard.select_set(True)
bpy.context.view_layer.objects.active = blade

# Join into single mesh
bpy.ops.object.join()
bpy.context.active_object.name = "Sword"

print("Sword created successfully!")
"""
```

### 4.3 Material & Texture Control

**Automatic Material Node Graph Construction:**

```python
def create_pbr_material(object_name: str, textures: dict):
    """
    Creates Principled BSDF material with proper texture mapping.

    textures = {
        "albedo": "/path/to/albedo.png",
        "roughness": "/path/to/roughness.png",
        "metallic": "/path/to/metallic.png",
        "normal": "/path/to/normal.png",
        "displacement": "/path/to/displacement.png"
    }
    """
    import bpy

    obj = bpy.data.objects[object_name]
    mat = bpy.data.materials.new(name=f"{object_name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get Principled BSDF
    bsdf = nodes.get("Principled BSDF")

    # Add texture nodes
    for tex_type, filepath in textures.items():
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(filepath)

        # Set color space
        if tex_type in ["roughness", "metallic", "normal"]:
            tex_node.image.colorspace_settings.name = "Non-Color"

        # Connect to appropriate input
        if tex_type == "albedo":
            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        elif tex_type == "roughness":
            links.new(tex_node.outputs["Color"], bsdf.inputs["Roughness"])
        elif tex_type == "metallic":
            links.new(tex_node.outputs["Color"], bsdf.inputs["Metallic"])
        elif tex_type == "normal":
            normal_map = nodes.new("ShaderNodeNormalMap")
            links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
```

### 4.4 Code Execution

**Security Warning:** The `execute_blender_code` tool can run arbitrary Python code in Blender, including file system access and network calls. Use with caution in production environments.

**Sandboxing Recommendations:**
1. Validate code before execution
2. Restrict file system paths
3. Disable network access in critical environments
4. Log all executed code for audit

### 4.5 Asset Integrations

**Asset Source Priority (Recommended):**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASSET CREATION STRATEGY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Priority 1: Sketchfab                                          │
│  ├── Best for: Specific named assets (sword, chair, car)        │
│  ├── Quality: Professional artist-created models                │
│  └── Latency: 10-30s (download + import)                        │
│                                                                  │
│  Priority 2: Poly Haven                                         │
│  ├── Best for: Environment assets, textures, HDRIs              │
│  ├── Quality: Photogrammetry-scanned, high quality              │
│  └── Latency: 5-20s (download + import)                         │
│                                                                  │
│  Priority 3: Hyper3D Rodin / Hunyuan3D                          │
│  ├── Best for: Custom assets not found elsewhere                │
│  ├── Quality: AI-generated (variable quality)                   │
│  └── Latency: 30-120s (generation + import)                     │
│                                                                  │
│  Priority 4: Python Scripting                                   │
│  ├── Best for: Procedural/parametric geometry                   │
│  ├── Quality: Dependent on script complexity                    │
│  └── Latency: <5s (local execution)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## External Asset Integrations

### 5.1 Poly Haven (HDRI, Textures, Models)

**API Endpoints:**
- Categories: `https://api.polyhaven.com/categories/{asset_type}`
- Search: `https://api.polyhaven.com/assets?t={type}`
- Download: `https://dl.polyhaven.org/file/ph-assets/{type}/{id}/{resolution}/`

**Supported Asset Types:**
| Type | Use Case | File Formats |
|------|----------|--------------|
| **HDRIs** | Environment lighting | EXR, HDR |
| **Textures** | PBR material textures | PNG (albedo, roughness, normal, displacement) |
| **Models** | 3D assets | GLTF, FBX, Blend |

**MCP Tools:**
```python
# Get available categories
get_polyhaven_categories(asset_type="textures")

# Search for assets
search_polyhaven_assets(
    asset_type="textures",
    categories=["brick", "metal"]
)

# Download and import
download_polyhaven_asset(
    asset_id="brick_wall_001",
    asset_type="textures",
    resolution="2k",
    file_format="png"
)
```

### 5.2 Sketchfab (3D Model Marketplace)

**API Endpoints:**
- Search: `https://api.sketchfab.com/v3/search?type=models&q={query}`
- Download: `https://api.sketchfab.com/v3/models/{uid}/download`

**Authentication:**
- API Token required (free tier available)
- Set via `SKETCHFAB_API_TOKEN` environment variable

**MCP Tools:**
```python
# Search models
search_sketchfab_models(
    query="medieval sword",
    categories=["weapons"],
    count=20,
    downloadable=True  # Only free/purchased models
)

# Download model
download_sketchfab_model(uid="abc123xyz")
```

**Import Process:**
1. Download GLTF archive
2. Extract to temporary directory
3. Import GLTF using Blender's importer
4. Clean up temporary files
5. Position in scene

### 5.3 Hyper3D Rodin (AI 3D Generation)

**Platforms:**
- **hyper3d.ai** - Official Hyper3D API
- **fal.ai** - Alternative hosting

**Generation Methods:**
| Method | Input | Best For |
|--------|-------|----------|
| **Text-to-3D** | Text prompt | General asset creation |
| **Image-to-3D** | Reference images | Specific object recreation |

**MCP Tools:**
```python
# Generate from text
generate_hyper3d_model_via_text(
    text_prompt="low poly medieval sword with leather wrapped handle",
    bbox_condition=[0.5, 0.1, 2.0]  # Optional bounding box [x, y, z]
)

# Generate from images
generate_hyper3d_model_via_images(
    input_image_urls=["https://example.com/sword_ref.jpg"],
    bbox_condition=None
)

# Poll job status
poll_rodin_job_status(subscription_key="job_abc123")

# Import generated model
import_generated_asset(
    name="GeneratedSword",
    task_uuid="uuid_xyz789"
)
```

**Generation Pipeline:**
```
Text/Image Input
    ↓
Hyper3D API (30-120s processing)
    ↓
Poll status until complete
    ↓
Download GLB file
    ↓
Import into Blender
    ↓
Position and clean up
```

### 5.4 Hunyuan3D (Tencent AI Generation)

**Platforms:**
- **Official Tencent Cloud API**
- **Local deployment (self-hosted)**

**Authentication:**
- Tencent Cloud SecretId and SecretKey
- Signature-based authentication (HMAC-SHA256)

**MCP Tools:**
```python
# Generate model
generate_hunyuan3d_model(
    text_prompt="fantasy crystal sword",
    input_image_url=None  # Optional reference image
)

# Poll status
poll_hunyuan_job_status(job_id="hunyuan_job_123")

# Import result
import_generated_asset_hunyuan(
    name="CrystalSword",
    zip_file_url="https://hunyuan.cloud.tencent.com/..."
)
```

**Output Format:**
- ZIP archive containing OBJ model
- Optional textures included
- Automatic material assignment on import

---

## Blender Addon Implementation

### 6.1 Socket Server Architecture

```python
# addon.py - Socket Server Implementation

import bpy
import socket
import threading
import json

class BlenderMCPServer:
    """
    TCP Socket server running within Blender.
    Receives JSON commands and executes in main thread.
    """

    def __init__(self, host="localhost", port=9876):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.command_queue = []
        self.response_queue = []

    def start(self):
        """Start socket server in daemon thread."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        # Start listener thread
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

        # Register timer for main thread execution
        bpy.app.timers.register(self._process_commands, first_interval=0.1)

        print(f"BlenderMCP Server started on {self.host}:{self.port}")

    def _listen_loop(self):
        """Accept connections and receive commands."""
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                data = client.recv(65536).decode('utf-8')
                command = json.loads(data)

                # Queue command for main thread
                self.command_queue.append((client, command))

            except Exception as e:
                print(f"Socket error: {e}")

    def _process_commands(self):
        """Execute commands in Blender's main thread (timer callback)."""
        if self.command_queue:
            client, command = self.command_queue.pop(0)

            try:
                result = self._execute_command(command)
                response = {"status": "success", "result": result}
            except Exception as e:
                response = {"status": "error", "message": str(e)}

            client.sendall(json.dumps(response).encode('utf-8'))
            client.close()

        return 0.1  # Continue timer

    def _execute_command(self, command):
        """Route command to appropriate handler."""
        cmd_type = command.get("type")
        params = command.get("params", {})

        handlers = {
            "get_scene_info": self._handle_get_scene_info,
            "get_object_info": self._handle_get_object_info,
            "execute_code": self._handle_execute_code,
            # ... more handlers
        }

        handler = handlers.get(cmd_type)
        if handler:
            return handler(params)
        else:
            raise ValueError(f"Unknown command type: {cmd_type}")
```

### 6.2 Command Handlers

**Scene Info Handler:**
```python
def _handle_get_scene_info(self, params):
    """Return scene metadata."""
    scene = bpy.context.scene
    objects = []

    for obj in scene.objects[:10]:  # Limit to 10 objects
        objects.append({
            "name": obj.name,
            "type": obj.type,
            "location": list(obj.location),
            "visible": obj.visible_get()
        })

    return {
        "scene_name": scene.name,
        "object_count": len(scene.objects),
        "objects": objects
    }
```

**Object Info Handler:**
```python
def _handle_get_object_info(self, params):
    """Return detailed object properties."""
    obj_name = params.get("object_name")
    obj = bpy.data.objects.get(obj_name)

    if not obj:
        raise ValueError(f"Object '{obj_name}' not found")

    info = {
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "rotation": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "materials": [m.name for m in obj.data.materials] if hasattr(obj.data, "materials") else []
    }

    if obj.type == "MESH":
        info["vertices"] = len(obj.data.vertices)
        info["faces"] = len(obj.data.polygons)
        info["edges"] = len(obj.data.edges)

    return info
```

**Code Execution Handler:**
```python
def _handle_execute_code(self, params):
    """Execute arbitrary Python code in Blender."""
    code = params.get("code")

    # Capture stdout
    import io
    import sys

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    try:
        exec(code, {"bpy": bpy, "__builtins__": __builtins__})
        output = stdout_capture.getvalue()
        return {"output": output, "success": True}
    finally:
        sys.stdout = old_stdout
```

### 6.3 Material Node Graph Construction

**PBR Material Creation:**

```python
def create_material_from_textures(obj_name: str, texture_paths: dict):
    """
    Create Principled BSDF material with proper texture connections.

    texture_paths = {
        "diffuse": "/path/to/diffuse.png",
        "roughness": "/path/to/roughness.png",
        "metallic": "/path/to/metallic.png",
        "normal": "/path/to/normal.png",
        "ao": "/path/to/ao.png"
    }
    """
    import bpy

    obj = bpy.data.objects[obj_name]
    mat = bpy.data.materials.new(name=f"{obj_name}_PBR")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create output and BSDF
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Shared texture coordinate
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-800, 0)

    # Create texture nodes
    y_offset = 300
    for tex_type, filepath in texture_paths.items():
        if not filepath:
            continue

        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.location = (-400, y_offset)
        tex_node.image = bpy.data.images.load(filepath)

        # Set color space
        if tex_type in ["roughness", "metallic", "normal", "ao"]:
            tex_node.image.colorspace_settings.name = "Non-Color"

        # Connect to BSDF
        if tex_type == "diffuse":
            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        elif tex_type == "roughness":
            links.new(tex_node.outputs["Color"], bsdf.inputs["Roughness"])
        elif tex_type == "metallic":
            links.new(tex_node.outputs["Color"], bsdf.inputs["Metallic"])
        elif tex_type == "normal":
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.location = (-200, y_offset)
            links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        elif tex_type == "ao":
            # AO multiplied with diffuse
            mix = nodes.new("ShaderNodeMixRGB")
            mix.blend_type = 'MULTIPLY'
            mix.location = (-200, y_offset)
            links.new(tex_node.outputs["Color"], mix.inputs[2])

        y_offset -= 300

    # Assign to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat.name
```

### 6.4 Asset Import Pipeline

**GLTF Import:**
```python
def import_gltf_asset(filepath: str, name: str = None):
    """Import GLTF/GLB file into scene."""
    import bpy

    # Import GLTF
    bpy.ops.import_scene.gltf(filepath=filepath)

    # Get imported objects
    imported = [obj for obj in bpy.context.selected_objects]

    # Rename if specified
    if name and imported:
        imported[0].name = name

    # Center at origin
    for obj in imported:
        obj.location = (0, 0, 0)

    return [obj.name for obj in imported]
```

**FBX Export for Game Engine:**
```python
def export_for_game_engine(objects: list, filepath: str, format: str = "fbx"):
    """Export selected objects for game engine import."""
    import bpy

    # Select objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj_name in objects:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.select_set(True)

    if format == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=filepath,
            use_selection=True,
            apply_scale_options='FBX_SCALE_ALL',
            bake_space_transform=True,
            mesh_smooth_type='FACE'
        )
    elif format == "gltf":
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            use_selection=True,
            export_format='GLB'
        )

    return filepath
```

---

## Game Engine Integration

### 7.1 Asset Export Formats

| Format | Extension | Best For | Unity | UE5 |
|--------|-----------|----------|-------|-----|
| **FBX** | .fbx | General 3D assets | ✅ Native | ✅ Native |
| **GLTF** | .gltf/.glb | Web-ready, PBR | ✅ Plugin | ✅ Native |
| **OBJ** | .obj | Simple meshes | ✅ Native | ✅ Native |
| **USD** | .usd/.usda | Complex scenes | ✅ Plugin | ✅ Native |

**Recommended Export Settings:**

```python
# FBX Export for Unity
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    global_scale=1.0,
    apply_unit_scale=True,
    apply_scale_options='FBX_SCALE_ALL',
    bake_space_transform=True,
    object_types={'MESH', 'ARMATURE'},
    use_mesh_modifiers=True,
    mesh_smooth_type='FACE',
    use_tspace=True,
    primary_bone_axis='Y',
    secondary_bone_axis='X'
)

# FBX Export for Unreal Engine 5
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    global_scale=1.0,
    apply_unit_scale=True,
    apply_scale_options='FBX_SCALE_NONE',
    bake_space_transform=False,
    object_types={'MESH', 'ARMATURE'},
    use_mesh_modifiers=True,
    mesh_smooth_type='FACE',
    add_leaf_bones=False,
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    axis_forward='-Z',
    axis_up='Y'
)
```

### 7.2 Unity Import Workflow

```
Blender Export (FBX)
    ↓
Unity Assets Folder
    ↓
Unity Auto-Import
    ├── Mesh extraction
    ├── Material generation
    ├── Texture assignment
    └── Prefab creation (optional)
    ↓
Ready for Scene Placement
```

**Unity Import Script (C#):**
```csharp
using UnityEngine;
using UnityEditor;

public class BlenderAssetImporter : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        ModelImporter importer = assetImporter as ModelImporter;

        // Configure import settings
        importer.globalScale = 1.0f;
        importer.importBlendShapes = true;
        importer.importAnimation = true;
        importer.materialImportMode = ModelImporterMaterialImportMode.ImportViaMaterialDescription;
    }

    void OnPostprocessModel(GameObject g)
    {
        // Post-process imported model
        Debug.Log($"Imported Blender asset: {g.name}");
    }
}
```

### 7.3 Unreal Engine 5 Import Workflow

```
Blender Export (FBX)
    ↓
UE5 Content Browser (Drag & Drop)
    ↓
FBX Import Dialog
    ├── Skeletal Mesh / Static Mesh
    ├── Material slots
    ├── Collision generation
    └── LOD settings
    ↓
UMaterial / UStaticMesh Asset
```

**UE5 Python Import Script:**
```python
import unreal

def import_blender_asset(fbx_path: str, destination: str):
    """Import FBX asset from Blender into UE5."""

    # Create import task
    task = unreal.AssetImportTask()
    task.filename = fbx_path
    task.destination_path = destination
    task.replace_existing = True
    task.automated = True
    task.save = True

    # Configure FBX settings
    options = unreal.FbxImportUI()
    options.import_mesh = True
    options.import_textures = True
    options.import_materials = True
    options.import_as_skeletal = False  # Static mesh

    task.options = options

    # Execute import
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    return task.imported_object_paths
```

### 7.4 Texture & Material Conversion

**Blender → Unity Material Mapping:**

| Blender (Principled BSDF) | Unity (Standard/URP) |
|---------------------------|---------------------|
| Base Color | Albedo |
| Metallic | Metallic |
| Roughness | 1 - Smoothness |
| Normal | Normal Map |
| Emission | Emission |
| Alpha | Alpha |

**Blender → UE5 Material Mapping:**

| Blender (Principled BSDF) | UE5 (Material) |
|---------------------------|----------------|
| Base Color | Base Color |
| Metallic | Metallic |
| Roughness | Roughness |
| Normal | Normal |
| Emission | Emissive Color |
| Subsurface | Subsurface Color |

---

## Security Considerations

### 8.1 Code Execution Risks

**The `execute_blender_code` tool presents significant security risks:**

- Arbitrary Python execution
- File system access (read/write)
- Network operations
- System command execution

**Mitigation Strategies:**

```python
def validate_blender_code(code: str) -> bool:
    """
    Validate code before execution.
    Block dangerous operations.
    """
    blocked_patterns = [
        r"import\s+os",
        r"import\s+subprocess",
        r"import\s+socket",
        r"__import__",
        r"eval\(",
        r"exec\(",
        r"open\(",
        r"os\.system",
        r"os\.popen",
        r"subprocess\.",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, code):
            return False

    return True

def execute_sandboxed(code: str):
    """Execute code with restricted globals."""
    safe_globals = {
        "bpy": bpy,
        "mathutils": mathutils,
        "math": math,
        # Explicitly whitelist safe modules
    }

    exec(code, safe_globals, {})
```

### 8.2 API Key Management

**Environment Variables (Recommended):**
```bash
export BLENDER_HOST=localhost
export BLENDER_PORT=9876
export SKETCHFAB_API_TOKEN=your_token_here
export POLYHAVEN_API_KEY=your_key_here
export HYPER3D_API_KEY=your_key_here
export TENCENT_SECRET_ID=your_id_here
export TENCENT_SECRET_KEY=your_key_here
```

**Never Hardcode API Keys in:**
- Source code
- Version control
- Log files
- Error messages

### 8.3 File System Access

**Restrict Paths:**
```python
ALLOWED_PATHS = [
    "/tmp/blender_mcp/",
    os.path.expanduser("~/BlenderMCP/assets/"),
    bpy.path.abspath("//")  # Current blend file directory
]

def validate_path(filepath: str) -> bool:
    """Ensure path is within allowed directories."""
    abs_path = os.path.abspath(filepath)
    return any(abs_path.startswith(allowed) for allowed in ALLOWED_PATHS)
```

### 8.4 Network Security

**Socket Security:**
- Bind to localhost only (no external access)
- Use TLS for remote connections
- Implement authentication tokens
- Rate limit requests

```python
# Secure socket configuration
server_socket.bind(("127.0.0.1", 9876))  # Localhost only

# For remote access (not recommended for production)
if REMOTE_ACCESS_ENABLED:
    # Use TLS
    import ssl
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain("server.crt", "server.key")
    secure_socket = context.wrap_socket(server_socket, server_side=True)
```

---

## Performance Optimization

### 9.1 Asset Caching Strategy

```python
class AssetCache:
    """
    Cache downloaded assets to avoid re-downloading.
    """

    def __init__(self, cache_dir: str = "/tmp/blender_mcp_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.manifest = self._load_manifest()

    def get_cached_asset(self, asset_id: str, source: str) -> str:
        """Return cached asset path or None."""
        key = f"{source}_{asset_id}"
        if key in self.manifest:
            path = self.manifest[key]["path"]
            if os.path.exists(path):
                return path
        return None

    def cache_asset(self, asset_id: str, source: str, filepath: str):
        """Add asset to cache."""
        key = f"{source}_{asset_id}"
        self.manifest[key] = {
            "path": filepath,
            "timestamp": time.time(),
            "source": source
        }
        self._save_manifest()

    def clear_old_cache(self, max_age_days: int = 7):
        """Remove assets older than max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        for key, info in list(self.manifest.items()):
            if info["timestamp"] < cutoff:
                if os.path.exists(info["path"]):
                    os.remove(info["path"])
                del self.manifest[key]
        self._save_manifest()
```

### 9.2 Batch Operations

```python
def batch_create_objects(objects_config: list):
    """
    Create multiple objects in a single operation.
    More efficient than individual tool calls.
    """
    import bpy

    created = []
    for config in objects_config:
        obj_type = config.get("type", "cube")
        location = config.get("location", (0, 0, 0))
        name = config.get("name", f"Object_{len(created)}")

        if obj_type == "cube":
            bpy.ops.mesh.primitive_cube_add(location=location)
        elif obj_type == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(location=location)
        elif obj_type == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(location=location)

        obj = bpy.context.active_object
        obj.name = name
        created.append(obj.name)

    return created
```

### 9.3 Background Processing

```python
# Use threading for long-running operations
import threading

def download_asset_async(url: str, callback):
    """Download asset in background thread."""
    def download():
        response = requests.get(url, timeout=60)
        filepath = save_to_temp(response.content)
        # Schedule callback in main thread
        bpy.app.timers.register(lambda: callback(filepath), first_interval=0)

    thread = threading.Thread(target=download, daemon=True)
    thread.start()
```

### 9.4 Memory Management

```python
def cleanup_unused_data():
    """Remove orphaned data blocks to free memory."""
    import bpy

    # Remove unused meshes
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    # Remove unused materials
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    # Remove unused images
    for img in bpy.data.images:
        if img.users == 0:
            bpy.data.images.remove(img)

    # Pack remaining images for portability
    bpy.ops.file.pack_all()
```

---

## 3D Game AI Assistant Integration

### 10.1 RAG Context for Blender Commands

**Blender Python API Knowledge Base:**

```
Document Categories:
├── bpy.ops - Operators (mesh creation, modifiers, export)
├── bpy.data - Data access (objects, materials, images)
├── bpy.context - Context (active object, selected, scene)
├── bpy.types - Type definitions (Object, Mesh, Material)
├── Modeling Patterns - Common 3D modeling workflows
├── Material Nodes - Shader node graph recipes
└── Export Settings - Game engine specific configurations
```

**Example RAG Chunks:**

```json
{
  "id": "blender_create_cube",
  "content": "To create a cube in Blender: bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0)). This creates a 2-unit cube at the origin.",
  "metadata": {
    "category": "mesh_creation",
    "operator": "bpy.ops.mesh.primitive_cube_add",
    "parameters": ["size", "location", "rotation"]
  }
}
```

### 10.2 Natural Language → MCP Tool Mapping

```python
# Example LLM prompt for tool selection
TOOL_SELECTION_PROMPT = """
You are a 3D modeling assistant with access to Blender via MCP tools.

Available tools:
- get_scene_info: Get current scene state
- get_object_info(object_name): Get object details
- execute_blender_code(code): Run Python in Blender
- search_sketchfab_models(query): Search 3D model marketplace
- download_sketchfab_model(uid): Import model from Sketchfab
- search_polyhaven_assets(type, categories): Find textures/HDRIs
- generate_hyper3d_model_via_text(prompt): AI-generate 3D model

User request: "{user_request}"

1. Analyze what the user wants
2. Select appropriate tool(s)
3. Provide tool parameters
4. Return structured response
"""
```

### 10.3 Asset Request Workflow

```
User: "Create a low-poly medieval sword for my game"
    ↓
Claude AI Analysis:
    ├── Intent: Create 3D asset (sword)
    ├── Style: Low-poly, medieval
    └── Use case: Game asset
    ↓
Tool Selection Strategy:
    1. Try: search_sketchfab_models("low poly medieval sword")
    2. If not found: generate_hyper3d_model_via_text("low poly medieval sword")
    3. Fallback: execute_blender_code(procedural sword script)
    ↓
Asset Import & Setup:
    ├── Download/generate model
    ├── Import into Blender
    ├── Apply materials
    └── Export as FBX
    ↓
Response: "Created 'MedievalSword.fbx' - ready for import into Unity/UE5"
```

### 10.4 Iterative Refinement Loop

```python
async def iterative_asset_creation(request: str, max_iterations: int = 5):
    """
    Create asset with iterative refinement based on viewport feedback.
    """
    for i in range(max_iterations):
        # Step 1: Generate/modify asset
        if i == 0:
            await create_initial_asset(request)
        else:
            await apply_refinements(feedback)

        # Step 2: Capture viewport
        screenshot = await mcp.get_viewport_screenshot(max_size=800)

        # Step 3: Analyze with Claude Vision
        analysis = await claude.analyze_image(
            screenshot,
            prompt=f"Original request: {request}\nDoes this look correct? What needs improvement?"
        )

        # Step 4: Check if done
        if analysis.get("is_complete"):
            break

        feedback = analysis.get("improvements_needed")

    # Step 5: Export final asset
    return await export_asset()
```

---

## Installation & Configuration

### 11.1 Prerequisites

**Required:**
- Blender 3.6+ (3.0+ minimum, 3.6+ recommended)
- Python 3.10+
- `uv` package manager

**Installation:**
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone BlenderMCP repository
git clone https://github.com/ahujasid/blender-mcp.git
cd blender-mcp

# Install dependencies
uv pip install -e .
```

### 11.2 Blender Addon Setup

1. Open Blender
2. Edit → Preferences → Add-ons → Install
3. Select `addon.py` from blender-mcp repository
4. Enable "BlenderMCP" addon
5. Configure in sidebar panel (N key → BlenderMCP tab)

**Addon Configuration:**
```
BlenderMCP Settings:
├── Server Port: 9876
├── Auto-start Server: ✓
├── Enable Poly Haven: ✓
├── Enable Sketchfab: ✓
│   └── API Token: [your_token]
├── Enable Hyper3D: ✓
│   └── API Key: [your_key]
└── Enable Hunyuan3D: ✓
    └── Tencent Credentials: [configured]
```

### 11.3 MCP Server Configuration

**Claude Desktop (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "blender": {
      "command": "uvx",
      "args": ["blender-mcp"]
    }
  }
}
```

**Cursor IDE (`.cursor/mcp.json`):**
```json
{
  "mcpServers": {
    "blender": {
      "command": "uvx",
      "args": ["blender-mcp"]
    }
  }
}
```

**VS Code:**
Use built-in MCP installer or configure `settings.json`.

### 11.4 Environment Variables

```bash
# Required for remote connections
export BLENDER_HOST=localhost
export BLENDER_PORT=9876

# API Keys (optional, for asset integrations)
export SKETCHFAB_API_TOKEN=your_sketchfab_token
export POLYHAVEN_API_KEY=your_polyhaven_key
export HYPER3D_API_KEY=your_hyper3d_key
export TENCENT_SECRET_ID=your_tencent_id
export TENCENT_SECRET_KEY=your_tencent_key
```

---

## Error Handling & Resilience

### 12.1 Connection Recovery

```python
def robust_send_command(command: str, params: dict, max_retries: int = 3):
    """Send command with automatic retry on failure."""
    global blender_connection

    for attempt in range(max_retries):
        try:
            return send_command(command, params)
        except (socket.error, ConnectionResetError) as e:
            print(f"Connection error (attempt {attempt + 1}): {e}")
            blender_connection = None  # Force reconnection
            time.sleep(1 * (attempt + 1))  # Exponential backoff

    raise ConnectionError("Failed to connect to Blender after retries")
```

### 12.2 Timeout Management

```python
# Socket timeout configuration
SOCKET_TIMEOUT = 180  # 3 minutes for long operations

# Per-operation timeouts
OPERATION_TIMEOUTS = {
    "get_scene_info": 5,
    "get_object_info": 5,
    "execute_blender_code": 60,
    "download_sketchfab_model": 120,
    "generate_hyper3d_model_via_text": 300,
}

def send_with_timeout(command: str, params: dict):
    timeout = OPERATION_TIMEOUTS.get(command, 60)
    conn = get_blender_connection()
    conn.settimeout(timeout)
    return send_command(command, params)
```

### 12.3 Fallback Strategies

```python
ASSET_FALLBACK_CHAIN = [
    ("sketchfab", search_sketchfab),
    ("polyhaven", search_polyhaven),
    ("hyper3d", generate_hyper3d),
    ("hunyuan3d", generate_hunyuan3d),
    ("procedural", generate_procedural),
]

async def get_asset_with_fallback(request: str):
    """Try each asset source in order until success."""
    for source_name, source_func in ASSET_FALLBACK_CHAIN:
        try:
            result = await source_func(request)
            if result:
                return {"source": source_name, "asset": result}
        except Exception as e:
            print(f"{source_name} failed: {e}")
            continue

    raise AssetNotFoundError(f"Could not find or generate: {request}")
```

### 12.4 Logging & Monitoring

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("blender_mcp.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BlenderMCP")

# Log all tool calls
def log_tool_call(tool_name: str, params: dict, result: dict, duration: float):
    logger.info(f"Tool: {tool_name}")
    logger.info(f"Params: {json.dumps(params, indent=2)}")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Status: {result.get('status')}")
```

---

## Implementation Roadmap

### 13.1 Phase 1: Basic MCP Setup

**Duration:** 1 week

- [ ] Install Blender 3.6+
- [ ] Install BlenderMCP addon
- [ ] Configure MCP server
- [ ] Connect Claude Desktop
- [ ] Test basic commands (get_scene_info, execute_blender_code)
- [ ] Verify socket communication

**Milestone:** AI can query and modify Blender scene

### 13.2 Phase 2: Asset Integration

**Duration:** 1 week

- [ ] Configure Sketchfab API
- [ ] Configure Poly Haven integration
- [ ] Configure Hyper3D Rodin (optional)
- [ ] Test asset search and download
- [ ] Implement asset caching
- [ ] Test material assignment

**Milestone:** AI can import external assets into scene

### 13.3 Phase 3: Game Engine Bridge

**Duration:** 1 week

- [ ] Configure FBX export settings (Unity)
- [ ] Configure FBX export settings (UE5)
- [ ] Test GLTF export
- [ ] Create export automation scripts
- [ ] Test import in Unity
- [ ] Test import in Unreal Engine

**Milestone:** Assets flow from AI → Blender → Game Engine

### 13.4 Phase 4: Production Hardening

**Duration:** 1 week

- [ ] Implement code validation/sandboxing
- [ ] Set up error logging
- [ ] Configure monitoring
- [ ] Load test with concurrent requests
- [ ] Security audit
- [ ] Documentation and training

**Milestone:** Production-ready deployment

---

## API Reference

### 14.1 MCP Server Endpoints

| Tool | Parameters | Returns |
|------|------------|---------|
| `get_scene_info` | - | `{scene_name, object_count, objects[]}` |
| `get_object_info` | `object_name: str` | `{name, type, location, rotation, scale, materials[], vertices, faces}` |
| `get_viewport_screenshot` | `max_size: int = 800` | `{image: base64}` |
| `execute_blender_code` | `code: str` | `{output: str, success: bool}` |
| `search_sketchfab_models` | `query: str, categories?: str[], count?: int` | `{models[]}` |
| `download_sketchfab_model` | `uid: str` | `{imported_objects[]}` |
| `get_polyhaven_categories` | `asset_type: str` | `{categories[]}` |
| `search_polyhaven_assets` | `asset_type: str, categories?: str[]` | `{assets[]}` |
| `download_polyhaven_asset` | `asset_id: str, asset_type: str, resolution: str` | `{filepath: str}` |
| `generate_hyper3d_model_via_text` | `text_prompt: str, bbox_condition?: float[]` | `{subscription_key: str}` |
| `poll_rodin_job_status` | `subscription_key: str` | `{status: str, progress: float}` |
| `import_generated_asset` | `name: str, task_uuid: str` | `{imported_objects[]}` |

### 14.2 Blender Addon Commands

Internal command types handled by the addon socket server:

| Command | Parameters | Description |
|---------|------------|-------------|
| `get_scene_info` | - | Return scene metadata |
| `get_object_info` | `object_name` | Return object properties |
| `get_viewport_screenshot` | `filepath`, `max_size` | Capture viewport |
| `execute_code` | `code` | Run Python in Blender |
| `create_rodin_job` | `prompt`, `images` | Submit AI generation |
| `poll_rodin_job_status` | `key` | Check generation status |
| `import_generated_asset` | `name`, `uuid` | Import generated model |

### 14.3 Configuration Options

**Addon Preferences:**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `server_port` | int | 9876 | Socket server port |
| `auto_start` | bool | True | Start server on Blender launch |
| `enable_polyhaven` | bool | True | Enable Poly Haven integration |
| `enable_sketchfab` | bool | False | Enable Sketchfab (requires API key) |
| `sketchfab_api_token` | str | "" | Sketchfab API token |
| `enable_hyper3d` | bool | False | Enable Hyper3D Rodin |
| `hyper3d_mode` | enum | "hyper3d" | "hyper3d" or "fal" |
| `enable_hunyuan3d` | bool | False | Enable Hunyuan3D |
| `hunyuan3d_mode` | enum | "official" | "official" or "local" |

---

## Appendix

### 15.1 Supported Blender Versions

| Version | Support Level | Notes |
|---------|--------------|-------|
| 4.0+ | Full | Latest features |
| 3.6 LTS | Full | Recommended for production |
| 3.3 LTS | Partial | Core features work |
| 3.0-3.2 | Partial | May have compatibility issues |
| 2.x | Not supported | Use 3.0+ |

### 15.2 Asset Source Comparison

| Source | Speed | Quality | Cost | Best For |
|--------|-------|---------|------|----------|
| **Sketchfab** | Fast (10-30s) | High | Free tier + paid | Specific named assets |
| **Poly Haven** | Fast (5-20s) | Very High | Free | Textures, HDRIs, environments |
| **Hyper3D Rodin** | Slow (30-120s) | Variable | Credits-based | Custom unique assets |
| **Hunyuan3D** | Slow (30-120s) | Variable | Pay-per-use | Custom unique assets |
| **Python Script** | Instant (<5s) | Depends | Free | Procedural geometry |

### 15.3 Troubleshooting Guide

**Issue: "Connection refused" on port 9876**
```
Solution:
1. Ensure Blender is running
2. Check BlenderMCP addon is enabled
3. Verify server started (check Blender console)
4. Try restarting Blender
```

**Issue: "Module 'bpy' not found"**
```
Solution:
This error occurs when running MCP server outside Blender.
The MCP server connects TO Blender; it doesn't run inside it.
Check your configuration.
```

**Issue: "Sketchfab API unauthorized"**
```
Solution:
1. Verify API token is correct
2. Check token hasn't expired
3. Ensure token has download permissions
4. Try regenerating token on Sketchfab
```

**Issue: "Asset generation taking too long"**
```
Solution:
1. AI generation (Hyper3D/Hunyuan) can take 30-120s
2. Check job status with poll_*_job_status tools
3. Network issues may cause delays
4. Consider using cached assets
```

---

## References & Resources

- [BlenderMCP GitHub Repository](https://github.com/ahujasid/blender-mcp)
- [Anthropic MCP Documentation](https://docs.anthropic.com/en/docs/mcp)
- [Blender Python API](https://docs.blender.org/api/current/)
- [Sketchfab API Documentation](https://docs.sketchfab.com/data-api/v3)
- [Poly Haven API](https://polyhaven.com/api)

---

**Document Version:** 1.0 | December 4, 2025
**Status:** Production-Ready
**Next Review:** Q2 2026
