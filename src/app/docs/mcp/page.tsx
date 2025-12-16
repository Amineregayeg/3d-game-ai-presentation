"use client";

import { DocPageLayout } from "@/components/docs/DocPageLayout";

const mcpIcon = (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
  </svg>
);

const phases = [
  {
    name: "Phase 1: Basic MCP Setup",
    duration: "1 Week",
    tasks: [
      "Install Blender 3.6+",
      "Install BlenderMCP addon",
      "Configure MCP server",
      "Connect Claude Desktop",
      "Test basic commands (get_scene_info, execute_blender_code)",
      "Verify socket communication",
    ],
    milestone: "AI can query and modify Blender scene",
  },
  {
    name: "Phase 2: Asset Integration",
    duration: "1 Week",
    tasks: [
      "Configure Sketchfab API",
      "Configure Poly Haven integration",
      "Configure Hyper3D Rodin (optional)",
      "Test asset search and download",
      "Implement asset caching",
      "Test material assignment",
    ],
    milestone: "AI can import external assets into scene",
  },
  {
    name: "Phase 3: Game Engine Bridge",
    duration: "1 Week",
    tasks: [
      "Configure FBX export settings (Unity)",
      "Configure FBX export settings (UE5)",
      "Test GLTF export",
      "Create export automation scripts",
      "Test import in Unity",
      "Test import in Unreal Engine",
    ],
    milestone: "Assets flow from AI to Blender to Game Engine",
  },
  {
    name: "Phase 4: Production Hardening",
    duration: "1 Week",
    tasks: [
      "Implement code validation/sandboxing",
      "Set up error logging",
      "Configure monitoring",
      "Load test with concurrent requests",
      "Security audit",
      "Documentation and training",
    ],
    milestone: "Production-ready deployment",
  },
];

const sections = [
  {
    title: "Overview",
    content: `The Blender MCP (Model Context Protocol) integration enables AI-driven 3D asset creation and manipulation. It connects Claude AI to Blender through a socket-based protocol, allowing natural language control of 3D workflows.

Key capabilities:
- Scene querying and modification via AI
- Asset search and import (Sketchfab, Poly Haven)
- AI-generated 3D models (Hyper3D Rodin)
- Automated game engine export (Unity, UE5)
- Python code execution in Blender`,
  },
  {
    title: "MCP Architecture",
    content: `The system uses a three-tier architecture:

1. Claude AI (Client):
   - Receives user requests
   - Generates MCP tool calls
   - Processes responses

2. MCP Server (Bridge):
   - Translates MCP protocol to Blender commands
   - Manages asset source APIs
   - Handles async operations

3. Blender Addon (Socket Server):
   - Listens on port 9876
   - Executes Python in Blender context
   - Returns scene data and screenshots

Communication Flow:
Claude -> MCP Server -> Socket -> Blender Addon -> bpy API`,
    code: `# MCP Server Configuration (claude_desktop_config.json)
{
  "mcpServers": {
    "blender": {
      "command": "uvx",
      "args": ["blender-mcp"]
    }
  }
}`,
  },
  {
    title: "MCP Tools Reference",
    content: `24 MCP tools for Blender interaction:

Scene Tools:
- get_scene_info: Query scene metadata
- get_object_info: Get object properties
- get_viewport_screenshot: Capture current view
- execute_blender_code: Run Python code

Asset Tools:
- search_sketchfab_models: Find 3D models
- download_sketchfab_model: Import model
- get_polyhaven_categories: List asset types
- search_polyhaven_assets: Find textures/HDRIs
- download_polyhaven_asset: Import asset

AI Generation:
- generate_hyper3d_model_via_text: Text-to-3D
- poll_rodin_job_status: Check generation
- import_generated_asset: Load generated model`,
  },
  {
    title: "Asset Sources",
    content: `Multiple asset sources for comprehensive coverage:

1. Sketchfab:
   - Speed: Fast (10-30s)
   - Quality: High
   - Best for: Named/specific assets
   - Requires: API token

2. Poly Haven:
   - Speed: Fast (5-20s)
   - Quality: Very High
   - Best for: Textures, HDRIs
   - Free, no authentication

3. Hyper3D Rodin:
   - Speed: Slow (30-120s)
   - Quality: Variable
   - Best for: Custom unique assets
   - Credits-based pricing

4. Python Scripts:
   - Speed: Instant (<5s)
   - Best for: Procedural geometry
   - No external dependencies`,
  },
  {
    title: "Game Engine Export",
    content: `Automated export to Unity and Unreal Engine:

FBX Export Settings:
- Scale: 1.0 (adjust for engine)
- Apply transforms
- Bake animations
- Include materials

Unity Specific:
- Y-up coordinate system
- Import settings preset
- Material remapping

Unreal Specific:
- Z-up coordinate system
- Nanite support (UE5)
- Auto LOD generation

GLTF Export:
- Web-compatible format
- Embedded textures option
- Draco compression`,
    code: `# Blender FBX Export
import bpy
bpy.ops.export_scene.fbx(
    filepath="asset.fbx",
    use_selection=True,
    apply_unit_scale=True,
    apply_scale_options='FBX_SCALE_ALL',
    bake_space_transform=True
)`,
  },
  {
    title: "Security Considerations",
    content: `Code execution safety measures:

1. Code Validation:
   - Whitelist allowed imports
   - Block file system access outside project
   - Prevent network operations in code

2. Sandboxing:
   - Blender runs in isolated process
   - Limited memory allocation
   - Timeout for long operations

3. API Security:
   - API tokens stored securely
   - Rate limiting per client
   - Audit logging enabled

Best Practices:
- Review generated code before execution
- Use read-only mode for demos
- Regular security audits`,
  },
];

const apiReference = [
  {
    endpoint: "get_scene_info",
    method: "MCP",
    description: "Get current Blender scene metadata",
    parameters: [],
  },
  {
    endpoint: "get_object_info",
    method: "MCP",
    description: "Get detailed information about a specific object",
    parameters: [
      { name: "object_name", type: "string", description: "Name of the Blender object" },
    ],
  },
  {
    endpoint: "get_viewport_screenshot",
    method: "MCP",
    description: "Capture the current viewport as an image",
    parameters: [
      { name: "max_size", type: "int", description: "Maximum dimension in pixels (default: 800)" },
    ],
  },
  {
    endpoint: "execute_blender_code",
    method: "MCP",
    description: "Execute Python code in Blender",
    parameters: [
      { name: "code", type: "string", description: "Python code to execute" },
    ],
  },
  {
    endpoint: "search_sketchfab_models",
    method: "MCP",
    description: "Search for 3D models on Sketchfab",
    parameters: [
      { name: "query", type: "string", description: "Search query" },
      { name: "categories", type: "array", description: "Filter by categories" },
      { name: "count", type: "int", description: "Number of results (default: 10)" },
    ],
  },
  {
    endpoint: "download_sketchfab_model",
    method: "MCP",
    description: "Download and import a Sketchfab model",
    parameters: [
      { name: "uid", type: "string", description: "Sketchfab model UID" },
    ],
  },
  {
    endpoint: "generate_hyper3d_model_via_text",
    method: "MCP",
    description: "Generate a 3D model from text description",
    parameters: [
      { name: "text_prompt", type: "string", description: "Description of the model to generate" },
      { name: "bbox_condition", type: "array", description: "Optional bounding box constraints" },
    ],
  },
  {
    endpoint: "poll_rodin_job_status",
    method: "MCP",
    description: "Check the status of a generation job",
    parameters: [
      { name: "subscription_key", type: "string", description: "Job subscription key" },
    ],
  },
  {
    endpoint: "import_generated_asset",
    method: "MCP",
    description: "Import a completed generated asset into Blender",
    parameters: [
      { name: "name", type: "string", description: "Name for the imported object" },
      { name: "task_uuid", type: "string", description: "Generation task UUID" },
    ],
  },
];

export default function MCPDocPage() {
  return (
    <DocPageLayout
      title="Blender MCP"
      subtitle="Model Context Protocol Integration"
      description="AI-driven 3D asset creation through Anthropic's Model Context Protocol. Connects Claude AI to Blender for natural language scene manipulation, asset import from Sketchfab/Poly Haven, AI generation via Hyper3D Rodin, and automated game engine export."
      gradient="from-orange-500 to-amber-600"
      accentColor="bg-orange-500"
      icon={mcpIcon}
      presentationLink="/mcp"
      technologies={["MCP", "Blender", "Socket", "Sketchfab", "Poly Haven", "Hyper3D Rodin", "FBX", "GLTF", "Unity", "UE5"]}
      phases={phases}
      sections={sections}
      apiReference={apiReference}
    />
  );
}
