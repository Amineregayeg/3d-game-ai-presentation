import anthropic 
import subprocess
import json
import sys
from typing import List, Dict, Any
import os

import os
api_key = os.environ.get("ANTHROPIC_API_KEY") or input("Enter API key: ")

class BlenderMCPClient:
    def __init__(self, api_key: str):
        """Initialize the client with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self.mcp_process = None
        self.tools = []
        
    def start_blender_mcp(self):
        """Start the Blender MCP server"""
        try:
            # Start the MCP server process
            self.mcp_process = subprocess.Popen(
                ['uvx', 'blender-mcp'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✓ Blender MCP server started")
            
            # Get available tools from MCP server
            self.discover_tools()
            
        except Exception as e:
            print(f"✗ Failed to start Blender MCP: {e}")
            sys.exit(1)
    
    def discover_tools(self):
        """Discover available tools from Blender MCP"""
        # Send initialization request to MCP
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "custom-claude-client",
                    "version": "1.0.0"
                }
            }
        }
        
        try:
            self.mcp_process.stdin.write(json.dumps(init_request) + '\n')
            self.mcp_process.stdin.flush()
            
            # Read response
            response = self.mcp_process.stdout.readline()
            print("✓ MCP initialized")
            
            # Request tools list
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            self.mcp_process.stdin.write(json.dumps(tools_request) + '\n')
            self.mcp_process.stdin.flush()
            
            tools_response = json.loads(self.mcp_process.stdout.readline())
            
            if 'result' in tools_response and 'tools' in tools_response['result']:
                # Convert MCP tools to Claude API format
                self.tools = self.convert_mcp_tools_to_claude(tools_response['result']['tools'])
                print(f"✓ Discovered {len(self.tools)} Blender tools")
            
        except Exception as e:
            print(f"✗ Failed to discover tools: {e}")
    
    def convert_mcp_tools_to_claude(self, mcp_tools: List[Dict]) -> List[Dict]:
        """Convert MCP tool format to Claude API tool format"""
        claude_tools = []
        
        for tool in mcp_tools:
            claude_tool = {
                "name": tool['name'],
                "description": tool.get('description', ''),
                "input_schema": tool.get('inputSchema', {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
            claude_tools.append(claude_tool)
        
        return claude_tools
    
    def call_mcp_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Call a tool on the Blender MCP server"""
        tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input
            }
        }
        
        try:
            self.mcp_process.stdin.write(json.dumps(tool_request) + '\n')
            self.mcp_process.stdin.flush()
            
            response = json.loads(self.mcp_process.stdout.readline())
            
            if 'result' in response:
                # Extract the content from the result
                content = response['result'].get('content', [])
                if content and len(content) > 0:
                    return content[0].get('text', str(response['result']))
                return str(response['result'])
            elif 'error' in response:
                return f"Error: {response['error']}"
            
        except Exception as e:
            return f"Tool execution error: {e}"
    
    def chat(self, user_message: str) -> str:
        """Send a message to Claude and handle tool calls"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        print(f"\n💬 You: {user_message}")
        
        # Send to Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=self.tools,
            messages=self.conversation_history
        )
        
        # Process response
        while response.stop_reason == "tool_use":
            # Extract tool calls
            tool_results = []
            assistant_content = []
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append(block)
                elif block.type == "tool_use":
                    print(f"🔧 Calling tool: {block.name}")
                    
                    # Execute tool on MCP
                    result = self.call_mcp_tool(block.name, block.input)
                    print(f"✓ Tool result: {result[:100]}...")
                    
                    assistant_content.append(block)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add assistant's tool use to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # Add tool results to history
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
            
            # Continue the conversation
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=self.conversation_history
            )
        
        # Extract final text response
        final_response = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        print(f"🤖 Claude: {final_response}")
        return final_response
    
    def cleanup(self):
        """Clean up MCP server process"""
        if self.mcp_process:
            self.mcp_process.terminate()
            print("\n✓ Blender MCP server stopped")


def main():
    # Get API key from user
    print("=== Blender MCP to Claude API Client ===\n")
    api_key = input("Enter your Claude API key: ").strip()
    
    if not api_key:
        print("Error: API key required")
        sys.exit(1)
    
    # Initialize client
    client = BlenderMCPClient(api_key)
    
    # Start Blender MCP
    client.start_blender_mcp()
    
    print("\n✓ Ready! Type your Blender commands or 'quit' to exit.\n")
    
    # Interactive loop
    try:
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            client.chat(user_input)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()