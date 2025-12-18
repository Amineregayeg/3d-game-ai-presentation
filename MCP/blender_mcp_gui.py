import streamlit as st
import anthropic
import subprocess
import json
import os
import pickle
import threading
import queue
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Blender MCP Assistant",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

class BlenderMCPClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.mcp_process = None
        self.tools = []
        self.output_queue = queue.Queue()
        self.reader_thread = None
        self.request_id = 1000
        
    def _read_output(self):
        """Background thread to read MCP output"""
        while self.mcp_process and self.mcp_process.poll() is None:
            try:
                line = self.mcp_process.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
            except Exception as e:
                break
    
    def _get_response(self, timeout=10):
        """Get response from queue with timeout"""
        try:
            response = self.output_queue.get(timeout=timeout)
            return json.loads(response)
        except queue.Empty:
            return None
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse response: {e}")
            return None
    
    def _send_request(self, request):
        """Send request to MCP server"""
        self.mcp_process.stdin.write(json.dumps(request) + '\n')
        self.mcp_process.stdin.flush()
        
    def start_blender_mcp(self):
        """Start the Blender MCP server"""
        try:
            self.mcp_process = subprocess.Popen(
                ['uvx', 'blender-mcp'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Start background reader thread
            self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self.reader_thread.start()
            
            # Give it a moment to start
            time.sleep(2)
            
            self.discover_tools()
            return True
        except Exception as e:
            st.error(f"Failed to start Blender MCP: {e}")
            return False
    
    def discover_tools(self):
        """Discover available tools from Blender MCP"""
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "streamlit-blender-client",
                    "version": "1.0.0"
                }
            }
        }
        self.request_id += 1
        
        try:
            self._send_request(init_request)
            init_response = self._get_response(timeout=5)
            
            if not init_response:
                st.error("No response from MCP initialization")
                return
            
            # Request tools list
            tools_request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/list"
            }
            self.request_id += 1
            
            self._send_request(tools_request)
            tools_response = self._get_response(timeout=5)
            
            if tools_response and 'result' in tools_response and 'tools' in tools_response['result']:
                self.tools = self.convert_mcp_tools_to_claude(tools_response['result']['tools'])
                st.success(f"✅ Loaded {len(self.tools)} Blender tools")
            else:
                st.warning("Could not load tools from MCP server")
        except Exception as e:
            st.error(f"Failed to discover tools: {e}")
    
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
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input
            }
        }
        self.request_id += 1
        
        try:
            if st.session_state.debug_mode:
                st.write(f"**Debug: Sending request:**")
                st.json(tool_request)
            
            self._send_request(tool_request)
            
            # Wait for response with longer timeout for Blender operations
            response = self._get_response(timeout=30)
            
            if st.session_state.debug_mode:
                st.write(f"**Debug: Received response:**")
                st.json(response if response else "TIMEOUT - No response")
            
            if not response:
                return f"Timeout: No response from tool '{tool_name}' within 30 seconds"
            
            if 'result' in response:
                content = response['result'].get('content', [])
                if content and len(content) > 0:
                    return content[0].get('text', str(response['result']))
                return str(response['result'])
            elif 'error' in response:
                return f"Error: {response['error']}"
            else:
                return f"Unexpected response format: {response}"
        except Exception as e:
            if st.session_state.debug_mode:
                st.error(f"**Debug: Exception occurred:** {e}")
            return f"Tool execution error: {e}"
    
    def chat(self, conversation_history: List[Dict], user_message: str, progress_callback=None) -> tuple:
        """Send a message to Claude and handle tool calls"""
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=self.tools,
            messages=conversation_history
        )
        
        tool_calls = []
        
        while response.stop_reason == "tool_use":
            tool_results = []
            assistant_content = []
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append(block)
                elif block.type == "tool_use":
                    if progress_callback:
                        progress_callback(f"🔧 Calling tool: {block.name}")
                    
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input
                    })
                    
                    result = self.call_mcp_tool(block.name, block.input)
                    
                    assistant_content.append(block)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            conversation_history.append({
                "role": "user",
                "content": tool_results
            })
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.tools,
                messages=conversation_history
            )
        
        final_response = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text
        
        conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return final_response, conversation_history, tool_calls
    
    def cleanup(self):
        """Clean up MCP server process"""
        if self.mcp_process:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
            except:
                self.mcp_process.kill()
            self.mcp_process = None

# Session state initialization
if 'client' not in st.session_state:
    st.session_state.client = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_conversation_file' not in st.session_state:
    st.session_state.current_conversation_file = None
if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = False
if 'batch_commands' not in st.session_state:
    st.session_state.batch_commands = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar
with st.sidebar:
    st.title("🎨 Blender MCP Assistant")
    st.markdown("---")
    
    # API Key input
    api_key = st.text_input(
        "Claude API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Enter your Anthropic API key or set ANTHROPIC_API_KEY environment variable"
    )
    
    # Connection status
    if st.session_state.client is None and api_key:
        if st.button("🔌 Connect to Blender MCP", use_container_width=True):
            with st.spinner("Connecting..."):
                client = BlenderMCPClient(api_key)
                if client.start_blender_mcp():
                    st.session_state.client = client
                    st.success("✅ Connected!")
                    st.rerun()
    elif st.session_state.client is not None:
        st.success("✅ Connected to Blender MCP")
        if st.button("🔌 Disconnect", use_container_width=True):
            st.session_state.client.cleanup()
            st.session_state.client = None
            st.rerun()
    
    st.markdown("---")
    
    # Conversation Management
    st.subheader("💾 Conversations")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 New Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.messages = []
            st.session_state.current_conversation_file = None
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            if st.session_state.messages:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = CONVERSATIONS_DIR / f"conversation_{timestamp}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump({
                        'messages': st.session_state.messages,
                        'history': st.session_state.conversation_history,
                        'timestamp': datetime.now()
                    }, f)
                st.session_state.current_conversation_file = filename
                st.success(f"Saved to {filename.name}")
    
    # Load conversation
    saved_conversations = sorted(CONVERSATIONS_DIR.glob("*.pkl"), reverse=True)
    if saved_conversations:
        selected_conv = st.selectbox(
            "Load Conversation",
            options=[""] + [f.name for f in saved_conversations],
            format_func=lambda x: "Select..." if x == "" else x
        )
        
        if selected_conv and selected_conv != "":
            conv_file = CONVERSATIONS_DIR / selected_conv
            if st.button("📂 Load Selected", use_container_width=True):
                with open(conv_file, 'rb') as f:
                    data = pickle.load(f)
                    st.session_state.messages = data['messages']
                    st.session_state.conversation_history = data['history']
                    st.session_state.current_conversation_file = conv_file
                    st.success(f"Loaded {selected_conv}")
                    st.rerun()
    
    st.markdown("---")
    
    # Batch Operations
    st.subheader("⚡ Batch Operations")
    st.session_state.batch_mode = st.toggle("Enable Batch Mode", value=st.session_state.batch_mode)
    
    if st.session_state.batch_mode:
        st.info("Enter multiple commands, one per line")
        batch_input = st.text_area(
            "Batch Commands",
            height=150,
            placeholder="Create a cube\nAdd a sphere\nApply red material to cube"
        )
        
        if st.button("▶️ Execute Batch", use_container_width=True):
            if batch_input and st.session_state.client:
                commands = [cmd.strip() for cmd in batch_input.split('\n') if cmd.strip()]
                st.session_state.batch_commands = commands
                st.rerun()
    
    st.markdown("---")
    
    # Debug Mode
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
    st.markdown("---")
    
    # Tools Info
    if st.session_state.client:
        with st.expander("🔧 Available Tools"):
            st.write(f"**{len(st.session_state.client.tools)} tools loaded**")
            for tool in st.session_state.client.tools:
                st.text(f"• {tool['name']}")

# Main content area
st.title("🎨 Blender MCP Assistant")

if not api_key:
    st.warning("⚠️ Please enter your Claude API key in the sidebar to get started.")
    st.info("You can get an API key from: https://console.anthropic.com/settings/keys")
elif st.session_state.client is None:
    st.info("👈 Click 'Connect to Blender MCP' in the sidebar to start.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "tool_calls" in message and message["tool_calls"]:
                with st.expander("🔧 Tool Calls"):
                    for tool_call in message["tool_calls"]:
                        st.json(tool_call)
    
    # Handle batch operations
    if st.session_state.batch_commands:
        commands = st.session_state.batch_commands
        st.session_state.batch_commands = []
        
        with st.chat_message("user"):
            st.markdown("**Batch Operation:**")
            for cmd in commands:
                st.markdown(f"- {cmd}")
        
        st.session_state.messages.append({
            "role": "user",
            "content": f"Batch Operation:\n" + "\n".join(f"- {cmd}" for cmd in commands)
        })
        
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            response_placeholder = st.empty()
            tool_calls_placeholder = st.empty()
            
            all_tool_calls = []
            all_responses = []
            
            for i, command in enumerate(commands, 1):
                progress_placeholder.info(f"Processing command {i}/{len(commands)}: {command}")
                
                response, st.session_state.conversation_history, tool_calls = st.session_state.client.chat(
                    st.session_state.conversation_history,
                    command,
                    lambda msg: progress_placeholder.info(msg)
                )
                
                all_responses.append(f"**{command}**\n{response}")
                all_tool_calls.extend(tool_calls)
            
            progress_placeholder.empty()
            
            final_response = "\n\n".join(all_responses)
            response_placeholder.markdown(final_response)
            
            if all_tool_calls:
                with tool_calls_placeholder.expander("🔧 Tool Calls"):
                    st.json(all_tool_calls)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "tool_calls": all_tool_calls
            })
    
    # Chat input
    if prompt := st.chat_input("Ask me to do something in Blender..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get assistant response
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            response_placeholder = st.empty()
            tool_calls_placeholder = st.empty()
            
            response, st.session_state.conversation_history, tool_calls = st.session_state.client.chat(
                st.session_state.conversation_history,
                prompt,
                lambda msg: progress_placeholder.info(msg)
            )
            
            progress_placeholder.empty()
            response_placeholder.markdown(response)
            
            if tool_calls:
                with tool_calls_placeholder.expander("🔧 Tool Calls"):
                    st.json(tool_calls)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": tool_calls
            })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit • Powered by Claude API • Connected to Blender MCP
    </div>
    """,
    unsafe_allow_html=True
)