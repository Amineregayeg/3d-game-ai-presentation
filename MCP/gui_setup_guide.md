# Blender MCP GUI - Complete Setup Guide

## 🎨 Features

### ✨ Core Features
- **Beautiful Web Interface**: Clean, modern Streamlit UI
- **Real-time Chat**: Interactive conversation with Claude about Blender
- **Tool Call Visualization**: See exactly what operations are being performed
- **Live Status Updates**: Progress indicators for long-running operations

### 💾 Memory & Persistence
- **Save Conversations**: Save your chat history for later reference
- **Load Previous Chats**: Resume conversations from where you left off
- **Automatic Timestamps**: All conversations are timestamped
- **Conversation Browser**: Easy selection from saved conversations

### ⚡ Batch Operations
- **Multi-Command Execution**: Run multiple Blender commands in sequence
- **Progress Tracking**: See which command is currently executing
- **Batch Results**: View all results together
- **Tool Call History**: Track all operations performed in batch mode

## 📋 Installation

### 1. Install Dependencies

```bash
# Install Streamlit and Anthropic SDK
pip install -r requirements.txt

# Or install manually:
pip install anthropic streamlit
```

### 2. Ensure Blender MCP is Installed

```bash
# Test if blender-mcp works
uvx blender-mcp
```

### 3. Get Your Claude API Key

1. Go to https://console.anthropic.com/settings/keys
2. Create a new API key
3. Copy it (you'll need it in the app)

## 🚀 Running the Application

### Option 1: With Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
# Set API key
$env:ANTHROPIC_API_KEY = "your-api-key-here"

# Run the app
streamlit run blender_mcp_gui.py
```

**Windows Command Prompt:**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
streamlit run blender_mcp_gui.py
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
streamlit run blender_mcp_gui.py
```

### Option 2: Enter API Key in App

```bash
streamlit run blender_mcp_gui.py
```
Then enter your API key in the sidebar.

## 📖 Usage Guide

### Basic Usage

1. **Launch the App**
   ```bash
   streamlit run blender_mcp_gui.py
   ```

2. **Enter API Key** (if not set as environment variable)
   - Look in the sidebar
   - Paste your Claude API key
   - It will be masked for security

3. **Connect to Blender MCP**
   - Click "🔌 Connect to Blender MCP"
   - Wait for connection confirmation
   - You'll see "✅ Connected to Blender MCP"

4. **Start Chatting!**
   - Type commands in the chat input at the bottom
   - Press Enter to send
   - Watch Claude execute Blender operations

### Example Commands

```
Create a cube at position (0, 0, 0)

Add a UV sphere with radius 2

Apply a red material to the cube

Delete all objects in the scene

Get information about all objects

Take a screenshot of the viewport

Create a camera and position it at (5, -5, 5) looking at origin

Add a light source above the scene
```

### 💾 Conversation Management

#### Saving Conversations
1. Click "💾 Save Chat" in the sidebar
2. Conversation is saved with timestamp
3. Files are stored in `conversations/` folder

#### Loading Conversations
1. Use the dropdown to select a saved conversation
2. Click "📂 Load Selected"
3. Previous chat history will be restored

#### Starting New Chat
1. Click "📝 New Chat"
2. Current conversation will be cleared
3. You can still load it later if you saved it

### ⚡ Batch Operations

Batch mode lets you execute multiple Blender operations in sequence without waiting for each response.

#### How to Use Batch Mode:

1. **Enable Batch Mode**
   - Toggle "Enable Batch Mode" in the sidebar
   - A text area will appear

2. **Enter Commands**
   ```
   Create a cube
   Create a sphere at (2, 0, 0)
   Create a cylinder at (4, 0, 0)
   Apply different materials to each
   Arrange them in a row
   ```

3. **Execute Batch**
   - Click "▶️ Execute Batch"
   - Watch as each command is processed
   - All results appear together

#### Batch Mode Use Cases:

- **Scene Setup**: Create multiple objects at once
- **Material Application**: Apply materials to multiple objects
- **Animation Keyframes**: Set multiple keyframes in sequence
- **Scene Configuration**: Set up camera, lights, and objects together
- **Cleanup Operations**: Delete multiple objects in sequence

## 🔧 Advanced Features

### Tool Call Inspection

Every assistant response that uses tools includes an expandable "🔧 Tool Calls" section:
- See exactly which Blender operations were called
- View the parameters passed to each tool
- Useful for understanding what Claude did
- Great for debugging

### Available Tools Display

In the sidebar, expand "🔧 Available Tools" to see:
- Total number of Blender tools loaded
- List of all available tool names
- Helps you understand Claude's capabilities

### Conversation Persistence

Conversations are saved as Python pickle files containing:
- All messages (user and assistant)
- Full conversation history (for Claude API)
- Timestamp of when conversation was saved

**Storage Location**: `conversations/` folder in the same directory as the app

## 🎯 Tips & Best Practices

### Getting Better Results

1. **Be Specific**: "Create a red cube at (2, 0, 0)" is better than "make a cube"
2. **One Task at a Time**: Unless using batch mode, focus on one operation
3. **Use Names**: "Apply material to 'MyCube'" helps track objects
4. **Check Results**: Ask "What objects are in the scene?" to verify

### Using Batch Mode Effectively

1. **Plan Your Operations**: Write out your steps before executing
2. **Start Simple**: Test with 2-3 commands before larger batches
3. **Use Comments**: Describe what you're trying to achieve
4. **Check Progress**: The app shows which command is executing

### Saving and Organizing

1. **Save After Important Work**: Click "💾 Save Chat" after complex operations
2. **Descriptive Names**: Conversation files are timestamped automatically
3. **Regular Cleanup**: Delete old conversations from the `conversations/` folder
4. **Export Results**: Save important conversations before clearing

## 🐛 Troubleshooting

### "Failed to start Blender MCP"

**Solution:**
```bash
# Verify uvx is installed
pip install uv

# Test blender-mcp directly
uvx blender-mcp

# Check if Blender is installed and in PATH
```

### "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit anthropic
```

### API Key Issues

**Problems:**
- "Invalid API key"
- "Authentication failed"

**Solutions:**
1. Verify key at https://console.anthropic.com/settings/keys
2. Ensure you copied the entire key
3. Check you have API credits available
4. Try generating a new key

### Connection Drops

If the MCP connection drops:
1. Click "🔌 Disconnect"
2. Wait a moment
3. Click "🔌 Connect to Blender MCP" again

### Conversation Not Loading

If a saved conversation won't load:
- The file might be corrupted
- Try deleting it from `conversations/` folder
- Start a new conversation

## 🔒 Security Best Practices

1. **Never Share API Keys**: Keep your API key private
2. **Use Environment Variables**: Safer than hardcoding
3. **Rotate Keys**: Regularly generate new API keys
4. **Limit Key Permissions**: Use keys with minimal necessary permissions
5. **Monitor Usage**: Check your API usage regularly

## 📊 File Structure

```
your-project/
│
├── blender_mcp_gui.py          # Main application
├── requirements.txt             # Python dependencies
│
└── conversations/              # Saved conversations (auto-created)
    ├── conversation_20241216_143022.pkl
    ├── conversation_20241216_151530.pkl
    └── ...
```

## 🚀 Next Steps

### Customization Ideas

1. **Custom Themes**: Modify Streamlit theme in `.streamlit/config.toml`
2. **Export Formats**: Add export to JSON/CSV for conversations
3. **Templates**: Add preset command templates
4. **Keyboard Shortcuts**: Implement hotkeys for common actions
5. **Multi-Model Support**: Switch between Claude models

### Integration Ideas

1. **Blender Add-on**: Create a Blender add-on that launches the GUI
2. **Desktop App**: Use PyInstaller to create standalone executable
3. **Cloud Deployment**: Deploy to Streamlit Cloud for remote access
4. **Team Features**: Add user authentication and shared conversations

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Anthropic API Docs](https://docs.anthropic.com)
- [Blender Python API](https://docs.blender.org/api/current/)
- [MCP Protocol](https://modelcontextprotocol.io)

## 🎉 Enjoy!

You now have a powerful GUI for controlling Blender through Claude AI. Happy creating! 🎨
