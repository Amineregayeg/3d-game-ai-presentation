# Salesforce Virtual Assistant MVP
## Real-Time Voice Consultant with ElevenLabs + RAG + Salesforce MCP

**Version:** 1.0
**Created:** December 29, 2024
**Status:** AWAITING APPROVAL
**Base:** Duplicated from `/full_demo`

---

## 1. Executive Summary

### Vision
Create a **real-time virtual Salesforce consultant** that:
- **Takes initiative** - Proactively guides users through Salesforce tasks
- **Has expertise** - Accesses Salesforce best practices via RAG knowledge base
- **Has the right tone** - Professional, confident, supportive consultant persona
- **Executes actions** - Directly operates on client's Salesforce org via MCP
- **Visible results** - Client sees changes happen in real-time on their screen

### Core Differentiators from Full Demo

| Aspect | Full Demo (Blender) | Salesforce Assistant |
|--------|---------------------|---------------------|
| **Domain** | 3D Game Development | Salesforce CRM |
| **MCP** | Blender MCP | Salesforce MCP |
| **TTS** | ElevenLabs (one-shot) | ElevenLabs Conversational AI (real-time) |
| **RAG** | Blender docs | Salesforce best practices |
| **Persona** | Technical assistant | Proactive consultant |
| **Interaction** | Request-response | Conversational turn-taking |

---

## 2. Technical Architecture

### 2.1 High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SALESFORCE VIRTUAL ASSISTANT                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ELEVENLABS CONVERSATIONAL AI                  │   │
│  │    ┌──────────┐    ┌──────────┐    ┌──────────┐                 │   │
│  │    │ WebSocket│    │  Claude  │    │   RAG    │                 │   │
│  │    │ Real-time│◄──►│   LLM    │◄──►│ Knowledge│                 │   │
│  │    │  Audio   │    │          │    │   Base   │                 │   │
│  │    └──────────┘    └──────────┘    └──────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      SALESFORCE MCP                              │   │
│  │    ┌──────────┐    ┌──────────┐    ┌──────────┐                 │   │
│  │    │  SOQL    │    │  CRUD    │    │   Apex   │                 │   │
│  │    │ Queries  │    │ Records  │    │ Execution│                 │   │
│  │    └──────────┘    └──────────┘    └──────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CLIENT SALESFORCE ORG                         │   │
│  │              (Visible in split-screen or embed)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Mapping (Full Demo → Salesforce Assistant)

| Full Demo Component | Salesforce Assistant Component | Changes |
|---------------------|-------------------------------|---------|
| `VoiceInput.tsx` | `ConversationalInput.tsx` | WebSocket-based, continuous |
| `SettingsPanel.tsx` | `SalesforceSettings.tsx` | Org connection, persona |
| `RAGPipeline.tsx` | `RAGPipeline.tsx` | Same UI, Salesforce data |
| `AvatarPlayer.tsx` | `ConsultantAvatar.tsx` | Real-time lip-sync |
| `BlenderMCPPanel.tsx` | `SalesforceMCPPanel.tsx` | SOQL/DML commands |
| `ThreeJSViewport.tsx` | `SalesforceEmbed.tsx` | iFrame or screenshot |
| `GreetingPlayer.tsx` | `ConsultantGreeting.tsx` | Proactive intro |

---

## 3. ElevenLabs Conversational AI Integration

### 3.1 Why Conversational AI 2.0?

The existing `/full_demo` uses ElevenLabs TTS for one-shot responses. For a **real consultant experience**, we need:

- **Real-time conversation** - No waiting for full response generation
- **Natural turn-taking** - Interruptions, clarifications, "um/ah" handling
- **Integrated RAG** - Knowledge retrieval built into the conversation
- **Low latency** - ~75ms audio latency

### 3.2 Agent Configuration

```typescript
// ElevenLabs Agent Configuration
const salesforceConsultantAgent = {
  name: "Salesforce Consultant",
  conversation_config: {
    llm: {
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
      system_prompt: `You are an expert Salesforce consultant with 15+ years of experience.
Your role is to:
1. PROACTIVELY guide clients through Salesforce tasks
2. Take initiative - don't wait for detailed instructions
3. Explain what you're doing and why
4. Execute actions directly via Salesforce MCP
5. Verify results and suggest next steps

Tone: Professional, confident, supportive. Like a senior consultant taking charge.

When the user describes a need:
- Immediately propose a solution
- Ask for confirmation: "Shall I do this now?"
- Execute upon approval
- Show results and explain what changed`
    },
    tts: {
      voice_id: "21m00Tcm4TlvDq8ikWAM", // Professional voice
      model_id: "eleven_turbo_v2_5",
      stability: 0.5,
      similarity_boost: 0.75
    },
    stt: {
      model: "eleven_turbo_v2",
      language: "en"
    },
    turn_taking: {
      mode: "natural",
      silence_threshold_ms: 700,
      allow_interruptions: true
    }
  },
  // RAG Integration
  knowledge_base: {
    enabled: true,
    sources: ["salesforce_best_practices", "admin_guide", "apex_reference"]
  },
  // Tool Calling (MCP Bridge)
  tools: [
    {
      type: "mcp",
      server: "salesforce-mcp",
      description: "Execute Salesforce operations"
    }
  ]
};
```

### 3.3 WebSocket Connection Flow

```typescript
// Browser-side WebSocket connection
class ConversationalClient {
  private ws: WebSocket;
  private audioContext: AudioContext;
  private mediaRecorder: MediaRecorder;

  async connect(agentId: string, signedUrl: string) {
    this.ws = new WebSocket(signedUrl);

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case "audio":
          this.playAudio(message.audio_base64);
          break;
        case "transcript":
          this.onTranscript(message.text, message.is_final);
          break;
        case "tool_call":
          this.onToolCall(message.tool, message.input);
          break;
        case "tool_result":
          this.onToolResult(message.result);
          break;
      }
    };
  }

  async startConversation() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mediaRecorder = new MediaRecorder(stream);

    this.mediaRecorder.ondataavailable = (event) => {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(event.data); // Stream audio chunks
      }
    };

    this.mediaRecorder.start(100); // 100ms chunks
  }
}
```

---

## 4. Salesforce MCP Integration

### 4.1 Selected MCP Server

**Package:** `@tsmztech/mcp-server-salesforce`

**Rationale:**
- 15 comprehensive tools (SOQL, CRUD, Apex, metadata)
- OAuth 2.0 support for secure client connections
- Active maintenance
- Claude-optimized

### 4.2 Available Tools

| Tool | Purpose | Example Use |
|------|---------|-------------|
| `salesforce_query_records` | SOQL queries | "Show me all accounts created this month" |
| `salesforce_dml_records` | Create/Update/Delete | "Create a new contact for John Smith" |
| `salesforce_describe_object` | Object metadata | "What fields are on the Opportunity object?" |
| `salesforce_search_all` | SOSL search | "Find anything mentioning 'renewal'" |
| `salesforce_execute_anonymous` | Run Apex | "Run a batch job to update all..." |
| `salesforce_manage_field` | Create fields | "Add a custom field for..." |

### 4.3 Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │     │   Backend   │     │  Salesforce │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │ 1. Start OAuth    │                   │
       │──────────────────>│                   │
       │                   │                   │
       │ 2. Redirect URL   │                   │
       │<──────────────────│                   │
       │                   │                   │
       │ 3. User authorizes│                   │
       │───────────────────────────────────────>│
       │                   │                   │
       │ 4. Callback + code│                   │
       │<──────────────────────────────────────│
       │                   │                   │
       │ 5. Exchange code  │                   │
       │──────────────────>│                   │
       │                   │ 6. Get tokens     │
       │                   │──────────────────>│
       │                   │<──────────────────│
       │                   │                   │
       │ 7. Session ready  │                   │
       │<──────────────────│                   │
       │                   │                   │
```

### 4.4 MCP Panel Component

```tsx
// SalesforceMCPPanel.tsx
interface SalesforceMCPPanelProps {
  isConnected: boolean;
  orgInfo: {
    name: string;
    instanceUrl: string;
    username: string;
  } | null;
  recentOperations: SalesforceOperation[];
  isExecuting: boolean;
}

interface SalesforceOperation {
  id: string;
  type: 'query' | 'insert' | 'update' | 'delete' | 'apex';
  description: string;
  soql?: string;
  recordCount?: number;
  status: 'pending' | 'running' | 'success' | 'error';
  result?: unknown;
  timestamp: number;
  duration_ms?: number;
}
```

---

## 5. RAG Knowledge Base

### 5.1 Salesforce Knowledge Sources

| Source | Content | Priority |
|--------|---------|----------|
| **Salesforce Help Docs** | Official documentation | High |
| **Trailhead Modules** | Best practices, how-tos | High |
| **Apex Developer Guide** | Code reference | Medium |
| **Lightning Component Guide** | UI development | Medium |
| **Admin Certification Guide** | Common configurations | High |
| **Release Notes** | Recent features | Medium |
| **Community Solutions** | Stack Exchange, forums | Low |

### 5.2 Document Schema

```typescript
interface SalesforceKnowledgeDocument {
  id: string;
  title: string;
  content: string;
  source: 'help_docs' | 'trailhead' | 'apex_guide' | 'admin_guide' | 'community';
  category: 'configuration' | 'development' | 'automation' | 'security' | 'reporting' | 'integration';
  salesforce_version: string; // e.g., "Winter '25"
  object_types?: string[]; // Related objects (Account, Contact, etc.)
  features?: string[]; // Related features (Flow, Apex, Lightning)
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  last_updated: string;
  url?: string;
}
```

### 5.3 RAG Configuration

```python
# Salesforce RAG Pipeline Configuration
RAG_CONFIG = {
    "embedding_model": "BAAI/bge-m3",  # Same as full_demo
    "embedding_dims": 4096,

    "retrieval": {
        "mode": "hybrid",  # dense + sparse
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "top_k_initial": 50,
        "top_k_final": 10
    },

    "reranker": {
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "threshold": 0.5
    },

    "context_assembly": {
        "max_tokens": 4000,  # Larger for Salesforce complexity
        "include_metadata": True,
        "prioritize_recent_version": True
    },

    "filters": {
        "enabled": True,
        "fields": ["category", "salesforce_version", "object_types"]
    }
}
```

---

## 6. Consultant Persona & Behavior

### 6.1 Persona Definition

```
Name: "Alex" (gender-neutral)
Role: Senior Salesforce Consultant
Experience: 15+ years, 10x certified
Specialization: Sales Cloud, Service Cloud, CPQ, Integration

Personality Traits:
- PROACTIVE: "I see you're trying to... Let me help with that"
- CONFIDENT: "Here's the best approach for this..."
- SUPPORTIVE: "Great question. This is a common challenge..."
- EFFICIENT: "I'll do this in 3 steps. First..."
- EDUCATIONAL: "This works because... Let me explain"

Communication Style:
- Uses "we" and "let's" (collaborative)
- Asks for confirmation before major actions
- Provides context without over-explaining
- Celebrates small wins ("Perfect, that's done")
```

### 6.2 Proactive Behaviors

| Trigger | Consultant Response |
|---------|---------------------|
| User says "I need to..." | "Got it. Here's my plan: [steps]. Shall I proceed?" |
| Vague request | "To clarify - are you looking for [A] or [B]?" |
| After completing task | "Done! Next, you might want to consider [suggestion]" |
| User hesitates | "No worries, take your time. I can also show you..." |
| Error occurs | "That didn't work because [reason]. Let me try [alternative]" |
| Long silence | "Still there? I'm ready when you are" |

### 6.3 Example Conversation Flow

```
USER: "I need to track when opportunities are overdue"

CONSULTANT: "Perfect, I can help with that. The best approach is to create
a formula field on Opportunity that calculates if the Close Date has passed.
I'll also add a List View so you can see all overdue deals at a glance.
Sound good?"

USER: "Yes please"

CONSULTANT: "Great, I'm on it. First, I'm adding a custom checkbox field
called 'Is Overdue' to your Opportunity object... Done. Now creating the
formula: it checks if Close Date is before today AND the stage isn't Closed Won.
Deploying now... Perfect. Let me create that list view too... And done!
You now have an 'Overdue Opportunities' list view. Would you like me to
add this to your Sales app navigation?"

USER: "That would be great"

CONSULTANT: "Added. You'll see it in your navigation bar now. One more thing -
want me to set up an automated email alert when opportunities become overdue?
That way your reps get notified automatically."
```

---

## 7. UI/UX Design

### 7.1 Page Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◀ Back   🏢 SALESFORCE VIRTUAL CONSULTANT                   ⚙️ Settings │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────┐  ┌──────────────────────────────────────┐│
│  │      CONSULTANT AVATAR    │  │        SALESFORCE ORG VIEW           ││
│  │    ┌─────────────────┐    │  │                                      ││
│  │    │                 │    │  │   ┌────────────────────────────────┐ ││
│  │    │    [Avatar]     │    │  │   │  Connected: Acme Corp          │ ││
│  │    │    Speaking...  │    │  │   │  User: admin@acme.com          │ ││
│  │    │                 │    │  │   └────────────────────────────────┘ ││
│  │    └─────────────────┘    │  │                                      ││
│  │                           │  │   [Salesforce Lightning iFrame]      ││
│  │   🎙️ Listening... 🔴      │  │   or                                 ││
│  │                           │  │   [Live Screenshot Preview]          ││
│  │   "I'm adding a formula   │  │                                      ││
│  │    field to Opportunity..." │  │                                    ││
│  └───────────────────────────┘  └──────────────────────────────────────┘│
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────┐  ┌────────────────────────────────┐  │
│  │    CONVERSATION TRANSCRIPT    │  │    SALESFORCE OPERATIONS       │  │
│  │                               │  │                                │  │
│  │  👤 "I need to track when    │  │  ✅ describe_object: Opportunity│  │
│  │      opportunities are        │  │     Retrieved 47 fields        │  │
│  │      overdue"                 │  │                                │  │
│  │                               │  │  ✅ manage_field: Is_Overdue   │  │
│  │  🤖 "Perfect, I can help     │  │     Created checkbox field     │  │
│  │      with that..."           │  │                                │  │
│  │                               │  │  🔄 query_records: Opportunity │  │
│  │  👤 "Yes please"             │  │     Executing SOQL...          │  │
│  │                               │  │                                │  │
│  └───────────────────────────────┘  └────────────────────────────────┘  │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                         RAG KNOWLEDGE CONTEXT                            │
│  📚 Sources: Admin Guide (3) | Apex Reference (1) | Trailhead (2)       │
│  🎯 Relevance: 0.94 | ⏱️ Latency: 340ms                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Key UI Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `ConsultantAvatar` | Visual presence | Real-time lip-sync, expressions |
| `ConversationPanel` | Transcript | Turn-by-turn, highlights |
| `SalesforceEmbed` | Org visibility | iFrame or live screenshots |
| `OperationsLog` | MCP actions | SOQL syntax highlight, status |
| `RAGContext` | Knowledge source | Collapsible, citations |
| `ConnectionStatus` | Org connection | OAuth status, org info |

### 7.3 Color Scheme

```css
/* Salesforce-inspired professional theme */
--primary: #0176D3;        /* Salesforce Blue */
--secondary: #032D60;      /* Navy */
--accent: #04844B;         /* Success Green */
--warning: #FE9339;        /* Warning Orange */
--background: #F3F3F3;     /* Lightning Gray */
--card: #FFFFFF;
--text: #181818;
--muted: #706E6B;

/* Gradient for consultant persona */
--gradient-consultant: linear-gradient(135deg, #0176D3, #032D60);
```

---

## 8. Implementation Phases

### Phase 1: Foundation (Days 1-2)
- [ ] Duplicate `/full_demo` → `/salesforce_demo`
- [ ] Create new component folder `src/components/salesforce-demo/`
- [ ] Adapt color scheme and branding
- [ ] Create `SalesforceSettings.tsx` with org connection
- [ ] Set up placeholder RAG with mock Salesforce data

### Phase 2: Salesforce MCP Integration (Days 3-4)
- [ ] Install `@tsmztech/mcp-server-salesforce`
- [ ] Create backend MCP bridge (`/api/salesforce/mcp`)
- [ ] Implement OAuth flow for client orgs
- [ ] Create `SalesforceMCPPanel.tsx` component
- [ ] Test SOQL queries and record operations

### Phase 3: ElevenLabs Conversational AI (Days 5-6)
- [ ] Replace one-shot TTS with Conversational AI agent
- [ ] Implement WebSocket connection in frontend
- [ ] Create real-time audio streaming
- [ ] Integrate turn-taking and interruption handling
- [ ] Connect RAG as knowledge base

### Phase 4: RAG Knowledge Base (Days 7-8)
- [ ] Collect Salesforce documentation corpus
- [ ] Generate embeddings with BGE-M3
- [ ] Store in PostgreSQL + pgvector
- [ ] Implement Salesforce-specific filters
- [ ] Test retrieval quality

### Phase 5: Consultant Persona (Day 9)
- [ ] Craft detailed system prompt
- [ ] Implement proactive behaviors
- [ ] Add conversation memory (last 10 turns)
- [ ] Test persona consistency
- [ ] Fine-tune voice settings

### Phase 6: Polish & Testing (Days 10-11)
- [ ] Salesforce org embed/screenshot
- [ ] Error handling and fallbacks
- [ ] Mobile responsive layout
- [ ] End-to-end user testing
- [ ] Performance optimization

### Phase 7: Demo Preparation (Day 12)
- [ ] Create demo Salesforce org
- [ ] Prepare demo scenarios
- [ ] Record backup video
- [ ] Deploy to VPS

---

## 9. File Structure

```
/src/app/salesforce_demo/
├── page.tsx                    # Main page component
├── layout.tsx                  # Metadata
└── loading.tsx                 # Loading skeleton

/src/components/salesforce-demo/
├── ConsultantAvatar.tsx        # Real-time avatar
├── ConversationalInput.tsx     # WebSocket audio
├── ConversationPanel.tsx       # Transcript display
├── SalesforceSettings.tsx      # Org connection
├── SalesforceMCPPanel.tsx      # Operations log
├── SalesforceEmbed.tsx         # Org preview
├── RAGContext.tsx              # Knowledge sources
├── ConsultantGreeting.tsx      # Proactive intro
└── index.ts                    # Barrel export

/src/app/api/salesforce/
├── connect/route.ts            # OAuth initiate
├── callback/route.ts           # OAuth callback
├── mcp/route.ts                # MCP bridge
├── status/route.ts             # Connection status
└── screenshot/route.ts         # Org screenshot

/backend/
├── salesforce_mcp_bridge.py    # MCP server wrapper
├── salesforce_rag_service.py   # RAG for Salesforce
└── elevenlabs_agent.py         # Conversational AI agent

/data/salesforce_knowledge/
├── help_docs/                  # Salesforce Help
├── trailhead/                  # Trailhead modules
├── apex_guide/                 # Developer docs
└── embeddings/                 # Pre-computed vectors
```

---

## 10. API Endpoints

### 10.1 New Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/salesforce/connect` | POST | Start OAuth flow |
| `/api/salesforce/callback` | GET | OAuth callback |
| `/api/salesforce/mcp` | POST | Execute MCP operations |
| `/api/salesforce/status` | GET | Connection status |
| `/api/salesforce/disconnect` | POST | Logout from org |
| `/api/elevenlabs/agent` | GET | Get agent config |
| `/api/elevenlabs/conversation` | POST | Start conversation |

### 10.2 WebSocket Endpoint

```typescript
// /api/conversation/ws
// Real-time bidirectional audio/text streaming
{
  // Client → Server
  "audio_chunk": "base64...",  // 100ms audio chunks
  "interrupt": true,            // User interruption

  // Server → Client
  "audio": "base64...",         // Assistant audio
  "transcript": "text...",      // User transcript
  "assistant_text": "text...",  // Assistant text
  "tool_call": { tool, input }, // MCP operation
  "tool_result": { result }     // Operation result
}
```

---

## 11. Dependencies

### 11.1 New NPM Packages

```json
{
  "dependencies": {
    "@tsmztech/mcp-server-salesforce": "^1.0.0",
    "@eleven-labs/client": "^1.0.0"
  }
}
```

### 11.2 New Python Packages

```
elevenlabs>=1.0.0
jsforce>=0.1.0  # If needed for direct Salesforce calls
```

### 11.3 External Services

| Service | Purpose | Estimated Cost |
|---------|---------|----------------|
| ElevenLabs Conversational AI | Voice agent | ~$0.10/min conversation |
| Salesforce Developer Org | Demo org | Free |
| PostgreSQL + pgvector | RAG storage | Existing |

---

## 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Conversation latency | <500ms | Audio round-trip time |
| Operation success rate | >95% | MCP operations completed |
| RAG relevance | >0.85 | RAGAS faithfulness score |
| User satisfaction | >4.5/5 | Demo feedback |
| Persona consistency | 100% | Proactive behavior triggers |

---

## 13. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ElevenLabs rate limits | High | Pre-purchase credits, queue requests |
| Salesforce OAuth complexity | Medium | Use Salesforce CLI for demo |
| RAG knowledge gaps | Medium | Start with admin-focused content |
| Real-time audio issues | Medium | Fallback to standard TTS |
| Client org data privacy | High | Demo org only, clear disclaimers |

---

## 14. Demo Scenarios

### Scenario 1: Create Custom Field
"I need a way to track customer satisfaction on Account records"

### Scenario 2: Query and Update
"Show me all Opportunities over $100K and mark them as priority"

### Scenario 3: Automation Setup
"When a Case is closed, automatically send a survey email"

### Scenario 4: Troubleshooting
"My users can't see the new field I created"

### Scenario 5: Best Practice Advice
"What's the best way to structure our Lead assignment rules?"

---

## 15. Sources & References

- [ElevenLabs Conversational AI](https://elevenlabs.io/conversational-ai)
- [ElevenLabs Agents Platform](https://elevenlabs.io/docs/agents-platform/overview)
- [Salesforce MCP Blog](https://developer.salesforce.com/blogs/2025/06/introducing-mcp-support-across-salesforce)
- [tsmztech/mcp-server-salesforce](https://github.com/tsmztech/mcp-server-salesforce)
- [Anthropic-Salesforce Partnership](https://www.salesforce.com/news/press-releases/2025/10/14/anthropic-regulated-industries-partnership-expansion-announcement/)

---

## 16. Approval Checklist

Please confirm before implementation:

- [ ] Overall architecture approved
- [ ] ElevenLabs Conversational AI approach approved
- [ ] Salesforce MCP server selection approved
- [ ] Consultant persona definition approved
- [ ] UI/UX layout approved
- [ ] Implementation timeline approved (12 days)
- [ ] Demo scenarios approved
- [ ] Budget for ElevenLabs credits confirmed

---

**STATUS: AWAITING APPROVAL**

---

*Document created: December 29, 2024*
*Based on: `/full_demo` architecture*
