# RAG Demo Implementation Plan
## `/rag_demo` - Interactive Advanced RAG System Demo

**Version:** 1.0
**Last Updated:** December 11, 2025
**Status:** Awaiting Approval

---

## 1. Overview

Create a top-tier interactive demo page at `/rag_demo` that showcases all features of the Advanced RAG system documented in `RAG_ARCHITECTURE_PLAN.md`. The demo will follow the same design patterns as `/demo` and `/avatar_demo` with shadcn/ui components and a polished, professional interface.

### 1.1 Key Features to Demonstrate

| Feature | Description | Interactive Element |
|---------|-------------|---------------------|
| **Query Input** | Natural language query with session memory | Text input + microphone |
| **Query Analysis** | Real-time intent extraction and entity recognition | Visual breakdown panel |
| **Hybrid Retrieval** | Dense (BGE-M3) + Sparse (BM25) search | Toggle between modes |
| **RRF Fusion** | Reciprocal Rank Fusion visualization | Score breakdown |
| **Cross-Encoder Reranking** | MiniLM reranking display | Before/after comparison |
| **Context Assembly** | Retrieved documents display | Expandable cards |
| **Validation Loop** | Hallucination detection, grounding checks | Status indicators |
| **RAGAS Metrics** | Real-time quality scores | Dashboard gauges |
| **Session Memory** | Conversation history | Chat-like interface |

---

## 2. Page Architecture

### 2.1 Layout Structure

```
┌────────────────────────────────────────────────────────────────┐
│                         HEADER                                  │
│  RAG Demo  |  Status Badges (System Health)  |  Settings       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌────────────────────────────────┐  │
│  │                      │  │                                │  │
│  │   QUERY INPUT        │  │   RAG PIPELINE VISUALIZATION   │  │
│  │   - Text input       │  │   - Query Analysis             │  │
│  │   - Voice (optional) │  │   - Retrieval Flow             │  │
│  │   - Session context  │  │   - Reranking                  │  │
│  │                      │  │   - Context Assembly           │  │
│  ├──────────────────────┤  │                                │  │
│  │                      │  │                                │  │
│  │   SETTINGS PANEL     │  │                                │  │
│  │   - Retrieval mode   │  │                                │  │
│  │   - Top-K            │  │                                │  │
│  │   - Filters          │  │                                │  │
│  │                      │  │                                │  │
│  └──────────────────────┘  └────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   RETRIEVED CONTEXT                       │  │
│  │   [Card 1] [Card 2] [Card 3] ... [Card N]                │  │
│  │   - Relevance score, source, preview                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────┐  ┌───────────────────────────┐   │
│  │   GENERATED RESPONSE    │  │   METRICS DASHBOARD       │   │
│  │   - Answer text         │  │   - Faithfulness          │   │
│  │   - Citations           │  │   - Relevancy             │   │
│  │   - Validation status   │  │   - Precision             │   │
│  │                         │  │   - Latency breakdown     │   │
│  └─────────────────────────┘  └───────────────────────────┘   │
│                                                                 │
├────────────────────────────────────────────────────────────────┤
│                     NAVIGATION DOCK                             │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Color Scheme

Following RAG slides pattern (emerald/cyan theme):

```typescript
const RAG_COLORS = {
  primary: 'emerald',      // Main accent
  secondary: 'cyan',       // Secondary elements
  tertiary: 'purple',      // Highlights
  success: 'green',        // Valid/pass
  warning: 'amber',        // Medium confidence
  error: 'red',           // Failed/hallucination
  background: {
    dark: 'slate-950',
    card: 'slate-900/80',
    hover: 'slate-800/50'
  }
};
```

---

## 3. Component Structure

### 3.1 File Organization

```
src/
├── app/
│   ├── rag_demo/
│   │   └── page.tsx                 # Main demo page
│   └── api/
│       └── rag/
│           ├── query/
│           │   └── route.ts         # Query endpoint
│           ├── retrieve/
│           │   └── route.ts         # Retrieval endpoint
│           └── status/
│               └── route.ts         # System status
├── components/
│   └── rag-demo/
│       ├── QueryInput.tsx           # Text/voice input
│       ├── QueryAnalysis.tsx        # Intent/entity display
│       ├── RetrievalFlow.tsx        # Pipeline visualization
│       ├── DocumentCards.tsx        # Retrieved docs
│       ├── ResponsePanel.tsx        # Generated answer
│       ├── MetricsDashboard.tsx     # RAGAS metrics
│       ├── SettingsPanel.tsx        # Configuration
│       ├── SessionHistory.tsx       # Conversation memory
│       └── index.ts                 # Exports
└── lib/
    └── rag/
        ├── types.ts                 # TypeScript types
        └── mock-data.ts             # Demo data
```

### 3.2 Component Breakdown

#### 3.2.1 QueryInput Component

```tsx
// Features:
// - Large text input with placeholder suggestions
// - Optional voice input button (leverages VoxFormer)
// - Session context indicator
// - Submit button with loading state
// - Sample query buttons

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  sessionId: string | null;
  recentQueries: string[];
}
```

#### 3.2.2 QueryAnalysis Component

```tsx
// Displays LLM-extracted:
// - Intent (e.g., "select_all_faces")
// - Entities (e.g., ["faces", "Blender"])
// - Query variations generated
// - Detected filters (version, category)

interface QueryAnalysisProps {
  analysis: {
    intent: string;
    entities: string[];
    variations: string[];
    filters: Record<string, string>;
    confidence: number;
  };
  isLoading: boolean;
}
```

#### 3.2.3 RetrievalFlow Component (Key Visualization)

```tsx
// Animated pipeline visualization showing:
// - Dense search (BGE-M3) - results count + latency
// - Sparse search (BM25) - results count + latency
// - RRF Fusion - merged results
// - Reranking - final ordered list

interface RetrievalFlowProps {
  stages: {
    name: string;
    status: 'pending' | 'running' | 'complete';
    results: number;
    latencyMs: number;
    details?: Record<string, any>;
  }[];
  currentStage: number;
}
```

#### 3.2.4 DocumentCards Component

```tsx
// Expandable cards for each retrieved document:
// - Relevance score bar
// - Source badge (API, tutorial, best practice)
// - Version badge
// - Preview text (truncated)
// - Expand for full content
// - Highlight matching keywords

interface DocumentCardProps {
  documents: {
    id: string;
    content: string;
    title?: string;
    source: string;
    version?: string;
    relevanceScore: number;
    bm25Score?: number;
    denseScore?: number;
    rrfScore: number;
    rerankedPosition: number;
  }[];
  expandedId: string | null;
  onExpand: (id: string) => void;
  highlightTerms: string[];
}
```

#### 3.2.5 ResponsePanel Component

```tsx
// Generated answer display:
// - Main answer text
// - Inline citations (clickable)
// - Validation status badges
// - Confidence indicator
// - Copy button

interface ResponsePanelProps {
  response: {
    text: string;
    citations: { index: number; docId: string; text: string }[];
    validation: {
      isValid: boolean;
      faithfulness: number;
      groundingIssues: string[];
      attempts: number;
    };
  };
  isLoading: boolean;
  onCitationClick: (docId: string) => void;
}
```

#### 3.2.6 MetricsDashboard Component

```tsx
// RAGAS metrics gauges:
// - Faithfulness (0-1)
// - Answer Relevancy (0-1)
// - Context Precision (0-1)
// - Context Recall (0-1)
// - Latency breakdown (pie chart)
// - Composite score

interface MetricsDashboardProps {
  metrics: {
    faithfulness: number;
    answerRelevancy: number;
    contextPrecision: number;
    contextRecall?: number;
    compositeScore: number;
    latencyBreakdown: {
      embedding: number;
      dense: number;
      sparse: number;
      fusion: number;
      rerank: number;
      generation: number;
    };
  };
}
```

#### 3.2.7 SettingsPanel Component

```tsx
// User-configurable settings:
// - Retrieval mode (hybrid, dense-only, sparse-only)
// - Top-K slider (1-50)
// - Reranking toggle
// - Version filter dropdown
// - Category filter
// - Show debug info toggle

interface SettingsPanelProps {
  settings: RAGSettings;
  onChange: (settings: RAGSettings) => void;
}

interface RAGSettings {
  retrievalMode: 'hybrid' | 'dense' | 'sparse';
  topK: number;
  enableReranking: boolean;
  blenderVersion: string | null;
  category: string | null;
  showDebugInfo: boolean;
}
```

---

## 4. Backend API Design

### 4.1 Main Query Endpoint

```typescript
// POST /api/rag/query

interface RAGQueryRequest {
  query: string;
  sessionId?: string;
  settings?: {
    retrievalMode: 'hybrid' | 'dense' | 'sparse';
    topK: number;
    enableReranking: boolean;
    filters?: {
      blenderVersion?: string;
      category?: string;
      source?: string;
    };
  };
}

interface RAGQueryResponse {
  requestId: string;
  sessionId: string;

  // Query analysis
  analysis: {
    intent: string;
    entities: string[];
    variations: string[];
    detectedFilters: Record<string, string>;
  };

  // Retrieval details
  retrieval: {
    denseResults: number;
    sparseResults: number;
    fusedResults: number;
    rerankedResults: number;
    documents: RetrievedDocument[];
  };

  // Generated response
  response: {
    text: string;
    citations: Citation[];
  };

  // Validation
  validation: {
    isValid: boolean;
    faithfulness: number;
    groundingIssues: string[];
    attempts: number;
  };

  // Metrics
  metrics: {
    faithfulness: number;
    answerRelevancy: number;
    contextPrecision: number;
    compositeScore: number;
  };

  // Performance
  latency: {
    total: number;
    embedding: number;
    dense: number;
    sparse: number;
    fusion: number;
    rerank: number;
    generation: number;
    validation: number;
  };
}
```

### 4.2 System Status Endpoint

```typescript
// GET /api/rag/status

interface RAGStatusResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    database: { status: boolean; latencyMs: number };
    embeddingModel: { status: boolean; model: string; latencyMs: number };
    reranker: { status: boolean; model: string };
    llm: { status: boolean; provider: string };
    cache: { status: boolean; hitRate: number };
  };
  documentCount: number;
  lastUpdated: string;
}
```

### 4.3 Mock Data for Demo Mode

For demo purposes (before full backend integration):

```typescript
// lib/rag/mock-data.ts

export const MOCK_DOCUMENTS = [
  {
    id: 'doc-1',
    content: 'To select all faces in Blender, enter Edit Mode by pressing Tab...',
    title: 'Face Selection in Blender',
    source: 'blender_api',
    version: '4.2',
    category: 'mesh',
    relevanceScore: 0.92
  },
  // ... more documents
];

export const MOCK_ANALYSIS = {
  intent: 'select_all_faces',
  entities: ['faces', 'Blender', 'Edit Mode'],
  variations: [
    'How to select faces in Blender',
    'Blender face selection tutorial',
    'bpy.ops.mesh.select_all'
  ],
  detectedFilters: { category: 'mesh' }
};

export const MOCK_METRICS = {
  faithfulness: 0.87,
  answerRelevancy: 0.91,
  contextPrecision: 0.85,
  compositeScore: 0.88
};
```

---

## 5. UI/UX Design Details

### 5.1 shadcn/ui Components Used

| Component | Purpose |
|-----------|---------|
| `Card`, `CardHeader`, `CardContent` | Container panels |
| `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent` | Mode switching |
| `Button` | Actions |
| `Input`, `Textarea` | Query input |
| `Badge` | Status indicators |
| `Progress` | Score bars |
| `Slider` | Settings controls |
| `Switch` | Toggles |
| `Select`, `SelectTrigger`, `SelectContent`, `SelectItem` | Dropdowns |
| `Tooltip` | Help text |
| `Skeleton` | Loading states |
| `Accordion` | Document expansion |
| `ScrollArea` | Scrollable regions |
| `Separator` | Visual dividers |

### 5.2 Animations (Framer Motion)

```tsx
// Pipeline stage animations
const stageVariants = {
  pending: { opacity: 0.5, scale: 0.95 },
  running: { opacity: 1, scale: 1, boxShadow: '0 0 20px rgba(16, 185, 129, 0.5)' },
  complete: { opacity: 1, scale: 1 }
};

// Document card entry
const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1 }
  })
};

// Metrics gauge fill
const gaugeVariants = {
  initial: { pathLength: 0 },
  animate: (value: number) => ({
    pathLength: value,
    transition: { duration: 1, ease: 'easeOut' }
  })
};
```

### 5.3 Responsive Design

```tsx
// Grid layout breakpoints
<div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
  {/* Input column */}
  <div className="lg:col-span-4 space-y-6">
    <QueryInput />
    <SettingsPanel />
  </div>

  {/* Visualization column */}
  <div className="lg:col-span-8 space-y-6">
    <RetrievalFlow />
    <DocumentCards />
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <ResponsePanel />
      <MetricsDashboard />
    </div>
  </div>
</div>
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Day 1)
- [ ] Create `/rag_demo/page.tsx` with basic layout
- [ ] Set up component file structure
- [ ] Implement `QueryInput` component
- [ ] Create mock data file
- [ ] Add basic styling and background effects

### Phase 2: Core Components (Day 2)
- [ ] Implement `SettingsPanel` with all controls
- [ ] Build `DocumentCards` with expand/collapse
- [ ] Create `ResponsePanel` with citations
- [ ] Add loading states with skeletons

### Phase 3: Visualization (Day 3)
- [ ] Build `RetrievalFlow` pipeline animation
- [ ] Implement `MetricsDashboard` with gauges
- [ ] Create `QueryAnalysis` breakdown panel
- [ ] Add framer-motion animations

### Phase 4: Backend Integration (Day 4)
- [ ] Create `/api/rag/query/route.ts`
- [ ] Create `/api/rag/status/route.ts`
- [ ] Connect frontend to API
- [ ] Implement session management

### Phase 5: Polish (Day 5)
- [ ] Add error handling and fallbacks
- [ ] Implement keyboard shortcuts
- [ ] Add help tooltips
- [ ] Performance optimization
- [ ] Mobile responsiveness testing
- [ ] Final visual polish

---

## 7. Sample Queries for Demo

```typescript
const SAMPLE_QUERIES = [
  {
    label: 'Selection',
    query: 'How do I select all faces in Blender?',
    category: 'mesh'
  },
  {
    label: 'Rotation',
    query: 'Rotate objects using Python scripting',
    category: 'object'
  },
  {
    label: 'Materials',
    query: 'Create a procedural material with nodes',
    category: 'shader'
  },
  {
    label: 'Animation',
    query: 'Add keyframes for character animation',
    category: 'animation'
  },
  {
    label: 'Multi-step',
    query: 'Select all faces, rotate them 45 degrees, and apply smooth shading',
    category: 'workflow'
  }
];
```

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Page Load Time | <2s | Lighthouse |
| Time to First Query | <500ms | Console timing |
| Query Response (mock) | <100ms | API latency |
| Query Response (real) | <5s | API latency |
| Mobile Usability | 100% | Lighthouse |
| Accessibility | AA | aXe audit |

---

## 9. Dependencies

### Required (Already Installed)
- `framer-motion` - Animations
- `lucide-react` - Icons
- `@radix-ui/*` - Component primitives
- `tailwind-merge` - Class merging
- `clsx` - Conditional classes

### May Need to Install
- None - all dependencies already in project

---

## 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Backend not ready | High | Use comprehensive mock data |
| Complex visualization | Medium | Start simple, iterate |
| Performance issues | Medium | Use virtual scrolling for docs |
| Mobile layout | Low | Design mobile-first |

---

## 11. Approval Checklist

Before implementation, please confirm:

- [ ] Layout structure approved
- [ ] Color scheme approved (emerald/cyan)
- [ ] Component breakdown approved
- [ ] API design approved
- [ ] Phase timeline approved
- [ ] Sample queries approved

---

**Ready for Implementation:** Awaiting user approval

---

*Generated: December 11, 2025*
*Document: RAG_DEMO_IMPLEMENTATION_PLAN.md*
