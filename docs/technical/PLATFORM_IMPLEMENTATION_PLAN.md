# SF Consultant AI - Platform Implementation Plan

## Executive Summary

This document outlines the comprehensive plan to make the SF Consultant AI platform fully functional based on **accurate codebase exploration**. The platform already has a working MVP at `/dashboard/product` - we need to create the `/agents` selection page and wire up the dashboard/settings with backend persistence.

---

## Part 1: Accurate Current State Analysis

### What's Already Built and Working

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Login Page** | ✅ Complete | `/src/app/login/page.tsx` | Simulated auth with demo credentials |
| **Landing Page** | ✅ Complete | `/src/app/page.tsx` | All sections designed |
| **Product MVP** | ✅ **FULLY WORKING** | `/src/app/dashboard/product/page.tsx` | 768-line full implementation |
| **Avatar System** | ✅ Complete | `/src/lib/avatars.ts` | 6 avatars defined with all metadata |
| **ElevenLabs Hook** | ✅ Complete | `/src/hooks/useElevenLabsConversation.ts` | WebSocket voice conversation |
| **RAG Components** | ✅ Complete | `/src/components/salesforce-demo/RAGContext.tsx` | 9-stage pipeline visualization |
| **MCP Components** | ✅ Complete | `/src/components/salesforce-demo/SalesforceMCPPanel.tsx` | Salesforce operations panel |
| **Conversation Panel** | ✅ Complete | `/src/components/salesforce-demo/ConversationPanel.tsx` | Message history with tool calls |
| **Avatar Display** | ✅ Complete | `/src/components/salesforce-demo/ConsultantAvatar.tsx` | Status-aware animated avatar |
| **API Routes** | ✅ Complete | `/src/app/api/` | ElevenLabs, Salesforce, RAG routes |
| **Flask Backend** | ✅ Configured | `/backend/app.py` | Blueprints for RAG, MCP, Avatar |
| **Environment** | ✅ Configured | `.env.local`, `backend/.env` | API keys ready |

### Configured API Keys

```
ElevenLabs API Key: ✅ Configured (sk_c30f2a70...)
OpenAI API Key: ✅ Configured (sk-proj-...)
Flask Backend URL: ✅ http://localhost:5000
Salesforce OAuth: ⚠️ Placeholder (demo mode works)
```

### 6 Avatars Already Defined (`/src/lib/avatars.ts`)

| ID | Name | Language | Level | Voice ID | Accent Color |
|----|------|----------|-------|----------|--------------|
| alex | Alex Thompson | EN | Beginner | TxGEqnHWrfWFTfGW9XjX | #00A1E0 |
| jordan | Jordan Mitchell | EN | Intermediate | VR6AewLTigWG4xSOukaG | #FF6B35 |
| morgan | Morgan Chen | EN | Advanced | EXAVITQu4vr4xnSDxMaL | #8B5CF6 |
| camille | Camille Dupont | FR | Beginner | D38z5RcWu1voky8WS1ja | #00A1E0 |
| robin | Robin Lefebvre | FR | Intermediate | cgSgspJ2msm6clMCkdW9 | #FF6B35 |
| dominique | Dominique Martin | FR | Advanced | Yko7PKs66fRI469Y4LCl | #8B5CF6 |

Each avatar includes:
- `systemPrompt` - Personality and expertise description
- `stats` - Experience, rating, sessions, specialties
- `expertiseLevel` - beginner | intermediate | advanced

### What Needs Implementation

| Feature | Priority | Effort |
|---------|----------|--------|
| **`/agents` page** | P0 | Medium |
| **Dashboard real data** | P1 | Low |
| **Settings persistence** | P1 | Low |
| **Session storage** | P2 | Medium |

---

## Part 2: Product MVP Analysis

### Current Product Page Features (`/dashboard/product/page.tsx`)

The MVP is **fully functional** with this 3-column layout:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRODUCT MVP LAYOUT                          │
├────────────────────┬────────────────────┬──────────────────────────┤
│   LEFT COLUMN      │   CENTER COLUMN    │    RIGHT COLUMN          │
│                    │                    │                          │
│ ┌────────────────┐ │ ┌────────────────┐ │ ┌──────────────────────┐ │
│ │ ConsultantAvatar│ │ │ Salesforce     │ │ │ RAGContext           │ │
│ │ (video/animated)│ │ │ Connection     │ │ │ - 9 Pipeline Stages  │ │
│ │                │ │ │ - OAuth Status │ │ │ - Query Analysis     │ │
│ │ Status Badges  │ │ │ - Org Info     │ │ │ - Retrieved Docs     │ │
│ └────────────────┘ │ └────────────────┘ │ │ - RAGAS Metrics      │ │
│                    │                    │ └──────────────────────┘ │
│ ┌────────────────┐ │ ┌────────────────┐ │                          │
│ │ ConversationIn │ │ │ SalesforceMCP  │ │                          │
│ │ - Mic Button   │ │ │ Panel          │ │                          │
│ │ - Text Input   │ │ │ - Operations   │ │                          │
│ │ - Voice Status │ │ │ - SOQL Queries │ │                          │
│ └────────────────┘ │ │ - Results      │ │                          │
│                    │ └────────────────┘ │                          │
│ ┌────────────────┐ │                    │                          │
│ │ Conversation   │ │                    │                          │
│ │ Panel          │ │                    │                          │
│ │ - Messages     │ │                    │                          │
│ │ - Tool Calls   │ │                    │                          │
│ └────────────────┘ │                    │                          │
│                    │                    │                          │
│ [Avatar Switcher]  │                    │                          │
└────────────────────┴────────────────────┴──────────────────────────┘
```

### How Avatar Selection Works

```typescript
// Product page reads avatar from URL
const searchParams = useSearchParams();
const avatarId = searchParams.get("avatar") || "alex";
const currentAvatar = getAvatarById(avatarId);

// Navigation: /dashboard/product?avatar=morgan
```

### Key Integration Points

1. **ElevenLabs Agent ID**: `agent_7001kdqegdr4eyct05t0cawfwxtf` (hardcoded)
2. **Voice Selection**: Uses avatar's `voiceId` for conversation
3. **RAG API**: `POST /api/salesforce/rag` with demo fallback
4. **MCP API**: `POST /api/salesforce/mcp/*` for Salesforce operations

---

## Part 3: `/agents` Page Specification

### Purpose

Entry point to the consultation experience. Users select:
1. Language (EN/FR)
2. Expertise level (Beginner/Intermediate/Advanced)
3. Specific avatar (filtered by language + level)
4. Session type (Start Now or Schedule Later)

### User Flow

```
/agents
  ├── Step 1: Select Language → Filters to 3 avatars
  │     [🇬🇧 English]  [🇫🇷 French]
  │
  ├── Step 2: Select Expertise Level → Filters to 1 avatar
  │     [Beginner]  [Intermediate]  [Expert]
  │
  ├── Step 3: View Selected Consultant Preview
  │     ┌─────────────────────────────────────────┐
  │     │  [Avatar Image]                         │
  │     │  Morgan Chen                            │
  │     │  Senior Enterprise Architect            │
  │     │  ⭐ 4.9 · 🎓 8+ years · 💬 1,200+ sessions │
  │     │                                         │
  │     │  Specialties:                           │
  │     │  ✓ Multi-cloud architecture             │
  │     │  ✓ Complex integrations                 │
  │     │  ✓ Enterprise-scale solutions           │
  │     └─────────────────────────────────────────┘
  │
  └── Step 4: Session Action
        [🚀 Start Consultation] → /dashboard/product?avatar=morgan
        [📅 Schedule Later] → Calendar modal (future)
```

### Page Layout Design

```
┌─────────────────────────────────────────────────────────────────────┐
│  [Navbar]                                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   🎯 Choose Your AI Consultant                                       │
│   Select your preferred language and expertise level                 │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │  LANGUAGE                                                      │ │
│   │                                                                │ │
│   │  ┌───────────────────┐  ┌───────────────────┐                 │ │
│   │  │ 🇬🇧 English        │  │ 🇫🇷 French          │                 │ │
│   │  │ 3 consultants     │  │ 3 consultants      │                 │ │
│   │  │ [●] Selected      │  │ [ ] Select         │                 │ │
│   │  └───────────────────┘  └───────────────────┘                 │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │  EXPERTISE LEVEL                                               │ │
│   │                                                                │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │
│   │  │  Beginner   │  │ Intermediate│  │   Expert    │            │ │
│   │  │  Alex       │  │  Jordan     │  │   Morgan    │            │ │
│   │  │  Basics &   │  │  Advanced   │  │  Enterprise │            │ │
│   │  │  Foundations│  │  Topics     │  │  Architecture│           │ │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │  YOUR CONSULTANT                                               │ │
│   │                                                                │ │
│   │  ┌────────┐                                                   │ │
│   │  │ Avatar │  Morgan Chen                                      │ │
│   │  │ Image  │  Senior Enterprise Architect                      │ │
│   │  │  👤    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                │ │
│   │  └────────┘  ⭐ 4.9 Rating  🎓 8+ Years  💬 1,200 Sessions    │ │
│   │                                                                │ │
│   │  "Your expert for complex Salesforce enterprise solutions,    │ │
│   │   multi-cloud architecture, and advanced integrations."       │ │
│   │                                                                │ │
│   │  Specialties:                                                  │ │
│   │  ┌───────────────┐ ┌──────────────┐ ┌────────────────┐       │ │
│   │  │Multi-cloud    │ │ API Design   │ │ Scalability    │       │ │
│   │  └───────────────┘ └──────────────┘ └────────────────┘       │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                                                                │ │
│   │  ┌─────────────────────────┐  ┌─────────────────────────┐    │ │
│   │  │  🚀 Start Consultation  │  │  📅 Schedule for Later  │    │ │
│   │  │  Begin your session now │  │  Pick a date & time     │    │ │
│   │  └─────────────────────────┘  └─────────────────────────┘    │ │
│   │                                                                │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  [Footer]                                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```typescript
// State management for /agents page
interface AgentSelectionState {
  language: "en" | "fr";
  level: "beginner" | "intermediate" | "advanced";
  selectedAvatar: Avatar | null;
}

// Filtering logic
const filteredAvatars = avatars.filter(a =>
  a.language === state.language &&
  a.expertiseLevel === state.level
);

// Navigation on "Start Now"
router.push(`/dashboard/product?avatar=${selectedAvatar.id}`);
```

### Components to Create

| Component | Description |
|-----------|-------------|
| `LanguageSelector` | Two toggle cards for EN/FR |
| `LevelSelector` | Three cards showing level + matching avatar name |
| `ConsultantCard` | Large preview card with avatar stats |
| `SessionActions` | Start Now + Schedule buttons |

---

## Part 4: Dashboard Implementation

### Current State

The dashboard at `/src/app/dashboard/page.tsx` shows:
- **Hardcoded stats**: 127 conversations, 42h saved, etc.
- **Recent sessions**: Static data
- **Quick access cards**: Links to `/dashboard/product?avatar={id}`
- **Language filter**: Toggles EN/FR avatars

### Required Changes

1. **Connect stats to backend** - Create API endpoint for user stats
2. **Real recent sessions** - Store/retrieve session history
3. **Consultant cards** - Already link correctly to product page

### Backend API Endpoints Needed

```python
# User Stats
GET /api/user/stats
Response: {
  "total_conversations": 127,
  "hours_saved": 42,
  "active_consultants": 4,
  "questions_answered": 523
}

# Recent Sessions
GET /api/sessions?limit=5
Response: {
  "sessions": [
    {
      "id": "sess_001",
      "consultant_id": "morgan",
      "title": "Apex Trigger Best Practices",
      "duration_minutes": 12,
      "created_at": "2025-01-02T14:30:00Z"
    }
  ]
}
```

---

## Part 5: Settings Implementation

### Current State

Settings page at `/src/app/dashboard/settings/page.tsx` has 5 sections:
- Profile (name, email, company)
- Notifications (email toggles)
- Preferences (language, timezone, voice)
- Security (password change)
- Billing (subscription info)

**Problem**: All state is local - clicking Save shows "Saved!" but nothing persists.

### Required Changes

1. **Create settings API endpoints**
2. **Load settings on page mount**
3. **Save settings on button click**

### Backend API Endpoints Needed

```python
# Get Settings
GET /api/settings
Response: {
  "profile": {
    "name": "John Doe",
    "email": "john@example.com",
    "company": "Acme Corp"
  },
  "notifications": {
    "email_sessions": true,
    "email_updates": true,
    "email_tips": false
  },
  "preferences": {
    "language": "en",
    "timezone": "America/New_York",
    "voice_enabled": true
  }
}

# Update Settings
PUT /api/settings
Body: { ... settings object ... }
Response: { "success": true }
```

---

## Part 6: Implementation Plan

### Phase 1: Create `/agents` Page (Priority 0)

**Goal**: Entry point for consultant selection

**Files to Create**:
- `src/app/agents/page.tsx` - Main agents selection page
- `src/components/agents/LanguageSelector.tsx`
- `src/components/agents/LevelSelector.tsx`
- `src/components/agents/ConsultantCard.tsx`
- `src/components/agents/SessionActions.tsx`

**Implementation Steps**:
1. Create page layout with gradient background
2. Implement language selector (EN/FR toggle)
3. Implement level selector (3 cards)
4. Create consultant preview card using avatar data
5. Add "Start Now" button → navigates to `/dashboard/product?avatar={id}`
6. Add "Schedule Later" button (placeholder for future)

### Phase 2: Wire Dashboard to Real Data (Priority 1)

**Goal**: Show actual user stats and sessions

**Backend Changes**:
1. Add `user_stats` table to SQLAlchemy models
2. Add `sessions` table with conversation history
3. Create `/api/user/stats` endpoint
4. Create `/api/sessions` endpoint

**Frontend Changes**:
1. Create `useUserStats` hook to fetch stats
2. Create `useSessions` hook to fetch recent sessions
3. Replace hardcoded data in dashboard

### Phase 3: Settings Persistence (Priority 1)

**Goal**: Save and load user settings

**Backend Changes**:
1. Add `user_settings` table
2. Create `GET /api/settings` endpoint
3. Create `PUT /api/settings` endpoint

**Frontend Changes**:
1. Add `useSettings` hook
2. Load settings on page mount
3. Save settings on button click

### Phase 4: Session Recording (Priority 2)

**Goal**: Store conversation transcripts

**Implementation**:
1. Hook into conversation end event
2. Send transcript to `/api/sessions` POST
3. Display in dashboard history

---

## Part 7: File Structure

### New Files

```
src/
├── app/
│   └── agents/
│       └── page.tsx                 # NEW: Agent selection page
├── components/
│   └── agents/
│       ├── LanguageSelector.tsx     # NEW
│       ├── LevelSelector.tsx        # NEW
│       ├── ConsultantCard.tsx       # NEW
│       └── SessionActions.tsx       # NEW
├── hooks/
│   ├── useUserStats.ts              # NEW: Fetch user statistics
│   ├── useSessions.ts               # NEW: Fetch session history
│   └── useSettings.ts               # NEW: Settings CRUD

backend/
├── models/
│   ├── user_stats.py                # NEW: Stats model
│   ├── sessions.py                  # NEW: Sessions model
│   └── user_settings.py             # NEW: Settings model
├── routes/
│   ├── stats_api.py                 # NEW: Stats endpoints
│   ├── sessions_api.py              # NEW: Sessions endpoints
│   └── settings_api.py              # NEW: Settings endpoints
```

### Existing Files to Modify

| File | Changes |
|------|---------|
| `src/app/dashboard/page.tsx` | Use hooks instead of hardcoded data |
| `src/app/dashboard/settings/page.tsx` | Add API calls on load/save |
| `backend/app.py` | Register new blueprints |

---

## Part 8: Technical Details

### Avatar Lookup Utility

The existing `getAvatarById` and `getAvatarsByLanguage` functions in `/src/lib/avatars.ts` provide all needed filtering:

```typescript
import { avatars, getAvatarById, getAvatarsByLanguage } from "@/lib/avatars";

// Get avatars by language
const enAvatars = getAvatarsByLanguage("en"); // [alex, jordan, morgan]
const frAvatars = getAvatarsByLanguage("fr"); // [camille, robin, dominique]

// Get specific avatar
const morgan = getAvatarById("morgan");
```

### Navigation Pattern

```typescript
// From /agents to /dashboard/product
import { useRouter } from "next/navigation";

const router = useRouter();

const handleStartSession = () => {
  router.push(`/dashboard/product?avatar=${selectedAvatar.id}`);
};
```

### Product Page Avatar Consumption

The product page already handles avatar selection via URL:

```typescript
// In /dashboard/product/page.tsx (existing code)
const searchParams = useSearchParams();
const avatarId = searchParams.get("avatar") || "alex";
const currentAvatar = getAvatarById(avatarId);
```

---

## Part 9: Success Criteria

### MVP Success (What must work)

- [ ] `/agents` page displays all 6 avatars filtered by language
- [ ] Level filter shows correct avatar for each level
- [ ] Consultant card shows full avatar details (stats, specialties)
- [ ] "Start Now" navigates to product page with correct avatar
- [ ] Product page loads selected avatar correctly
- [ ] Voice conversation works with selected avatar's voice

### Full Success (Nice to have)

- [ ] Dashboard shows real statistics
- [ ] Recent sessions display actual history
- [ ] Settings persist across sessions
- [ ] "Schedule Later" opens calendar picker
- [ ] Scheduled sessions appear in dashboard

---

## Approval Request

**Implementation Ready:**

1. ✅ `/agents` page creation - All data/components mapped
2. ✅ Dashboard data flow - API endpoints defined
3. ✅ Settings persistence - Clear implementation path
4. ✅ Environment configured - API keys ready

**Awaiting Approval:**
- Proceed with Phase 1 (`/agents` page)?
- Proceed with Phase 2 (Dashboard real data)?
- Proceed with Phase 3 (Settings persistence)?

Once approved, implementation will begin immediately with the `/agents` page.
