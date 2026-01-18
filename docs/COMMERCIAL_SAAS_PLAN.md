# Commercial SaaS Implementation Plan

> **Status**: Awaiting Approval
> **Purpose**: Transform academic presentation into commercial SaaS for fundraising
> **Date**: January 3, 2026

---

## Executive Summary

Transform the current academic 3D Game AI Presentation into a commercial SaaS platform featuring:
- **AI-Powered Salesforce Consulting Marketplace**
- **6 Virtual Avatar Consultants** (3 English, 3 French)
- **Expertise-Based Matching** (User expertise + Avatar specialization)
- **Premium UI** using shadcn MCP components + Freepik assets

---

## 1. Route Architecture

### Public Routes (No Auth Required)
| Route | Component | Description |
|-------|-----------|-------------|
| `/` | LandingPage | Sticky scroll hero with product showcase |
| `/login` | LoginPage | Authentication with social login options |
| `/signup` | SignupPage | User registration with plan selection |

### Protected Routes (Auth Required)
| Route | Component | Description |
|-------|-----------|-------------|
| `/dashboard` | DashboardLayout | Main app shell with dock navigation |
| `/dashboard/home` | DashboardHome | User overview, recent sessions, quick actions |
| `/dashboard/docs` | DocsPage | Documentation, guides, API reference |
| `/dashboard/settings` | SettingsPage | Account, preferences, billing |
| `/dashboard/product` | ProductPage | Enhanced Salesforce demo experience |
| `/dashboard/marketplace` | MarketplacePage | Avatar selection and booking |

### Preserved Routes (Academic Content - Excluded from Build)
```
/technical, /rag, /demo, /full_demo, /training, /avatar_demo
```
These routes remain in codebase but are excluded from production build navigation.

---

## 2. Landing Page Design

### Component: `@ui-layouts/sticky-hero-section`

**Layout Structure:**
```
┌────────────────────────────────────────────────────────────┐
│                    NAVIGATION BAR                          │
│  Logo    Features  Pricing  About    [Login] [Get Started] │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   STICKY SCROLL HERO                                       │
│   ┌──────────────────────────────────────────────────┐    │
│   │  Section 1: AI-Powered Salesforce Consulting     │    │
│   │  "Your Personal Salesforce Expert, 24/7"         │    │
│   │  [Animated avatar preview]                        │    │
│   ├──────────────────────────────────────────────────┤    │
│   │  Section 2: Choose Your Expert                   │    │
│   │  "Match with consultants based on your needs"    │    │
│   │  [3 avatar cards with specializations]           │    │
│   ├──────────────────────────────────────────────────┤    │
│   │  Section 3: Real Conversations                   │    │
│   │  "Natural voice interactions with RAG context"   │    │
│   │  [Demo conversation snippet]                     │    │
│   ├──────────────────────────────────────────────────┤    │
│   │  Section 4: Enterprise Ready                     │    │
│   │  "MCP integration with your Salesforce org"      │    │
│   │  [Integration diagram]                           │    │
│   └──────────────────────────────────────────────────┘    │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                    PRICING SECTION                         │
│   Starter ($29)  |  Pro ($99)  |  Enterprise (Custom)     │
├────────────────────────────────────────────────────────────┤
│                    FOOTER                                  │
└────────────────────────────────────────────────────────────┘
```

**Freepik Assets Needed:**
- Hero background gradient/pattern
- Abstract AI/tech illustrations
- Professional avatar placeholder images
- Integration/workflow icons

---

## 3. Authentication System

### Login Page Component: `@shadcnblocks/login-02`

**Features:**
- Email/password authentication
- Social login (Google, Microsoft - Salesforce ecosystem)
- "Remember me" option
- Forgot password flow
- Link to signup

**Implementation:**
```tsx
// src/app/login/page.tsx
import { LoginForm } from "@/components/auth/login-form"

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 to-slate-800">
      <LoginForm />
    </div>
  )
}
```

### Signup Flow:
1. Email verification
2. Plan selection (Starter/Pro/Enterprise)
3. Expertise assessment quiz (determines user level)
4. Avatar recommendation

---

## 4. Dashboard Architecture

### Layout with Dock Navigation

**Based on Current Implementation:** `presentation-dock.tsx`

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: Logo | Search | Notifications | Profile           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    MAIN CONTENT AREA                        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                                                     │  │
│   │            [Current Route Content]                  │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              MAGNIFIED DOCK (Bottom Center)                 │
│   ┌───────────────────────────────────────────────────┐    │
│   │  🏠    📚    ⚙️    🎯    🛒                       │    │
│   │  Home  Docs  Settings  Product  Marketplace       │    │
│   └───────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Dock Items:**
| Icon | Route | Label |
|------|-------|-------|
| Home | `/dashboard/home` | Dashboard |
| FileText | `/dashboard/docs` | Documentation |
| Settings | `/dashboard/settings` | Settings |
| Target | `/dashboard/product` | Product Demo |
| Store | `/dashboard/marketplace` | Marketplace |

---

## 5. Marketplace - Avatar Consultant Selection

### 6 Avatar Profiles

#### English Avatars (3)

| Avatar | Name | Specialization | Target User |
|--------|------|----------------|-------------|
| **Beginner Guide** | Alex | Basic Salesforce navigation, terminology, simple queries | New Salesforce users |
| **Power User** | Jordan | Reports, dashboards, automation basics, SOQL | Intermediate users |
| **Enterprise Architect** | Morgan | Complex integrations, Apex, LWC, architecture | Advanced developers |

#### French Avatars (3)

| Avatar | Name | Specialization | Target User |
|--------|------|----------------|-------------|
| **Guide Debutant** | Camille | Navigation de base, terminologie, requetes simples | Nouveaux utilisateurs |
| **Utilisateur Avance** | Robin | Rapports, tableaux de bord, automatisation, SOQL | Utilisateurs intermediaires |
| **Architecte Expert** | Dominique | Integrations complexes, Apex, LWC, architecture | Developpeurs avances |

### Marketplace UI Design

```
┌─────────────────────────────────────────────────────────────┐
│  MARKETPLACE HEADER                                         │
│  "Choose Your Salesforce Expert"                            │
│                                                             │
│  [Language Toggle: EN | FR]  [Expertise Filter: All ▼]     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  [Avatar]   │  │  [Avatar]   │  │  [Avatar]   │         │
│  │   Alex      │  │   Jordan    │  │   Morgan    │         │
│  │  ⭐ Beginner │  │  ⭐⭐ Power  │  │  ⭐⭐⭐ Expert │         │
│  │             │  │             │  │             │         │
│  │ "Perfect    │  │ "For users  │  │ "Enterprise │         │
│  │  for        │  │  ready to   │  │  solutions  │         │
│  │  learning"  │  │  scale"     │  │  & Apex"    │         │
│  │             │  │             │  │             │         │
│  │ [Select]    │  │ [Select]    │  │ [Select]    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ───────────── French Consultants ─────────────            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  [Avatar]   │  │  [Avatar]   │  │  [Avatar]   │         │
│  │  Camille    │  │   Robin     │  │  Dominique  │         │
│  │  ⭐ Debutant │  │  ⭐⭐ Avance │  │  ⭐⭐⭐ Expert │         │
│  │             │  │             │  │             │         │
│  │ [Select]    │  │ [Select]    │  │ [Select]    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Matching Algorithm

```typescript
interface UserProfile {
  language: 'en' | 'fr';
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  focus: 'admin' | 'developer' | 'analyst';
}

interface AvatarProfile {
  id: string;
  name: string;
  language: 'en' | 'fr';
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  specializations: string[];
  voiceId: string; // ElevenLabs voice ID
  systemPrompt: string; // RAG context prompt
}

function recommendAvatar(user: UserProfile): AvatarProfile[] {
  return avatars
    .filter(a => a.language === user.language)
    .sort((a, b) => {
      // Score based on expertise match
      const levelMatch = (a.expertiseLevel === user.expertiseLevel) ? 10 : 0;
      // Score based on focus area
      const focusMatch = a.specializations.includes(user.focus) ? 5 : 0;
      return (levelMatch + focusMatch);
    });
}
```

---

## 6. Product Page (Salesforce Demo Evolution)

### Current Structure → Enhanced Structure

**Current Components (to preserve):**
- `ConsultantAvatar.tsx` - 3D avatar display
- `ConversationalInput.tsx` - Voice input handling
- `RAGContext.tsx` - RAG pipeline display
- `SalesforceMCPPanel.tsx` - MCP operations panel

**Enhancements:**

1. **Avatar Selection Integration**
   - Selected avatar from marketplace determines:
     - ElevenLabs voice ID
     - System prompt complexity
     - UI language
     - Example queries

2. **Session Management**
   - Save/load conversation sessions
   - Export conversation transcripts
   - Rate avatar responses

3. **Enhanced RAG Context**
   - Show confidence scores
   - Display source documents
   - Allow feedback on relevance

---

## 7. Freepik API Integration

### API Configuration

```typescript
// src/lib/freepik.ts
const FREEPIK_API_KEY = 'FPSX91458a8c93d4f38269317273e044d399';

export const freepikClient = {
  baseUrl: 'https://api.freepik.com/v1',
  headers: {
    'x-freepik-api-key': FREEPIK_API_KEY,
    'Content-Type': 'application/json'
  }
};

// Search icons
export async function searchIcons(query: string, limit = 10) {
  const response = await fetch(
    `${freepikClient.baseUrl}/icons?query=${encodeURIComponent(query)}&limit=${limit}`,
    { headers: freepikClient.headers }
  );
  return response.json();
}

// Search images/illustrations
export async function searchResources(query: string, type: 'photo' | 'vector' | 'psd') {
  const response = await fetch(
    `${freepikClient.baseUrl}/resources?query=${encodeURIComponent(query)}&type=${type}`,
    { headers: freepikClient.headers }
  );
  return response.json();
}

// Generate AI avatar image
export async function generateAvatarImage(prompt: string) {
  const response = await fetch(
    `${freepikClient.baseUrl}/ai/text-to-image`,
    {
      method: 'POST',
      headers: freepikClient.headers,
      body: JSON.stringify({
        prompt,
        model: 'mystic',
        resolution: '2k'
      })
    }
  );
  return response.json();
}
```

### Asset Usage Plan

| Asset Type | Source | Usage |
|------------|--------|-------|
| Landing page hero | Freepik AI (Mystic) | Custom generated abstract tech art |
| Avatar portraits | Freepik AI + Stock | Professional consultant images |
| Icons (dock, UI) | Freepik Icons API | Consistent icon set |
| Illustrations | Freepik Vectors | Feature explanations |
| Background patterns | Freepik Stock | Section backgrounds |

---

## 8. Technical Implementation

### File Structure

```
src/
├── app/
│   ├── (public)/
│   │   ├── page.tsx                 # Landing page
│   │   ├── login/page.tsx           # Login
│   │   └── signup/page.tsx          # Signup
│   ├── (protected)/
│   │   └── dashboard/
│   │       ├── layout.tsx           # Dashboard shell with dock
│   │       ├── page.tsx             # Redirects to /home
│   │       ├── home/page.tsx        # Dashboard home
│   │       ├── docs/page.tsx        # Documentation
│   │       ├── settings/page.tsx    # Settings
│   │       ├── product/page.tsx     # Salesforce demo
│   │       └── marketplace/page.tsx # Avatar marketplace
│   ├── api/
│   │   ├── auth/[...nextauth]/      # NextAuth.js
│   │   ├── freepik/                 # Freepik proxy API
│   │   └── avatars/                 # Avatar CRUD
│   └── globals.css
├── components/
│   ├── auth/
│   │   ├── login-form.tsx
│   │   └── signup-form.tsx
│   ├── dashboard/
│   │   ├── dashboard-dock.tsx       # Magnified dock
│   │   ├── dashboard-header.tsx
│   │   └── dashboard-shell.tsx
│   ├── landing/
│   │   ├── sticky-hero.tsx
│   │   ├── pricing-section.tsx
│   │   └── features-section.tsx
│   ├── marketplace/
│   │   ├── avatar-card.tsx
│   │   ├── avatar-grid.tsx
│   │   └── expertise-filter.tsx
│   └── salesforce-demo/             # Existing (preserved)
├── lib/
│   ├── freepik.ts                   # Freepik API client
│   ├── auth.ts                      # Auth utilities
│   └── avatars.ts                   # Avatar definitions
└── hooks/
    ├── useAuth.ts
    ├── useFreepik.ts
    └── useAvatar.ts
```

### New Dependencies

```json
{
  "dependencies": {
    "next-auth": "^5.0.0",
    "framer-motion": "^11.0.0",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

### shadcn Components to Install

```bash
# From @shadcn registry
npx shadcn@latest add card button input label tabs avatar badge dialog dropdown-menu

# From @shadcnblocks registry
npx shadcn@latest add login-02 --registry @shadcnblocks

# From @ui-layouts registry
npx shadcn@latest add sticky-hero-section --registry @ui-layouts
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up route groups `(public)` and `(protected)`
- [ ] Install authentication (NextAuth.js)
- [ ] Create dashboard layout with dock
- [ ] Install shadcn components from MCP registries

### Phase 2: Landing & Auth (Week 2)
- [ ] Implement sticky scroll landing page
- [ ] Create login/signup pages
- [ ] Set up Freepik API integration
- [ ] Add social authentication

### Phase 3: Dashboard Core (Week 3)
- [ ] Build dashboard home page
- [ ] Create settings page
- [ ] Implement docs page structure
- [ ] Add user profile management

### Phase 4: Marketplace & Avatars (Week 4)
- [ ] Define 6 avatar profiles
- [ ] Build marketplace UI
- [ ] Implement avatar selection flow
- [ ] Connect to ElevenLabs voices

### Phase 5: Product Enhancement (Week 5)
- [ ] Integrate avatar selection with Salesforce demo
- [ ] Add session management
- [ ] Implement conversation history
- [ ] Add export/share features

### Phase 6: Polish & Launch (Week 6)
- [ ] Add Freepik assets throughout
- [ ] Performance optimization
- [ ] Mobile responsiveness
- [ ] Final testing & bug fixes

---

## 10. Avatar Voice Configuration

### ElevenLabs Voice IDs

| Avatar | Language | Voice Style | Voice ID (Example) |
|--------|----------|-------------|-------------------|
| Alex | EN | Friendly, clear | `21m00Tcm4TlvDq8ikWAM` |
| Jordan | EN | Professional, confident | `AZnzlk1XvdvUeBnXmlld` |
| Morgan | EN | Technical, authoritative | `ErXwobaYiN019PkySvjV` |
| Camille | FR | Welcoming, patient | `MF3mGyEYCl7XYWbV9V6O` |
| Robin | FR | Dynamic, efficient | `TxGEqnHWrfWFTfGW9XjX` |
| Dominique | FR | Expert, precise | `VR6AewLTigWG4xSOukaG` |

### System Prompts (Excerpt)

```typescript
const avatarPrompts = {
  alex: `You are Alex, a friendly Salesforce guide for beginners.
    - Use simple language, avoid jargon
    - Explain concepts step-by-step
    - Offer encouragement and patience
    - Focus on basic navigation and terminology`,

  morgan: `You are Morgan, an enterprise Salesforce architect.
    - Discuss advanced patterns and best practices
    - Reference Apex, LWC, and integration patterns
    - Provide code examples when relevant
    - Consider scalability and security implications`
};
```

---

## 11. Migration Strategy

### Files to Preserve (Academic Content)
```
src/app/technical/
src/app/rag/
src/app/demo/
src/app/full_demo/
src/app/training/
src/app/avatar_demo/
src/components/slides/
src/components/tech-slides/
src/components/rag-slides/
```

### Files to Transform
```
src/app/page.tsx → New landing page (backup old as slides-home.tsx)
src/app/salesforce_demo/ → /dashboard/product/
src/components/salesforce-demo/ → Enhanced with avatar integration
```

### Build Exclusion
```typescript
// next.config.ts
module.exports = {
  experimental: {
    // Exclude academic routes from production sitemap
  },
  async redirects() {
    return process.env.NODE_ENV === 'production' ? [
      { source: '/technical', destination: '/dashboard', permanent: false },
      { source: '/rag', destination: '/dashboard', permanent: false },
      { source: '/demo', destination: '/dashboard/product', permanent: false },
    ] : [];
  }
};
```

---

## 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Landing page load time | < 2s | Lighthouse |
| User signup conversion | > 5% | Analytics |
| Avatar session duration | > 5 min | Session tracking |
| User satisfaction | > 4.2/5 | In-app rating |
| Mobile responsiveness | 100% | Cross-device testing |

---

## Approval Checklist

Before proceeding, please confirm:

- [ ] Route structure approved
- [ ] 6 avatar configuration approved
- [ ] UI component selection approved (sticky-hero, login-02)
- [ ] Freepik API usage approved
- [ ] Migration strategy approved
- [ ] Implementation timeline acceptable

---

**Awaiting your approval to begin implementation.**
