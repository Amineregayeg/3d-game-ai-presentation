# Persona Selection Feature - Implementation Plan

## Overview

Add a persona selection page before the Salesforce demo where users choose from 4 AI consultants (2 English, 2 French), each with unique voice, personality, and avatar.

---

## 🎭 The 4 Personas

### English Personas

| Persona | Name | Voice | Personality | Avatar |
|---------|------|-------|-------------|--------|
| **EN-M** | **Alex** | Bill (pqHfZKP75CvOlQylNhV4) - Wise, Mature | Senior consultant, 15+ years, direct & confident | Male, 40s, professional |
| **EN-F** | **Sarah** | Sarah (EXAVITQu4vr4xnSDxMaL) - Reassuring | Solution architect, empathetic, thorough | Female, 35, approachable |

### French Personas

| Persona | Name | Voice (from shared library) | Personality | Avatar |
|---------|------|------------------------------|-------------|--------|
| **FR-M** | **Laurent** | Mr. Laurent (necQJzI1X0vLpdnJteap) - Warm, friendly | Expert Salesforce France, 12 ans d'expérience | Male, 45, distinguished |
| **FR-F** | **Amélie** | Amélie (39BbQfJTexvpWtOQZ4Xr) - Warm and Gentle | Consultante senior, spécialiste automatisation | Female, 32, dynamic |

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NEW FLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  /salesforce_demo                    /salesforce_demo/chat       │
│  ┌──────────────────────┐           ┌──────────────────────┐    │
│  │   PERSONA SELECTION  │  ──────►  │   CURRENT DEMO UI    │    │
│  │                      │  (click)  │   (with selected     │    │
│  │  ┌────┐    ┌────┐   │           │    persona config)   │    │
│  │  │Alex│    │Sarah│   │           │                      │    │
│  │  └────┘    └────┘   │           └──────────────────────┘    │
│  │  ┌────┐    ┌────┐   │                                        │
│  │  │Laurent│ │Amélie│  │                                        │
│  │  └────┘    └────┘   │                                        │
│  └──────────────────────┘                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ File Structure

```
src/
├── app/
│   └── salesforce_demo/
│       ├── page.tsx              # NEW: Persona selection page
│       └── chat/
│           └── page.tsx          # MOVED: Current demo (renamed)
│
├── components/
│   └── salesforce-demo/
│       ├── PersonaSelection.tsx  # NEW: Persona cards grid
│       ├── PersonaCard.tsx       # NEW: Individual persona card
│       ├── personas.ts           # NEW: Persona configurations
│       └── ...existing...
│
└── lib/
    └── persona-context.tsx       # NEW: React context for selected persona
```

---

## 📋 Implementation Steps

### Phase 1: Setup Personas Config

**File: `src/components/salesforce-demo/personas.ts`**

```typescript
export interface Persona {
  id: string;
  name: string;
  language: 'en' | 'fr';
  gender: 'male' | 'female';
  voiceId: string;
  voiceSource: 'default' | 'shared'; // shared = from voice library
  agentId?: string; // ElevenLabs agent ID (to be created)

  // Display
  title: string;
  subtitle: string;
  description: string;
  avatarUrl: string;
  accentColor: string;

  // AI Config
  systemPrompt: string;
  firstMessage: string;
  traits: string[];
  specializations: string[];
}

export const PERSONAS: Persona[] = [
  {
    id: 'alex',
    name: 'Alex',
    language: 'en',
    gender: 'male',
    voiceId: 'pqHfZKP75CvOlQylNhV4', // Bill
    voiceSource: 'default',
    title: 'Senior Salesforce Consultant',
    subtitle: '15+ years experience • 10x Certified',
    description: 'Direct, confident, and efficient. Alex gets straight to the point with battle-tested solutions.',
    avatarUrl: '/avatars/alex.png',
    accentColor: '#0176D3', // Salesforce blue
    systemPrompt: `You are Alex, a senior Salesforce consultant...`,
    firstMessage: "Hi, I'm Alex. 15 years in the Salesforce ecosystem. What are we solving today?",
    traits: ['Direct', 'Confident', 'Efficient'],
    specializations: ['Sales Cloud', 'CPQ', 'Integration']
  },
  {
    id: 'sarah',
    name: 'Sarah',
    language: 'en',
    gender: 'female',
    voiceId: 'EXAVITQu4vr4xnSDxMaL', // Sarah
    voiceSource: 'default',
    title: 'Solution Architect',
    subtitle: '12 years experience • Platform Specialist',
    description: 'Thorough and empathetic. Sarah ensures you understand every step of the solution.',
    avatarUrl: '/avatars/sarah.png',
    accentColor: '#1B96FF', // Lightning blue
    systemPrompt: `You are Sarah, a Salesforce solution architect...`,
    firstMessage: "Hello! I'm Sarah. I specialize in designing scalable Salesforce solutions. Tell me about your challenge.",
    traits: ['Thorough', 'Empathetic', 'Educational'],
    specializations: ['Architecture', 'Lightning', 'Data Model']
  },
  {
    id: 'laurent',
    name: 'Laurent',
    language: 'fr',
    gender: 'male',
    voiceId: 'necQJzI1X0vLpdnJteap', // Mr. Laurent (shared)
    voiceSource: 'shared',
    title: 'Expert Salesforce France',
    subtitle: '12 ans d\'expérience • Partenaire Certifié',
    description: 'Chaleureux et rassurant. Laurent vous guide avec expertise dans l\'écosystème Salesforce.',
    avatarUrl: '/avatars/laurent.png',
    accentColor: '#032D60', // Salesforce dark blue
    systemPrompt: `Tu es Laurent, un expert Salesforce français...`,
    firstMessage: "Bonjour ! Je suis Laurent, expert Salesforce depuis 12 ans. Comment puis-je vous aider aujourd'hui ?",
    traits: ['Chaleureux', 'Expert', 'Pédagogue'],
    specializations: ['Sales Cloud', 'Service Cloud', 'Intégration']
  },
  {
    id: 'amelie',
    name: 'Amélie',
    language: 'fr',
    gender: 'female',
    voiceId: '39BbQfJTexvpWtOQZ4Xr', // Amélie (shared)
    voiceSource: 'shared',
    title: 'Consultante Senior',
    subtitle: '8 ans d\'expérience • Spécialiste Automatisation',
    description: 'Dynamique et précise. Amélie transforme vos processus avec des solutions élégantes.',
    avatarUrl: '/avatars/amelie.png',
    accentColor: '#00A1E0', // Salesforce light blue
    systemPrompt: `Tu es Amélie, consultante Salesforce senior...`,
    firstMessage: "Salut ! Moi c'est Amélie. Je suis passionnée par l'automatisation Salesforce. Qu'est-ce qu'on optimise aujourd'hui ?",
    traits: ['Dynamique', 'Précise', 'Créative'],
    specializations: ['Flow', 'Automation', 'Lightning Web Components']
  }
];
```

### Phase 2: Create ElevenLabs Agents

**Action Required**: Create 4 agents in ElevenLabs via API:

```bash
# For each persona, call:
POST /api/elevenlabs/agent
{
  "action": "create",
  "config": {
    "name": "Forward - {PersonaName}",
    "first_message": "{persona.firstMessage}",
    "system_prompt": "{persona.systemPrompt}",
    "voice_id": "{persona.voiceId}",
    "language": "{persona.language}"
  }
}
```

Store resulting `agent_id` in personas config.

### Phase 3: Add Shared Voices to Library

**Action Required**: Add French voices from shared library:

```bash
# Add voice to library
POST https://api.elevenlabs.io/v1/voices/add
{
  "voice_id": "necQJzI1X0vLpdnJteap"  # Mr. Laurent
}
```

### Phase 4: Persona Selection Page

**File: `src/app/salesforce_demo/page.tsx`** (NEW)

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { PERSONAS, Persona } from "@/components/salesforce-demo/personas";
import { PersonaCard } from "@/components/salesforce-demo/PersonaCard";

export default function PersonaSelectionPage() {
  const router = useRouter();
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const handleSelect = (persona: Persona) => {
    // Store in localStorage or URL params
    localStorage.setItem('selectedPersona', JSON.stringify(persona));
    router.push('/salesforce_demo/chat');
  };

  const englishPersonas = PERSONAS.filter(p => p.language === 'en');
  const frenchPersonas = PERSONAS.filter(p => p.language === 'fr');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="pt-12 pb-8 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">
          Choose Your Consultant
        </h1>
        <p className="text-slate-400 text-lg">
          Select an AI consultant to help you with Salesforce
        </p>
      </header>

      {/* English Section */}
      <section className="max-w-6xl mx-auto px-6 mb-12">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          🇬🇧 English Consultants
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {englishPersonas.map(persona => (
            <PersonaCard
              key={persona.id}
              persona={persona}
              isHovered={hoveredId === persona.id}
              onHover={() => setHoveredId(persona.id)}
              onLeave={() => setHoveredId(null)}
              onSelect={() => handleSelect(persona)}
            />
          ))}
        </div>
      </section>

      {/* French Section */}
      <section className="max-w-6xl mx-auto px-6 pb-16">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          🇫🇷 Consultants Français
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {frenchPersonas.map(persona => (
            <PersonaCard
              key={persona.id}
              persona={persona}
              isHovered={hoveredId === persona.id}
              onHover={() => setHoveredId(persona.id)}
              onLeave={() => setHoveredId(null)}
              onSelect={() => handleSelect(persona)}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
```

### Phase 5: Persona Card Component

**File: `src/components/salesforce-demo/PersonaCard.tsx`**

```tsx
"use client";

import { motion } from "framer-motion";
import { Mic, Globe, Zap, ArrowRight } from "lucide-react";
import { Persona } from "./personas";

interface PersonaCardProps {
  persona: Persona;
  isHovered: boolean;
  onHover: () => void;
  onLeave: () => void;
  onSelect: () => void;
}

export function PersonaCard({ persona, isHovered, onHover, onLeave, onSelect }: PersonaCardProps) {
  return (
    <motion.div
      className="relative rounded-2xl overflow-hidden cursor-pointer group"
      style={{ backgroundColor: persona.accentColor + '20' }}
      whileHover={{ scale: 1.02 }}
      onMouseEnter={onHover}
      onMouseLeave={onLeave}
      onClick={onSelect}
    >
      <div className="p-6 flex gap-6">
        {/* Avatar */}
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-4xl">
          {persona.gender === 'male' ? '👨‍💼' : '👩‍💼'}
        </div>

        {/* Info */}
        <div className="flex-1">
          <h3 className="text-2xl font-bold text-white">{persona.name}</h3>
          <p className="text-slate-300 text-sm">{persona.title}</p>
          <p className="text-slate-500 text-xs mt-1">{persona.subtitle}</p>

          {/* Traits */}
          <div className="flex gap-2 mt-3">
            {persona.traits.map(trait => (
              <span
                key={trait}
                className="px-2 py-0.5 text-xs rounded-full"
                style={{ backgroundColor: persona.accentColor + '40', color: 'white' }}
              >
                {trait}
              </span>
            ))}
          </div>

          {/* Description */}
          <p className="text-slate-400 text-sm mt-3">
            {persona.description}
          </p>
        </div>

        {/* Select Arrow */}
        <motion.div
          className="self-center"
          animate={{ x: isHovered ? 5 : 0 }}
        >
          <ArrowRight className="w-6 h-6 text-white opacity-50 group-hover:opacity-100" />
        </motion.div>
      </div>

      {/* Accent border */}
      <div
        className="absolute bottom-0 left-0 right-0 h-1"
        style={{ backgroundColor: persona.accentColor }}
      />
    </motion.div>
  );
}
```

### Phase 6: Move Current Demo

**Action**: Rename/move current page:
- `src/app/salesforce_demo/page.tsx` → `src/app/salesforce_demo/chat/page.tsx`
- Update imports to read selected persona from localStorage

### Phase 7: Update Chat Page for Persona

**Modify chat/page.tsx**:

```tsx
// At the top of component
const [persona, setPersona] = useState<Persona | null>(null);

useEffect(() => {
  const stored = localStorage.getItem('selectedPersona');
  if (stored) {
    setPersona(JSON.parse(stored));
  } else {
    // Redirect to selection if no persona
    router.push('/salesforce_demo');
  }
}, []);

// Use persona.agentId when starting conversation
// Use persona.voiceId for TTS fallback
// Use persona.systemPrompt for RAG context
// Display persona.name in UI
```

---

## 🎨 UI Design

### Persona Selection Page

```
┌─────────────────────────────────────────────────────────────────┐
│                     Choose Your Consultant                       │
│              Select an AI consultant to help you                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🇬🇧 English Consultants                                         │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │  👨‍💼  ALEX               →│  │  👩‍💼  SARAH              →│       │
│  │  Senior Consultant       │  │  Solution Architect      │       │
│  │  15+ years • 10x Cert    │  │  12 years • Platform     │       │
│  │                          │  │                          │       │
│  │  [Direct] [Confident]    │  │  [Thorough] [Empathetic] │       │
│  │                          │  │                          │       │
│  │  Gets straight to the    │  │  Ensures you understand  │       │
│  │  point with solutions.   │  │  every step.             │       │
│  │━━━━━━━━━━━━━━━━━━━━━━━━━│  │━━━━━━━━━━━━━━━━━━━━━━━━━│       │
│  └─────────────────────────┘  └─────────────────────────┘       │
│                                                                  │
│  🇫🇷 Consultants Français                                        │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │  👨‍💼  LAURENT            →│  │  👩‍💼  AMÉLIE             →│       │
│  │  Expert Salesforce       │  │  Consultante Senior      │       │
│  │  12 ans • Partenaire     │  │  8 ans • Automatisation  │       │
│  │                          │  │                          │       │
│  │  [Chaleureux] [Expert]   │  │  [Dynamique] [Créative]  │       │
│  │                          │  │                          │       │
│  │  Vous guide avec         │  │  Transforme vos process  │       │
│  │  expertise.              │  │  avec élégance.          │       │
│  │━━━━━━━━━━━━━━━━━━━━━━━━━│  │━━━━━━━━━━━━━━━━━━━━━━━━━│       │
│  └─────────────────────────┘  └─────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 ElevenLabs Setup Tasks

### 1. Add Shared Voices to Library

```bash
# Mr. Laurent (French male)
curl -X POST "https://api.elevenlabs.io/v1/voices/add/necQJzI1X0vLpdnJteap" \
  -H "xi-api-key: $ELEVENLABS_API_KEY"

# Amélie (French female)
curl -X POST "https://api.elevenlabs.io/v1/voices/add/39BbQfJTexvpWtOQZ4Xr" \
  -H "xi-api-key: $ELEVENLABS_API_KEY"
```

### 2. Create 4 Conversational Agents

Each agent needs:
- Unique name
- Language-specific system prompt
- Correct voice_id
- First message in correct language

---

## ⏱️ Estimated Effort

| Task | Effort |
|------|--------|
| Personas config file | 30 min |
| ElevenLabs agent setup | 1 hour |
| Persona selection page | 1 hour |
| PersonaCard component | 45 min |
| Move/update chat page | 1 hour |
| Context integration | 1 hour |
| Testing & polish | 1 hour |
| **Total** | **~6 hours** |

---

## ✅ Success Criteria

1. User lands on persona selection page at `/salesforce_demo`
2. 4 personas displayed (2 EN, 2 FR) with clear visual distinction
3. Clicking a persona navigates to chat with that consultant
4. Voice matches selected persona (language + gender)
5. System prompt adapts to persona personality
6. First message is in correct language
7. UI shows selected persona name/avatar during chat

---

## 🔮 Future Enhancements

1. **Custom Avatars**: Generate unique avatar images for each persona
2. **Voice Preview**: Play sample before selecting
3. **Persona Memory**: Remember last used persona
4. **Animated Avatars**: SadTalker/Wav2Lip integration per persona
5. **More Personas**: Add Spanish, German consultants
