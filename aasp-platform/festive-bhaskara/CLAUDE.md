# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AASP (AI Agent Security Platform) is a security control plane for AI agents. It intercepts agent actions (tool calls, API requests, database queries), evaluates them against security policies, and either allows, blocks, or requires human approval before execution.

## Commands

```bash
# Development
pnpm dev          # Start Next.js dev server at http://localhost:3000
pnpm build        # Production build
pnpm lint         # Run ESLint

# Database (Prisma with SQLite)
npx prisma generate    # Generate Prisma client
npx prisma db push     # Push schema changes to database
npx prisma studio      # Open Prisma Studio GUI

# Python SDK (in sdk/python/)
pip install -e ".[dev,langchain]"   # Install SDK with dev dependencies
pytest                               # Run SDK tests
ruff check .                         # Lint Python code
```

## Architecture

### Core Components

1. **Policy Engine** (`src/lib/policy-engine.ts`)
   - Evaluates agent actions against organization policies stored in database
   - Returns decisions: `allow`, `block`, or `require_approval`
   - Rules support regex target patterns and parameter conditions (eq, gt, lt, contains, matches)

2. **Ingest API** (`src/app/api/v1/ingest/route.ts`)
   - Main SDK endpoint: `POST /api/v1/ingest`
   - Validates API key, evaluates action, logs to database, creates approval requests if needed
   - Action types: `tool_call`, `api_call`, `db_query`, `file_access`

3. **Authentication**
   - API Keys (`src/lib/auth/api-key.ts`): Format `aasp_{live|test}_{random}`, SHA-256 hashed for storage
   - Sessions (`src/lib/auth/session.ts`): JWT-based with httpOnly cookies for dashboard

### Data Model (Prisma/SQLite)

- **Organization** - Multi-tenant root entity
- **User** - Organization members with roles (admin/member)
- **ApiKey** - SDK authentication (only hash stored, prefix shown)
- **Policy** - Security rules with JSON rules array
- **Action** - Audit log of all evaluated actions
- **ApprovalRequest** - Human-in-the-loop approval workflow

### Frontend Structure

- **Landing page** (`src/app/page.tsx`): Tabbed sections (Business, Technical, MVP, Demo)
- **Dashboard** (`src/app/(dashboard)/`): Authenticated area for policies, approvals, actions, API keys
- **Demo** (`src/components/sections/demo/`): Interactive demo with in-memory state (`src/lib/demo/`)
- **Docs** (`src/app/docs/page.tsx`): API and SDK documentation

### Python SDK (`sdk/python/`)

- `AASPCallback`: LangChain callback handler for automatic tool interception
- `AASPClient`: Direct HTTP client for custom integrations
- Handles blocking, approval waiting, and result logging

## Key Patterns

- Path alias: `@/*` maps to `src/*`
- UI components use shadcn/ui patterns with Radix primitives
- Demo mode uses separate in-memory store (`src/lib/demo/store.ts`) vs production database
- Animations via `motion/react` (Framer Motion)
- Form validation with `zod` and `react-hook-form`

## Environment Variables

```bash
DATABASE_URL="file:./dev.db"     # SQLite database path
JWT_SECRET="your-secret"         # Session JWT signing key
```
