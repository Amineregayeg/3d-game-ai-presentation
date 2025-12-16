/**
 * API client for Flask backend
 */

// Use empty string for relative URLs (through nginx proxy), or explicit URL for direct access
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '';

// Vault token storage
let vaultToken: string | null = null;

export function setVaultToken(token: string) {
  vaultToken = token;
  if (typeof window !== 'undefined') {
    sessionStorage.setItem('vault_token', token);
  }
}

export function getVaultToken(): string | null {
  if (vaultToken) return vaultToken;
  if (typeof window !== 'undefined') {
    return sessionStorage.getItem('vault_token');
  }
  return null;
}

export function clearVaultToken() {
  vaultToken = null;
  if (typeof window !== 'undefined') {
    sessionStorage.removeItem('vault_token');
  }
}

async function fetchAPI<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const token = getVaultToken();
  if (token && endpoint.includes('/vault/')) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.error || 'Request failed');
  }

  return response.json();
}

// ============== Vault API ==============

export interface Secret {
  id: number;
  name: string;
  category: string;
  value: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export async function authenticateVault(password: string): Promise<{ token: string; expires_in: number }> {
  const result = await fetchAPI<{ token: string; expires_in: number }>('/api/vault/auth', {
    method: 'POST',
    body: JSON.stringify({ password }),
  });
  setVaultToken(result.token);
  return result;
}

export async function getSecrets(): Promise<Secret[]> {
  return fetchAPI<Secret[]>('/api/vault/secrets');
}

export async function revealSecret(id: number): Promise<{ value: string }> {
  return fetchAPI<{ value: string }>(`/api/vault/secrets/${id}/reveal`);
}

export async function createSecret(data: Omit<Secret, 'id' | 'created_at' | 'updated_at'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/vault/secrets', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateSecret(id: number, data: Partial<Secret>): Promise<void> {
  await fetchAPI(`/api/vault/secrets/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteSecret(id: number): Promise<void> {
  await fetchAPI(`/api/vault/secrets/${id}`, { method: 'DELETE' });
}

// ============== Tasks API ==============

export interface Task {
  id: string;
  title: string;
  description: string;
  component: string;
  phase: string;
  status: 'todo' | 'in_progress' | 'done';
  priority: 'high' | 'medium' | 'low';
  assignee?: string;
  notes?: string;
  due_date?: string;
  time_spent: number;
  created_at: string;
  updated_at: string;
}

export async function getTasks(filters?: { component?: string; status?: string }): Promise<Task[]> {
  const params = new URLSearchParams();
  if (filters?.component) params.append('component', filters.component);
  if (filters?.status) params.append('status', filters.status);
  const query = params.toString();
  return fetchAPI<Task[]>(`/api/tasks${query ? `?${query}` : ''}`);
}

export async function updateTask(id: string, data: Partial<Task>): Promise<Task> {
  return fetchAPI<Task>(`/api/tasks/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function createTask(data: Omit<Task, 'created_at' | 'updated_at' | 'time_spent'>): Promise<Task> {
  return fetchAPI<Task>('/api/tasks', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============== Team API ==============

export interface TeamMember {
  id: number;
  name: string;
  role: string;
  email?: string;
  github?: string;
  avatar_url?: string;
  components: string[];
  status: 'active' | 'away' | 'offline';
  bio?: string;
  created_at: string;
}

export async function getTeam(): Promise<TeamMember[]> {
  return fetchAPI<TeamMember[]>('/api/team');
}

export async function createTeamMember(data: Omit<TeamMember, 'id' | 'created_at'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/team', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateTeamMember(id: number, data: Partial<TeamMember>): Promise<void> {
  await fetchAPI(`/api/team/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteTeamMember(id: number): Promise<void> {
  await fetchAPI(`/api/team/${id}`, { method: 'DELETE' });
}

// ============== Activity API ==============

export interface Activity {
  id: number;
  type: string;
  title: string;
  description?: string;
  component?: string;
  user?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export async function getActivity(filters?: { limit?: number; component?: string; type?: string }): Promise<Activity[]> {
  const params = new URLSearchParams();
  if (filters?.limit) params.append('limit', filters.limit.toString());
  if (filters?.component) params.append('component', filters.component);
  if (filters?.type) params.append('type', filters.type);
  const query = params.toString();
  return fetchAPI<Activity[]>(`/api/activity${query ? `?${query}` : ''}`);
}

export async function createActivity(data: Omit<Activity, 'id' | 'created_at'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/activity', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============== Milestones API ==============

export interface Milestone {
  id: number;
  title: string;
  description?: string;
  component?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'delayed';
  target_date?: string;
  completed_date?: string;
  progress: number;
  created_at: string;
}

export async function getMilestones(component?: string): Promise<Milestone[]> {
  const query = component ? `?component=${component}` : '';
  return fetchAPI<Milestone[]>(`/api/milestones${query}`);
}

export async function createMilestone(data: Omit<Milestone, 'id' | 'created_at' | 'completed_date'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/milestones', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateMilestone(id: number, data: Partial<Milestone>): Promise<void> {
  await fetchAPI(`/api/milestones/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteMilestone(id: number): Promise<void> {
  await fetchAPI(`/api/milestones/${id}`, { method: 'DELETE' });
}

// ============== Decisions API ==============

export interface Decision {
  id: number;
  title: string;
  status: 'proposed' | 'accepted' | 'rejected' | 'superseded';
  context?: string;
  decision?: string;
  consequences?: string;
  component?: string;
  author?: string;
  created_at: string;
  updated_at: string;
}

export async function getDecisions(filters?: { component?: string; status?: string }): Promise<Decision[]> {
  const params = new URLSearchParams();
  if (filters?.component) params.append('component', filters.component);
  if (filters?.status) params.append('status', filters.status);
  const query = params.toString();
  return fetchAPI<Decision[]>(`/api/decisions${query ? `?${query}` : ''}`);
}

export async function createDecision(data: Omit<Decision, 'id' | 'created_at' | 'updated_at'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/decisions', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateDecision(id: number, data: Partial<Decision>): Promise<void> {
  await fetchAPI(`/api/decisions/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteDecision(id: number): Promise<void> {
  await fetchAPI(`/api/decisions/${id}`, { method: 'DELETE' });
}

// ============== Resources API ==============

export interface Resource {
  id: number;
  title: string;
  url: string;
  category: 'paper' | 'tutorial' | 'tool' | 'library' | 'docs';
  description?: string;
  component?: string;
  tags: string[];
  created_at: string;
}

export async function getResources(filters?: { category?: string; component?: string }): Promise<Resource[]> {
  const params = new URLSearchParams();
  if (filters?.category) params.append('category', filters.category);
  if (filters?.component) params.append('component', filters.component);
  const query = params.toString();
  return fetchAPI<Resource[]>(`/api/resources${query ? `?${query}` : ''}`);
}

export async function createResource(data: Omit<Resource, 'id' | 'created_at'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/resources', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function deleteResource(id: number): Promise<void> {
  await fetchAPI(`/api/resources/${id}`, { method: 'DELETE' });
}

// ============== Changelog API ==============

export interface ChangelogEntry {
  id: number;
  version: string;
  title: string;
  description?: string;
  changes: string[];
  component?: string;
  release_date: string;
  author?: string;
}

export async function getChangelog(component?: string): Promise<ChangelogEntry[]> {
  const query = component ? `?component=${component}` : '';
  return fetchAPI<ChangelogEntry[]>(`/api/changelog${query}`);
}

export async function createChangelogEntry(data: Omit<ChangelogEntry, 'id'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/changelog', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============== Glossary API ==============

export interface GlossaryTerm {
  id: number;
  term: string;
  definition: string;
  category?: string;
  related_terms: string[];
  component?: string;
  created_at: string;
}

export async function getGlossary(filters?: { category?: string; component?: string }): Promise<GlossaryTerm[]> {
  const params = new URLSearchParams();
  if (filters?.category) params.append('category', filters.category);
  if (filters?.component) params.append('component', filters.component);
  const query = params.toString();
  return fetchAPI<GlossaryTerm[]>(`/api/glossary${query ? `?${query}` : ''}`);
}

export async function createGlossaryTerm(data: Omit<GlossaryTerm, 'id' | 'created_at'>): Promise<{ id: number }> {
  return fetchAPI<{ id: number }>('/api/glossary', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function deleteGlossaryTerm(id: number): Promise<void> {
  await fetchAPI(`/api/glossary/${id}`, { method: 'DELETE' });
}

// ============== Context API ==============

export interface ProjectContext {
  project: {
    name: string;
    description: string;
    deadline: string;
    total_tasks: number;
    completed_tasks: number;
  };
  components: Record<string, {
    name: string;
    description: string;
    technologies: string[];
    doc_path: string;
    presentation: string;
    timeline: string;
    progress: number;
    tasks: {
      total: number;
      completed: number;
      in_progress: number;
    };
  }>;
  milestones: Array<{
    id: number;
    title: string;
    component: string;
    status: string;
    target_date: string;
    progress: number;
  }>;
  decisions: Array<{
    id: number;
    title: string;
    component: string;
    decision: string;
  }>;
}

export async function getProjectContext(): Promise<ProjectContext> {
  return fetchAPI<ProjectContext>('/api/context');
}

// ============== Health API ==============

export async function checkHealth(): Promise<{ status: string; timestamp: string; database: string }> {
  return fetchAPI('/api/health');
}
