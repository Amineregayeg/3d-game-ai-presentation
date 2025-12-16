"use client";

import { useState, useEffect } from "react";
import PageLayout from "@/components/page-layout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { getTeam, createTeamMember, updateTeamMember, deleteTeamMember, type TeamMember } from "@/lib/api";

const componentOptions = [
  { id: "stt", label: "VoxFormer STT", color: "cyan" },
  { id: "rag", label: "Advanced RAG", color: "emerald" },
  { id: "tts-lipsync", label: "TTS + LipSync", color: "rose" },
  { id: "mcp", label: "Blender MCP", color: "orange" },
];

const statusColors: Record<string, { bg: string; text: string; dot: string }> = {
  active: { bg: "bg-emerald-500/20", text: "text-emerald-400", dot: "bg-emerald-500" },
  away: { bg: "bg-amber-500/20", text: "text-amber-400", dot: "bg-amber-500" },
  offline: { bg: "bg-slate-500/20", text: "text-slate-400", dot: "bg-slate-500" },
};

export default function TeamPage() {
  const [team, setTeam] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newMember, setNewMember] = useState<{
    name: string;
    role: string;
    email: string;
    github: string;
    avatar_url: string;
    components: string[];
    status: "active" | "away" | "offline";
    bio: string;
  }>({
    name: "",
    role: "",
    email: "",
    github: "",
    avatar_url: "",
    components: [],
    status: "active",
    bio: "",
  });

  useEffect(() => {
    loadTeam();
  }, []);

  const loadTeam = async () => {
    try {
      const data = await getTeam();
      setTeam(data);
    } catch (error) {
      console.error("Failed to load team:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddMember = async () => {
    try {
      await createTeamMember(newMember);
      setAddDialogOpen(false);
      setNewMember({
        name: "",
        role: "",
        email: "",
        github: "",
        avatar_url: "",
        components: [],
        status: "active",
        bio: "",
      });
      await loadTeam();
    } catch (error) {
      console.error("Failed to create team member:", error);
    }
  };

  const handleUpdateStatus = async (id: number, status: TeamMember["status"]) => {
    try {
      await updateTeamMember(id, { status });
      await loadTeam();
    } catch (error) {
      console.error("Failed to update status:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to remove this team member?")) return;
    try {
      await deleteTeamMember(id);
      await loadTeam();
    } catch (error) {
      console.error("Failed to delete team member:", error);
    }
  };

  const toggleComponent = (componentId: string) => {
    setNewMember((prev) => ({
      ...prev,
      components: prev.components.includes(componentId)
        ? prev.components.filter((c) => c !== componentId)
        : [...prev.components, componentId],
    }));
  };

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <PageLayout
      title="Team Directory"
      description="View team members and their responsibilities"
      badges={[{ label: `${team.length} Members` }]}
      gradientOrbs={[
        { color: "blue", position: "top-1/4 -left-32" },
        { color: "indigo", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Actions */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          {Object.entries(statusColors).map(([status, colors]) => (
            <div key={status} className="flex items-center gap-2 text-sm text-slate-400">
              <div className={`w-2 h-2 rounded-full ${colors.dot}`} />
              <span className="capitalize">{status}</span>
              <span className="text-slate-500">({team.filter((m) => m.status === status).length})</span>
            </div>
          ))}
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
              </svg>
              Add Team Member
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700 max-w-lg">
            <DialogHeader>
              <DialogTitle className="text-white">Add Team Member</DialogTitle>
              <DialogDescription className="text-slate-400">
                Add a new member to the project team
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4 max-h-[60vh] overflow-y-auto">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Name</Label>
                  <Input
                    value={newMember.name}
                    onChange={(e) => setNewMember((p) => ({ ...p, name: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="John Doe"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Role</Label>
                  <Input
                    value={newMember.role}
                    onChange={(e) => setNewMember((p) => ({ ...p, role: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="ML Engineer"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Email</Label>
                  <Input
                    value={newMember.email}
                    onChange={(e) => setNewMember((p) => ({ ...p, email: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="john@example.com"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">GitHub</Label>
                  <Input
                    value={newMember.github}
                    onChange={(e) => setNewMember((p) => ({ ...p, github: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="johndoe"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Avatar URL (optional)</Label>
                <Input
                  value={newMember.avatar_url}
                  onChange={(e) => setNewMember((p) => ({ ...p, avatar_url: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="https://..."
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Components</Label>
                <div className="grid grid-cols-2 gap-2">
                  {componentOptions.map((comp) => (
                    <div
                      key={comp.id}
                      className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/50 border border-slate-700"
                    >
                      <Checkbox
                        checked={newMember.components.includes(comp.id)}
                        onCheckedChange={() => toggleComponent(comp.id)}
                      />
                      <span className="text-sm text-slate-300">{comp.label}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Status</Label>
                <Select
                  value={newMember.status}
                  onValueChange={(v: TeamMember["status"]) => setNewMember((p) => ({ ...p, status: v }))}
                >
                  <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700">
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="away">Away</SelectItem>
                    <SelectItem value="offline">Offline</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Bio (optional)</Label>
                <Textarea
                  value={newMember.bio}
                  onChange={(e) => setNewMember((p) => ({ ...p, bio: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="Brief description..."
                  rows={2}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                Cancel
              </Button>
              <Button onClick={handleAddMember} className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white">
                Add Member
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Team Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="bg-slate-900/50 border-slate-800 animate-pulse">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-slate-700" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-slate-700 rounded w-3/4" />
                    <div className="h-3 bg-slate-700 rounded w-1/2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {team.map((member) => (
            <Card key={member.id} className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-all group">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <Avatar className="w-16 h-16 border-2 border-slate-700">
                        <AvatarImage src={member.avatar_url} />
                        <AvatarFallback className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white text-lg">
                          {getInitials(member.name)}
                        </AvatarFallback>
                      </Avatar>
                      <div className={`absolute bottom-0 right-0 w-4 h-4 rounded-full ${statusColors[member.status].dot} border-2 border-slate-900`} />
                    </div>
                    <div>
                      <CardTitle className="text-white text-lg">{member.name}</CardTitle>
                      <CardDescription className="text-slate-400">{member.role}</CardDescription>
                    </div>
                  </div>
                  <Select
                    value={member.status}
                    onValueChange={(v: TeamMember["status"]) => handleUpdateStatus(member.id, v)}
                  >
                    <SelectTrigger className="w-24 h-7 text-xs bg-transparent border-0 hover:bg-slate-800">
                      <Badge className={`${statusColors[member.status].bg} ${statusColors[member.status].text}`}>
                        {member.status}
                      </Badge>
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="away">Away</SelectItem>
                      <SelectItem value="offline">Offline</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardHeader>
              <CardContent>
                {member.bio && (
                  <p className="text-sm text-slate-400 mb-4">{member.bio}</p>
                )}
                <div className="flex flex-wrap gap-2 mb-4">
                  {member.components.map((compId) => {
                    const comp = componentOptions.find((c) => c.id === compId);
                    return comp ? (
                      <Badge key={compId} variant="secondary" className={`bg-${comp.color}-500/20 text-${comp.color}-400 text-xs`}>
                        {comp.label}
                      </Badge>
                    ) : null;
                  })}
                </div>
                <div className="flex items-center justify-between pt-4 border-t border-slate-800">
                  <div className="flex items-center gap-3">
                    {member.email && (
                      <a href={`mailto:${member.email}`} className="text-slate-400 hover:text-white transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                      </a>
                    )}
                    {member.github && (
                      <a href={`https://github.com/${member.github}`} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                          <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                        </svg>
                      </a>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(member.id)}
                    className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
          {team.length === 0 && (
            <Card className="col-span-full bg-slate-900/50 border-slate-800">
              <CardContent className="p-12 text-center">
                <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-white mb-2">No team members yet</h3>
                <p className="text-slate-400 mb-4">Add your first team member to get started</p>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </PageLayout>
  );
}
