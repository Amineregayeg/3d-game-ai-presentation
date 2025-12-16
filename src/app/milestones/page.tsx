"use client";

import { useState, useEffect } from "react";
import PageLayout from "@/components/page-layout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
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
import { getMilestones, createMilestone, updateMilestone, deleteMilestone, type Milestone } from "@/lib/api";
import { format, isPast, isFuture, differenceInDays } from "date-fns";

const statusColors: Record<string, { bg: string; text: string; border: string }> = {
  pending: { bg: "bg-slate-500/20", text: "text-slate-400", border: "border-slate-500/30" },
  in_progress: { bg: "bg-blue-500/20", text: "text-blue-400", border: "border-blue-500/30" },
  completed: { bg: "bg-emerald-500/20", text: "text-emerald-400", border: "border-emerald-500/30" },
  delayed: { bg: "bg-red-500/20", text: "text-red-400", border: "border-red-500/30" },
};

const componentColors: Record<string, { bg: string; text: string }> = {
  stt: { bg: "bg-cyan-500/20", text: "text-cyan-400" },
  rag: { bg: "bg-emerald-500/20", text: "text-emerald-400" },
  "tts-lipsync": { bg: "bg-rose-500/20", text: "text-rose-400" },
  mcp: { bg: "bg-orange-500/20", text: "text-orange-400" },
};

export default function MilestonesPage() {
  const [milestones, setMilestones] = useState<Milestone[]>([]);
  const [loading, setLoading] = useState(true);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newMilestone, setNewMilestone] = useState({
    title: "",
    description: "",
    component: "",
    status: "pending" as Milestone["status"],
    target_date: "",
    progress: 0,
  });

  useEffect(() => {
    loadMilestones();
  }, []);

  const loadMilestones = async () => {
    try {
      const data = await getMilestones();
      setMilestones(data);
    } catch (error) {
      console.error("Failed to load milestones:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddMilestone = async () => {
    try {
      await createMilestone(newMilestone);
      setAddDialogOpen(false);
      setNewMilestone({
        title: "",
        description: "",
        component: "",
        status: "pending",
        target_date: "",
        progress: 0,
      });
      await loadMilestones();
    } catch (error) {
      console.error("Failed to create milestone:", error);
    }
  };

  const handleUpdateStatus = async (id: number, status: Milestone["status"]) => {
    try {
      await updateMilestone(id, { status });
      await loadMilestones();
    } catch (error) {
      console.error("Failed to update milestone:", error);
    }
  };

  const handleUpdateProgress = async (id: number, progress: number) => {
    try {
      await updateMilestone(id, { progress });
      await loadMilestones();
    } catch (error) {
      console.error("Failed to update milestone:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this milestone?")) return;
    try {
      await deleteMilestone(id);
      await loadMilestones();
    } catch (error) {
      console.error("Failed to delete milestone:", error);
    }
  };

  const getTimeStatus = (targetDate: string | null | undefined) => {
    if (!targetDate) return null;
    const date = new Date(targetDate);
    const days = differenceInDays(date, new Date());

    if (isPast(date)) {
      return { label: `${Math.abs(days)} days overdue`, color: "text-red-400" };
    } else if (days <= 7) {
      return { label: `${days} days left`, color: "text-amber-400" };
    } else {
      return { label: `${days} days left`, color: "text-slate-400" };
    }
  };

  const completedCount = milestones.filter((m) => m.status === "completed").length;
  const inProgressCount = milestones.filter((m) => m.status === "in_progress").length;

  return (
    <PageLayout
      title="Project Milestones"
      description="Track major project deliverables and deadlines"
      badges={[
        { label: `${completedCount}/${milestones.length} Complete` },
        { label: `${inProgressCount} In Progress` },
      ]}
      gradientOrbs={[
        { color: "purple", position: "top-1/4 -left-32" },
        { color: "pink", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {Object.entries(statusColors).map(([status, colors]) => {
          const count = milestones.filter((m) => m.status === status).length;
          return (
            <Card key={status} className={`bg-slate-900/50 border-slate-800`}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-white">{count}</p>
                    <p className={`text-sm capitalize ${colors.text}`}>{status.replace("_", " ")}</p>
                  </div>
                  <div className={`w-10 h-10 rounded-full ${colors.bg} flex items-center justify-center`}>
                    <div className={`w-3 h-3 rounded-full ${colors.text.replace("text-", "bg-")}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Add Milestone */}
      <div className="flex justify-end mb-6">
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-purple-500 to-pink-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Milestone
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700">
            <DialogHeader>
              <DialogTitle className="text-white">Add Milestone</DialogTitle>
              <DialogDescription className="text-slate-400">
                Create a new project milestone
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-slate-300">Title</Label>
                <Input
                  value={newMilestone.title}
                  onChange={(e) => setNewMilestone((p) => ({ ...p, title: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="e.g., STT Audio Frontend Complete"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Description</Label>
                <Textarea
                  value={newMilestone.description}
                  onChange={(e) => setNewMilestone((p) => ({ ...p, description: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="What does this milestone include?"
                  rows={2}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Component</Label>
                  <Select
                    value={newMilestone.component}
                    onValueChange={(v) => setNewMilestone((p) => ({ ...p, component: v }))}
                  >
                    <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                      <SelectValue placeholder="Select component" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="">All / General</SelectItem>
                      <SelectItem value="stt">VoxFormer STT</SelectItem>
                      <SelectItem value="rag">Advanced RAG</SelectItem>
                      <SelectItem value="tts-lipsync">TTS + LipSync</SelectItem>
                      <SelectItem value="mcp">Blender MCP</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Target Date</Label>
                  <Input
                    type="date"
                    value={newMilestone.target_date}
                    onChange={(e) => setNewMilestone((p) => ({ ...p, target_date: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                Cancel
              </Button>
              <Button onClick={handleAddMilestone} className="bg-gradient-to-r from-purple-500 to-pink-600 text-white">
                Add Milestone
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Timeline */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Milestone Timeline</CardTitle>
          <CardDescription className="text-slate-400">
            Project milestones ordered by target date
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse flex gap-4">
                  <div className="w-4 h-4 rounded-full bg-slate-700" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-slate-700 rounded w-1/3" />
                    <div className="h-3 bg-slate-700 rounded w-2/3" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-6 relative">
              <div className="absolute left-2 top-0 bottom-0 w-0.5 bg-slate-800" />
              {milestones.map((milestone) => {
                const timeStatus = getTimeStatus(milestone.target_date);
                const colors = statusColors[milestone.status];
                return (
                  <div key={milestone.id} className="flex gap-4 relative group">
                    <div className={`w-4 h-4 rounded-full ${colors.bg} border-2 ${colors.border} z-10 mt-1 shrink-0`} />
                    <div className="flex-1 p-4 bg-slate-800/30 rounded-lg hover:bg-slate-800/50 transition-colors">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="text-white font-medium">{milestone.title}</h3>
                          {milestone.description && (
                            <p className="text-sm text-slate-400 mt-1">{milestone.description}</p>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Select
                            value={milestone.status}
                            onValueChange={(v: Milestone["status"]) => handleUpdateStatus(milestone.id, v)}
                          >
                            <SelectTrigger className="w-32 h-8 text-xs bg-transparent border-0">
                              <Badge className={`${colors.bg} ${colors.text} ${colors.border}`}>
                                {milestone.status.replace("_", " ")}
                              </Badge>
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-slate-700">
                              <SelectItem value="pending">Pending</SelectItem>
                              <SelectItem value="in_progress">In Progress</SelectItem>
                              <SelectItem value="completed">Completed</SelectItem>
                              <SelectItem value="delayed">Delayed</SelectItem>
                            </SelectContent>
                          </Select>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(milestone.id)}
                            className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </Button>
                        </div>
                      </div>
                      <div className="flex items-center gap-4 mb-3">
                        {milestone.component && (
                          <Badge className={`${componentColors[milestone.component]?.bg || "bg-slate-500/20"} ${componentColors[milestone.component]?.text || "text-slate-400"}`}>
                            {milestone.component === "tts-lipsync" ? "TTS" : (milestone.component?.toUpperCase() || "General")}
                          </Badge>
                        )}
                        {milestone.target_date && (
                          <span className="text-xs text-slate-500">
                            Target: {format(new Date(milestone.target_date), "MMM d, yyyy")}
                          </span>
                        )}
                        {timeStatus && milestone.status !== "completed" && (
                          <span className={`text-xs ${timeStatus.color}`}>{timeStatus.label}</span>
                        )}
                        {milestone.completed_date && (
                          <span className="text-xs text-emerald-400">
                            Completed: {format(new Date(milestone.completed_date), "MMM d, yyyy")}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="flex-1">
                          <Progress value={milestone.progress} className="h-2" />
                        </div>
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            min={0}
                            max={100}
                            value={milestone.progress}
                            onChange={(e) => handleUpdateProgress(milestone.id, parseInt(e.target.value) || 0)}
                            className="w-16 h-7 text-xs bg-slate-800 border-slate-700 text-white text-center"
                          />
                          <span className="text-xs text-slate-500">%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
              {milestones.length === 0 && (
                <div className="text-center py-12 relative z-10">
                  <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-white mb-2">No milestones yet</h3>
                  <p className="text-slate-400">Add your first milestone to track project progress</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </PageLayout>
  );
}
