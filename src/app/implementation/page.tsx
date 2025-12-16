"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Calendar } from "@/components/ui/calendar";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { getTasks, updateTask, createTask, type Task } from "@/lib/api";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

type TaskStatus = "todo" | "in_progress" | "done";
type TaskPriority = "high" | "medium" | "low";

const componentColors: Record<string, { bg: string; text: string; border: string }> = {
  stt: { bg: "bg-cyan-500/20", text: "text-cyan-400", border: "border-cyan-500/30" },
  rag: { bg: "bg-emerald-500/20", text: "text-emerald-400", border: "border-emerald-500/30" },
  "tts-lipsync": { bg: "bg-rose-500/20", text: "text-rose-400", border: "border-rose-500/30" },
  mcp: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
};

const priorityColors: Record<TaskPriority, { bg: string; text: string }> = {
  high: { bg: "bg-red-500/20", text: "text-red-400" },
  medium: { bg: "bg-amber-500/20", text: "text-amber-400" },
  low: { bg: "bg-slate-500/20", text: "text-slate-400" },
};

const statusColors: Record<TaskStatus, { bg: string; text: string }> = {
  todo: { bg: "bg-slate-500/20", text: "text-slate-400" },
  in_progress: { bg: "bg-blue-500/20", text: "text-blue-400" },
  done: { bg: "bg-emerald-500/20", text: "text-emerald-400" },
};

export default function ImplementationPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedComponent, setSelectedComponent] = useState<string>("all");
  const [selectedStatus, setSelectedStatus] = useState<string>("all");
  const [date, setDate] = useState<Date | undefined>(new Date());
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [editingTask, setEditingTask] = useState<Task | null>(null);
  const [newTask, setNewTask] = useState({
    id: "",
    title: "",
    description: "",
    component: "stt",
    phase: "",
    status: "todo" as TaskStatus,
    priority: "medium" as TaskPriority,
    assignee: "",
    notes: "",
  });

  const deadline = new Date(2025, 11, 18); // December 18, 2025

  const loadTasks = useCallback(async () => {
    try {
      const data = await getTasks();
      setTasks(data);
    } catch (error) {
      console.error("Failed to load tasks:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTasks();
  }, [loadTasks]);

  const filteredTasks = tasks.filter((task) => {
    if (selectedComponent !== "all" && task.component !== selectedComponent) return false;
    if (selectedStatus !== "all" && task.status !== selectedStatus) return false;
    return true;
  });

  const tasksByStatus = {
    todo: filteredTasks.filter((t) => t.status === "todo"),
    in_progress: filteredTasks.filter((t) => t.status === "in_progress"),
    done: filteredTasks.filter((t) => t.status === "done"),
  };

  const totalTasks = tasks.length;
  const completedTasks = tasks.filter((t) => t.status === "done").length;
  const progressPercent = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;

  const toggleTaskStatus = async (taskId: string) => {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;

    const newStatus: TaskStatus =
      task.status === "todo" ? "in_progress" : task.status === "in_progress" ? "done" : "todo";

    try {
      await updateTask(taskId, { status: newStatus });
      setTasks((prev) =>
        prev.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t))
      );
    } catch (error) {
      console.error("Failed to update task:", error);
    }
  };

  const handleAddTask = async () => {
    try {
      const taskId = `${newTask.component}-${Date.now()}`;
      await createTask({ ...newTask, id: taskId });
      setAddDialogOpen(false);
      setNewTask({
        id: "",
        title: "",
        description: "",
        component: "stt",
        phase: "",
        status: "todo",
        priority: "medium",
        assignee: "",
        notes: "",
      });
      await loadTasks();
    } catch (error) {
      console.error("Failed to create task:", error);
    }
  };

  const handleUpdateTask = async () => {
    if (!editingTask) return;
    try {
      await updateTask(editingTask.id, editingTask);
      setEditingTask(null);
      await loadTasks();
    } catch (error) {
      console.error("Failed to update task:", error);
    }
  };

  return (
    <div className="dark min-h-screen bg-slate-950">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute top-3/4 left-1/3 w-64 h-64 bg-red-500/10 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-xl sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back
              </Link>
              <div className="w-px h-6 bg-slate-700" />
              <h1 className="text-xl font-semibold text-white">Implementation Tracker</h1>
              {loading && (
                <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
              )}
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300">
                {completedTasks}/{totalTasks} Tasks
              </Badge>
              <Badge className="bg-red-500/20 text-red-400 border border-red-500/30">
                Deadline: Dec 18, 2025
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Progress Overview */}
        <Card className="bg-slate-900/50 border-slate-800 mb-8">
          <CardHeader className="pb-2">
            <CardTitle className="text-white">Overall Progress</CardTitle>
            <CardDescription className="text-slate-400">
              Track implementation progress across all components (synced with backend)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between text-sm text-slate-400 mb-2">
              <span>{completedTasks} of {totalTasks} tasks completed</span>
              <span>{progressPercent}%</span>
            </div>
            <Progress value={progressPercent} className="h-3" />
            <div className="grid grid-cols-4 gap-4 mt-4">
              {[
                { label: "VoxFormer STT", component: "stt", color: "cyan" },
                { label: "Advanced RAG", component: "rag", color: "emerald" },
                { label: "TTS + LipSync", component: "tts-lipsync", color: "rose" },
                { label: "Blender MCP", component: "mcp", color: "orange" },
              ].map((item) => {
                const componentTasks = tasks.filter((t) => t.component === item.component);
                const componentDone = componentTasks.filter((t) => t.status === "done").length;
                const percent = componentTasks.length > 0 ? Math.round((componentDone / componentTasks.length) * 100) : 0;
                return (
                  <div key={item.component} className="text-center">
                    <p className={`text-xs text-${item.color}-400 mb-1`}>{item.label}</p>
                    <p className="text-lg font-bold text-white">{percent}%</p>
                    <p className="text-xs text-slate-500">{componentDone}/{componentTasks.length}</p>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Calendar Sidebar */}
          <div className="lg:col-span-1">
            <Card className="bg-slate-900/50 border-slate-800 sticky top-24">
              <CardHeader className="pb-2">
                <CardTitle className="text-white text-lg">Timeline</CardTitle>
              </CardHeader>
              <CardContent>
                <Calendar
                  mode="single"
                  selected={date}
                  onSelect={setDate}
                  className="rounded-md"
                  modifiers={{
                    deadline: [deadline],
                  }}
                  modifiersClassNames={{
                    deadline: "bg-red-500 text-white hover:bg-red-600 font-bold",
                  }}
                />
                <Separator className="my-4 bg-slate-800" />
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <span className="text-sm text-slate-400">Deadline: Dec 18, 2025</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald-500" />
                    <span className="text-sm text-slate-400">Today</span>
                  </div>
                </div>
                <Separator className="my-4 bg-slate-800" />
                <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
                  <DialogTrigger asChild>
                    <Button className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 text-white hover:opacity-90">
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                      </svg>
                      Add Task
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="bg-slate-900 border-slate-700">
                    <DialogHeader>
                      <DialogTitle className="text-white">Add New Task</DialogTitle>
                      <DialogDescription className="text-slate-400">
                        Create a new task for the project
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <div className="space-y-2">
                        <Label className="text-slate-300">Title</Label>
                        <Input
                          value={newTask.title}
                          onChange={(e) => setNewTask((p) => ({ ...p, title: e.target.value }))}
                          className="bg-slate-800 border-slate-700 text-white"
                          placeholder="Task title"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-slate-300">Description</Label>
                        <Textarea
                          value={newTask.description}
                          onChange={(e) => setNewTask((p) => ({ ...p, description: e.target.value }))}
                          className="bg-slate-800 border-slate-700 text-white"
                          placeholder="Task description"
                          rows={2}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label className="text-slate-300">Component</Label>
                          <Select
                            value={newTask.component}
                            onValueChange={(v) => setNewTask((p) => ({ ...p, component: v }))}
                          >
                            <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-slate-700">
                              <SelectItem value="stt">VoxFormer STT</SelectItem>
                              <SelectItem value="rag">Advanced RAG</SelectItem>
                              <SelectItem value="tts-lipsync">TTS + LipSync</SelectItem>
                              <SelectItem value="mcp">Blender MCP</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label className="text-slate-300">Priority</Label>
                          <Select
                            value={newTask.priority}
                            onValueChange={(v: TaskPriority) => setNewTask((p) => ({ ...p, priority: v }))}
                          >
                            <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-slate-700">
                              <SelectItem value="high">High</SelectItem>
                              <SelectItem value="medium">Medium</SelectItem>
                              <SelectItem value="low">Low</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label className="text-slate-300">Phase</Label>
                        <Input
                          value={newTask.phase}
                          onChange={(e) => setNewTask((p) => ({ ...p, phase: e.target.value }))}
                          className="bg-slate-800 border-slate-700 text-white"
                          placeholder="e.g., Phase 1: Foundation"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-slate-300">Assignee (optional)</Label>
                        <Input
                          value={newTask.assignee}
                          onChange={(e) => setNewTask((p) => ({ ...p, assignee: e.target.value }))}
                          className="bg-slate-800 border-slate-700 text-white"
                          placeholder="Assignee name"
                        />
                      </div>
                    </div>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                        Cancel
                      </Button>
                      <Button onClick={handleAddTask} className="bg-gradient-to-r from-cyan-500 to-purple-600 text-white">
                        Add Task
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </CardContent>
            </Card>
          </div>

          {/* Kanban Board */}
          <div className="lg:col-span-3">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-white">Backlog</CardTitle>
                    <CardDescription className="text-slate-400">
                      Task management board (click task to cycle status)
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Select value={selectedComponent} onValueChange={setSelectedComponent}>
                      <SelectTrigger className="w-40 bg-slate-800 border-slate-700 text-slate-300">
                        <SelectValue placeholder="Component" />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-slate-700">
                        <SelectItem value="all">All Components</SelectItem>
                        <SelectItem value="stt">VoxFormer STT</SelectItem>
                        <SelectItem value="rag">Advanced RAG</SelectItem>
                        <SelectItem value="tts-lipsync">TTS + LipSync</SelectItem>
                        <SelectItem value="mcp">Blender MCP</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={selectedStatus} onValueChange={setSelectedStatus}>
                      <SelectTrigger className="w-32 bg-slate-800 border-slate-700 text-slate-300">
                        <SelectValue placeholder="Status" />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-slate-700">
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="todo">To Do</SelectItem>
                        <SelectItem value="in_progress">In Progress</SelectItem>
                        <SelectItem value="done">Done</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="kanban" className="w-full">
                  <TabsList className="bg-slate-800/50 mb-4">
                    <TabsTrigger value="kanban" className="data-[state=active]:bg-slate-700">Kanban</TabsTrigger>
                    <TabsTrigger value="list" className="data-[state=active]:bg-slate-700">List</TabsTrigger>
                  </TabsList>

                  <TabsContent value="kanban">
                    <div className="grid grid-cols-3 gap-4">
                      {(["todo", "in_progress", "done"] as TaskStatus[]).map((status) => (
                        <div key={status} className="space-y-3">
                          <div className="flex items-center justify-between">
                            <Badge className={`${statusColors[status].bg} ${statusColors[status].text}`}>
                              {status === "todo" ? "To Do" : status === "in_progress" ? "In Progress" : "Done"}
                            </Badge>
                            <span className="text-xs text-slate-500">{tasksByStatus[status].length}</span>
                          </div>
                          <ScrollArea className="h-[500px] pr-2">
                            <div className="space-y-2">
                              {tasksByStatus[status].map((task) => (
                                <div
                                  key={task.id}
                                  className="p-3 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-slate-600 transition-all cursor-pointer group"
                                  onClick={() => toggleTaskStatus(task.id)}
                                >
                                  <div className="flex items-start gap-2 mb-2">
                                    <Checkbox
                                      checked={task.status === "done"}
                                      className="mt-1"
                                    />
                                    <p className={`text-sm font-medium ${task.status === "done" ? "text-slate-500 line-through" : "text-white"}`}>
                                      {task.title}
                                    </p>
                                  </div>
                                  <p className="text-xs text-slate-500 mb-2 ml-6">{task.description}</p>
                                  <div className="flex items-center gap-2 ml-6 flex-wrap">
                                    <Badge className={`text-xs ${componentColors[task.component]?.bg || "bg-slate-500/20"} ${componentColors[task.component]?.text || "text-slate-400"} ${componentColors[task.component]?.border || ""}`}>
                                      {task.component === "tts-lipsync" ? "TTS" : task.component.toUpperCase()}
                                    </Badge>
                                    <Badge className={`text-xs ${priorityColors[task.priority as TaskPriority]?.bg || "bg-slate-500/20"} ${priorityColors[task.priority as TaskPriority]?.text || "text-slate-400"}`}>
                                      {task.priority}
                                    </Badge>
                                    {task.assignee && (
                                      <span className="text-xs text-slate-500">{task.assignee}</span>
                                    )}
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-white"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setEditingTask(task);
                                    }}
                                  >
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                    </svg>
                                  </Button>
                                </div>
                              ))}
                            </div>
                          </ScrollArea>
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="list">
                    <ScrollArea className="h-[600px]">
                      <div className="space-y-2">
                        {filteredTasks.map((task) => (
                          <div
                            key={task.id}
                            className="p-4 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-slate-600 transition-all group relative"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                <Checkbox
                                  checked={task.status === "done"}
                                  onCheckedChange={() => toggleTaskStatus(task.id)}
                                />
                                <div>
                                  <p className={`font-medium ${task.status === "done" ? "text-slate-500 line-through" : "text-white"}`}>
                                    {task.title}
                                  </p>
                                  <p className="text-xs text-slate-500">{task.description}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge className={`text-xs ${componentColors[task.component]?.bg || "bg-slate-500/20"} ${componentColors[task.component]?.text || "text-slate-400"}`}>
                                  {task.component === "tts-lipsync" ? "TTS" : task.component.toUpperCase()}
                                </Badge>
                                <Badge className={`text-xs ${statusColors[task.status as TaskStatus]?.bg || "bg-slate-500/20"} ${statusColors[task.status as TaskStatus]?.text || "text-slate-400"}`}>
                                  {task.status === "in_progress" ? "In Progress" : task.status === "done" ? "Done" : "To Do"}
                                </Badge>
                                <Badge className={`text-xs ${priorityColors[task.priority as TaskPriority]?.bg || "bg-slate-500/20"} ${priorityColors[task.priority as TaskPriority]?.text || "text-slate-400"}`}>
                                  {task.priority}
                                </Badge>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-white"
                                  onClick={() => setEditingTask(task)}
                                >
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                  </svg>
                                </Button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Quick Links */}
        <div className="mt-8 flex gap-4">
          <Button asChild variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700">
            <Link href="/docs">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              View Documentation
            </Link>
          </Button>
          <Button asChild variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700">
            <Link href="/milestones">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              View Milestones
            </Link>
          </Button>
          <Button asChild className="bg-gradient-to-r from-cyan-500 to-purple-600 text-white hover:opacity-90">
            <Link href="/">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
              </svg>
              View Presentation
            </Link>
          </Button>
        </div>
      </main>

      {/* Edit Task Dialog */}
      <Dialog open={!!editingTask} onOpenChange={(open) => !open && setEditingTask(null)}>
        <DialogContent className="bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">Edit Task</DialogTitle>
            <DialogDescription className="text-slate-400">
              Update task details and notes
            </DialogDescription>
          </DialogHeader>
          {editingTask && (
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-slate-300">Title</Label>
                <Input
                  value={editingTask.title}
                  onChange={(e) => setEditingTask((p) => p ? { ...p, title: e.target.value } : null)}
                  className="bg-slate-800 border-slate-700 text-white"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Description</Label>
                <Textarea
                  value={editingTask.description}
                  onChange={(e) => setEditingTask((p) => p ? { ...p, description: e.target.value } : null)}
                  className="bg-slate-800 border-slate-700 text-white"
                  rows={2}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Status</Label>
                  <Select
                    value={editingTask.status}
                    onValueChange={(v: TaskStatus) => setEditingTask((p) => p ? { ...p, status: v } : null)}
                  >
                    <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="todo">To Do</SelectItem>
                      <SelectItem value="in_progress">In Progress</SelectItem>
                      <SelectItem value="done">Done</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Priority</Label>
                  <Select
                    value={editingTask.priority}
                    onValueChange={(v: TaskPriority) => setEditingTask((p) => p ? { ...p, priority: v } : null)}
                  >
                    <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Assignee</Label>
                <Input
                  value={editingTask.assignee || ""}
                  onChange={(e) => setEditingTask((p) => p ? { ...p, assignee: e.target.value } : null)}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="Assignee name"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Notes</Label>
                <Textarea
                  value={editingTask.notes || ""}
                  onChange={(e) => setEditingTask((p) => p ? { ...p, notes: e.target.value } : null)}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="Additional notes..."
                  rows={3}
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingTask(null)} className="border-slate-700 text-slate-300">
              Cancel
            </Button>
            <Button onClick={handleUpdateTask} className="bg-gradient-to-r from-cyan-500 to-purple-600 text-white">
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Navigation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}
