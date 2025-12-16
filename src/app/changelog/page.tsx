"use client";

import { useState, useEffect } from "react";
import PageLayout from "@/components/page-layout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
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
import { getChangelog, createChangelogEntry, type ChangelogEntry } from "@/lib/api";
import { format } from "date-fns";

const componentColors: Record<string, { bg: string; text: string; gradient: string }> = {
  stt: { bg: "bg-cyan-500/20", text: "text-cyan-400", gradient: "from-cyan-500 to-blue-600" },
  rag: { bg: "bg-emerald-500/20", text: "text-emerald-400", gradient: "from-emerald-500 to-teal-600" },
  "tts-lipsync": { bg: "bg-rose-500/20", text: "text-rose-400", gradient: "from-rose-500 to-pink-600" },
  mcp: { bg: "bg-orange-500/20", text: "text-orange-400", gradient: "from-orange-500 to-amber-600" },
};

export default function ChangelogPage() {
  const [entries, setEntries] = useState<ChangelogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newEntry, setNewEntry] = useState({
    version: "",
    title: "",
    description: "",
    changes: [] as string[],
    component: "",
    release_date: new Date().toISOString().split("T")[0],
    author: "",
  });
  const [changeInput, setChangeInput] = useState("");

  useEffect(() => {
    loadChangelog();
  }, [filter]);

  const loadChangelog = async () => {
    try {
      const data = await getChangelog(filter !== "all" ? filter : undefined);
      setEntries(data);
    } catch (error) {
      console.error("Failed to load changelog:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddEntry = async () => {
    try {
      await createChangelogEntry(newEntry);
      setAddDialogOpen(false);
      setNewEntry({
        version: "",
        title: "",
        description: "",
        changes: [],
        component: "",
        release_date: new Date().toISOString().split("T")[0],
        author: "",
      });
      setChangeInput("");
      await loadChangelog();
    } catch (error) {
      console.error("Failed to create changelog entry:", error);
    }
  };

  const handleAddChange = () => {
    if (changeInput.trim() && !newEntry.changes.includes(changeInput.trim())) {
      setNewEntry((p) => ({ ...p, changes: [...p.changes, changeInput.trim()] }));
      setChangeInput("");
    }
  };

  const handleRemoveChange = (change: string) => {
    setNewEntry((p) => ({ ...p, changes: p.changes.filter((c) => c !== change) }));
  };

  // Group by month/year
  const groupedEntries = entries.reduce((acc, entry) => {
    const date = new Date(entry.release_date);
    const key = format(date, "MMMM yyyy");
    if (!acc[key]) acc[key] = [];
    acc[key].push(entry);
    return acc;
  }, {} as Record<string, ChangelogEntry[]>);

  return (
    <PageLayout
      title="Changelog"
      description="Version history and release notes"
      badges={[{ label: `${entries.length} Releases` }]}
      gradientOrbs={[
        { color: "green", position: "top-1/4 -left-32" },
        { color: "emerald", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Filters and Actions */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Select value={filter} onValueChange={setFilter}>
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
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Release
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700 max-w-lg">
            <DialogHeader>
              <DialogTitle className="text-white">Add Changelog Entry</DialogTitle>
              <DialogDescription className="text-slate-400">
                Document a new release or version update
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[60vh]">
              <div className="space-y-4 py-4 pr-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-slate-300">Version</Label>
                    <Input
                      value={newEntry.version}
                      onChange={(e) => setNewEntry((p) => ({ ...p, version: e.target.value }))}
                      className="bg-slate-800 border-slate-700 text-white"
                      placeholder="e.g., v0.1.0"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">Release Date</Label>
                    <Input
                      type="date"
                      value={newEntry.release_date}
                      onChange={(e) => setNewEntry((p) => ({ ...p, release_date: e.target.value }))}
                      className="bg-slate-800 border-slate-700 text-white"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Title</Label>
                  <Input
                    value={newEntry.title}
                    onChange={(e) => setNewEntry((p) => ({ ...p, title: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="e.g., Audio Frontend Complete"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-slate-300">Component</Label>
                    <Select
                      value={newEntry.component}
                      onValueChange={(v) => setNewEntry((p) => ({ ...p, component: v }))}
                    >
                      <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-slate-700">
                        <SelectItem value="">General</SelectItem>
                        <SelectItem value="stt">VoxFormer STT</SelectItem>
                        <SelectItem value="rag">Advanced RAG</SelectItem>
                        <SelectItem value="tts-lipsync">TTS + LipSync</SelectItem>
                        <SelectItem value="mcp">Blender MCP</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">Author</Label>
                    <Input
                      value={newEntry.author}
                      onChange={(e) => setNewEntry((p) => ({ ...p, author: e.target.value }))}
                      className="bg-slate-800 border-slate-700 text-white"
                      placeholder="Your name"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Description</Label>
                  <Textarea
                    value={newEntry.description}
                    onChange={(e) => setNewEntry((p) => ({ ...p, description: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="Brief summary of this release"
                    rows={2}
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Changes</Label>
                  <div className="flex gap-2">
                    <Input
                      value={changeInput}
                      onChange={(e) => setChangeInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddChange())}
                      className="bg-slate-800 border-slate-700 text-white"
                      placeholder="Add a change..."
                    />
                    <Button type="button" variant="outline" onClick={handleAddChange} className="border-slate-700 text-slate-300">
                      Add
                    </Button>
                  </div>
                  {newEntry.changes.length > 0 && (
                    <ul className="space-y-1 mt-2">
                      {newEntry.changes.map((change, i) => (
                        <li key={i} className="flex items-center gap-2 text-sm text-slate-300 bg-slate-800/50 p-2 rounded">
                          <span className="text-emerald-400">+</span>
                          <span className="flex-1">{change}</span>
                          <button
                            type="button"
                            onClick={() => handleRemoveChange(change)}
                            className="text-slate-500 hover:text-red-400"
                          >
                            ×
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </ScrollArea>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                Cancel
              </Button>
              <Button onClick={handleAddEntry} className="bg-gradient-to-r from-green-500 to-emerald-600 text-white">
                Add Release
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Changelog */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Release History</CardTitle>
          <CardDescription className="text-slate-400">
            All project releases and updates
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px] pr-4">
            {loading ? (
              <div className="space-y-6">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="animate-pulse">
                    <div className="h-4 bg-slate-700 rounded w-1/4 mb-4" />
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <div className="h-5 bg-slate-700 rounded w-1/3 mb-2" />
                      <div className="h-4 bg-slate-700 rounded w-2/3" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-8">
                {Object.entries(groupedEntries).map(([period, periodEntries]) => (
                  <div key={period}>
                    <h3 className="text-sm font-medium text-slate-400 mb-4 sticky top-0 bg-slate-900/50 py-2">
                      {period}
                    </h3>
                    <div className="space-y-4 relative">
                      <div className="absolute left-3 top-0 bottom-0 w-0.5 bg-slate-800" />
                      {periodEntries.map((entry) => {
                        const colors = componentColors[entry.component || ""] || null;
                        return (
                          <div key={entry.id} className="flex gap-4 relative">
                            <div className={`w-6 h-6 rounded-full ${colors?.bg || "bg-slate-700"} flex items-center justify-center z-10 shrink-0`}>
                              <div className={`w-2 h-2 rounded-full ${colors?.text?.replace("text-", "bg-") || "bg-slate-500"}`} />
                            </div>
                            <div className="flex-1 bg-slate-800/30 rounded-lg p-4 hover:bg-slate-800/50 transition-colors">
                              <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <Badge className={`${colors?.bg || "bg-slate-700"} ${colors?.text || "text-slate-300"} font-mono`}>
                                    {entry.version}
                                  </Badge>
                                  <h4 className="text-white font-medium">{entry.title}</h4>
                                </div>
                                <span className="text-xs text-slate-500">
                                  {format(new Date(entry.release_date), "MMM d, yyyy")}
                                </span>
                              </div>
                              {entry.description && (
                                <p className="text-sm text-slate-400 mb-3">{entry.description}</p>
                              )}
                              {entry.changes.length > 0 && (
                                <ul className="space-y-1">
                                  {entry.changes.map((change, i) => (
                                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                                      <span className="text-emerald-400 mt-0.5">+</span>
                                      <span>{change}</span>
                                    </li>
                                  ))}
                                </ul>
                              )}
                              <div className="flex items-center gap-4 mt-3 pt-3 border-t border-slate-700">
                                {entry.component && (
                                  <Badge className={`${colors?.bg || "bg-slate-500/20"} ${colors?.text || "text-slate-400"} text-xs`}>
                                    {entry.component === "tts-lipsync" ? "TTS" : (entry.component?.toUpperCase() || "General")}
                                  </Badge>
                                )}
                                {entry.author && (
                                  <span className="text-xs text-slate-500">by {entry.author}</span>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                {entries.length === 0 && (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium text-white mb-2">No releases yet</h3>
                    <p className="text-slate-400">Add your first changelog entry to track releases</p>
                  </div>
                )}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </PageLayout>
  );
}
