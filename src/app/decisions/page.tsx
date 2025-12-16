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
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { getDecisions, createDecision, updateDecision, deleteDecision, type Decision } from "@/lib/api";
import { format } from "date-fns";

const statusColors: Record<string, { bg: string; text: string; border: string }> = {
  proposed: { bg: "bg-amber-500/20", text: "text-amber-400", border: "border-amber-500/30" },
  accepted: { bg: "bg-emerald-500/20", text: "text-emerald-400", border: "border-emerald-500/30" },
  rejected: { bg: "bg-red-500/20", text: "text-red-400", border: "border-red-500/30" },
  superseded: { bg: "bg-slate-500/20", text: "text-slate-400", border: "border-slate-500/30" },
};

const componentColors: Record<string, { bg: string; text: string }> = {
  stt: { bg: "bg-cyan-500/20", text: "text-cyan-400" },
  rag: { bg: "bg-emerald-500/20", text: "text-emerald-400" },
  "tts-lipsync": { bg: "bg-rose-500/20", text: "text-rose-400" },
  mcp: { bg: "bg-orange-500/20", text: "text-orange-400" },
};

export default function DecisionsPage() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ component: "all", status: "all" });
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newDecision, setNewDecision] = useState({
    title: "",
    status: "proposed" as Decision["status"],
    context: "",
    decision: "",
    consequences: "",
    component: "",
    author: "",
  });

  useEffect(() => {
    loadDecisions();
  }, [filter]);

  const loadDecisions = async () => {
    try {
      const filters: { component?: string; status?: string } = {};
      if (filter.component !== "all") filters.component = filter.component;
      if (filter.status !== "all") filters.status = filter.status;
      const data = await getDecisions(filters);
      setDecisions(data);
    } catch (error) {
      console.error("Failed to load decisions:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddDecision = async () => {
    try {
      await createDecision(newDecision);
      setAddDialogOpen(false);
      setNewDecision({
        title: "",
        status: "proposed",
        context: "",
        decision: "",
        consequences: "",
        component: "",
        author: "",
      });
      await loadDecisions();
    } catch (error) {
      console.error("Failed to create decision:", error);
    }
  };

  const handleUpdateStatus = async (id: number, status: Decision["status"]) => {
    try {
      await updateDecision(id, { status });
      await loadDecisions();
    } catch (error) {
      console.error("Failed to update decision:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this ADR?")) return;
    try {
      await deleteDecision(id);
      await loadDecisions();
    } catch (error) {
      console.error("Failed to delete decision:", error);
    }
  };

  const acceptedCount = decisions.filter((d) => d.status === "accepted").length;
  const proposedCount = decisions.filter((d) => d.status === "proposed").length;

  return (
    <PageLayout
      title="Architecture Decisions"
      description="Architecture Decision Records (ADRs) for the project"
      badges={[
        { label: `${acceptedCount} Accepted` },
        { label: `${proposedCount} Proposed` },
      ]}
      gradientOrbs={[
        { color: "indigo", position: "top-1/4 -left-32" },
        { color: "violet", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Filters and Actions */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Select value={filter.component} onValueChange={(v) => setFilter((p) => ({ ...p, component: v }))}>
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
          <Select value={filter.status} onValueChange={(v) => setFilter((p) => ({ ...p, status: v }))}>
            <SelectTrigger className="w-32 bg-slate-800 border-slate-700 text-slate-300">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-700">
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="proposed">Proposed</SelectItem>
              <SelectItem value="accepted">Accepted</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
              <SelectItem value="superseded">Superseded</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-indigo-500 to-violet-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New ADR
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700 max-w-2xl">
            <DialogHeader>
              <DialogTitle className="text-white">Create Architecture Decision Record</DialogTitle>
              <DialogDescription className="text-slate-400">
                Document an architecture decision using the ADR format
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[60vh]">
              <div className="space-y-4 py-4 pr-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Title</Label>
                  <Input
                    value={newDecision.title}
                    onChange={(e) => setNewDecision((p) => ({ ...p, title: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="e.g., Use Conformer over vanilla Transformer"
                  />
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label className="text-slate-300">Component</Label>
                    <Select
                      value={newDecision.component}
                      onValueChange={(v) => setNewDecision((p) => ({ ...p, component: v }))}
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
                    <Label className="text-slate-300">Status</Label>
                    <Select
                      value={newDecision.status}
                      onValueChange={(v: Decision["status"]) => setNewDecision((p) => ({ ...p, status: v }))}
                    >
                      <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-slate-700">
                        <SelectItem value="proposed">Proposed</SelectItem>
                        <SelectItem value="accepted">Accepted</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-300">Author</Label>
                    <Input
                      value={newDecision.author}
                      onChange={(e) => setNewDecision((p) => ({ ...p, author: e.target.value }))}
                      className="bg-slate-800 border-slate-700 text-white"
                      placeholder="Your name"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Context</Label>
                  <Textarea
                    value={newDecision.context}
                    onChange={(e) => setNewDecision((p) => ({ ...p, context: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="What is the issue or problem we're addressing?"
                    rows={3}
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Decision</Label>
                  <Textarea
                    value={newDecision.decision}
                    onChange={(e) => setNewDecision((p) => ({ ...p, decision: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="What is the decision we're making?"
                    rows={3}
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Consequences</Label>
                  <Textarea
                    value={newDecision.consequences}
                    onChange={(e) => setNewDecision((p) => ({ ...p, consequences: e.target.value }))}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="What are the positive and negative consequences?"
                    rows={3}
                  />
                </div>
              </div>
            </ScrollArea>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                Cancel
              </Button>
              <Button onClick={handleAddDecision} className="bg-gradient-to-r from-indigo-500 to-violet-600 text-white">
                Create ADR
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Decisions List */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Decision Records</CardTitle>
          <CardDescription className="text-slate-400">
            All architecture decisions for the project
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse p-4 bg-slate-800/50 rounded-lg">
                  <div className="h-5 bg-slate-700 rounded w-1/3 mb-2" />
                  <div className="h-4 bg-slate-700 rounded w-2/3" />
                </div>
              ))}
            </div>
          ) : (
            <Accordion type="single" collapsible className="space-y-2">
              {decisions.map((decision) => {
                const colors = statusColors[decision.status];
                return (
                  <AccordionItem
                    key={decision.id}
                    value={decision.id.toString()}
                    className="border border-slate-800 rounded-lg bg-slate-800/30 px-4"
                  >
                    <AccordionTrigger className="hover:no-underline py-4">
                      <div className="flex items-center gap-4 w-full">
                        <span className="text-slate-500 font-mono text-sm">ADR-{decision.id}</span>
                        <span className="text-white font-medium flex-1 text-left">{decision.title}</span>
                        <div className="flex items-center gap-2">
                          {decision.component && (
                            <Badge className={`${componentColors[decision.component]?.bg || "bg-slate-500/20"} ${componentColors[decision.component]?.text || "text-slate-400"}`}>
                              {decision.component === "tts-lipsync" ? "TTS" : (decision.component?.toUpperCase() || "General")}
                            </Badge>
                          )}
                          <Badge className={`${colors.bg} ${colors.text} ${colors.border}`}>
                            {decision.status}
                          </Badge>
                        </div>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pb-4">
                      <div className="space-y-4 pt-2">
                        {decision.context && (
                          <div>
                            <h4 className="text-sm font-medium text-slate-300 mb-1">Context</h4>
                            <p className="text-sm text-slate-400">{decision.context}</p>
                          </div>
                        )}
                        {decision.decision && (
                          <div>
                            <h4 className="text-sm font-medium text-slate-300 mb-1">Decision</h4>
                            <p className="text-sm text-slate-400">{decision.decision}</p>
                          </div>
                        )}
                        {decision.consequences && (
                          <div>
                            <h4 className="text-sm font-medium text-slate-300 mb-1">Consequences</h4>
                            <p className="text-sm text-slate-400">{decision.consequences}</p>
                          </div>
                        )}
                        <div className="flex items-center justify-between pt-4 border-t border-slate-700">
                          <div className="flex items-center gap-4 text-xs text-slate-500">
                            {decision.author && <span>By {decision.author}</span>}
                            <span>{format(new Date(decision.created_at), "MMM d, yyyy")}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Select
                              value={decision.status}
                              onValueChange={(v: Decision["status"]) => handleUpdateStatus(decision.id, v)}
                            >
                              <SelectTrigger className="w-28 h-7 text-xs bg-slate-800 border-slate-700">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-slate-800 border-slate-700">
                                <SelectItem value="proposed">Proposed</SelectItem>
                                <SelectItem value="accepted">Accepted</SelectItem>
                                <SelectItem value="rejected">Rejected</SelectItem>
                                <SelectItem value="superseded">Superseded</SelectItem>
                              </SelectContent>
                            </Select>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDelete(decision.id)}
                              className="text-red-400 hover:text-red-300"
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </Button>
                          </div>
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                );
              })}
              {decisions.length === 0 && (
                <div className="text-center py-12">
                  <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-white mb-2">No decisions recorded</h3>
                  <p className="text-slate-400">Create your first ADR to document architecture decisions</p>
                </div>
              )}
            </Accordion>
          )}
        </CardContent>
      </Card>
    </PageLayout>
  );
}
