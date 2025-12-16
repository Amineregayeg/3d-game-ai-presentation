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
import { getGlossary, createGlossaryTerm, deleteGlossaryTerm, type GlossaryTerm } from "@/lib/api";

const categoryColors: Record<string, { bg: string; text: string; label: string }> = {
  ml: { bg: "bg-purple-500/20", text: "text-purple-400", label: "Machine Learning" },
  audio: { bg: "bg-cyan-500/20", text: "text-cyan-400", label: "Audio/DSP" },
  graphics: { bg: "bg-rose-500/20", text: "text-rose-400", label: "Graphics/3D" },
  general: { bg: "bg-slate-500/20", text: "text-slate-400", label: "General" },
};

const componentColors: Record<string, { bg: string; text: string }> = {
  stt: { bg: "bg-cyan-500/20", text: "text-cyan-400" },
  rag: { bg: "bg-emerald-500/20", text: "text-emerald-400" },
  "tts-lipsync": { bg: "bg-rose-500/20", text: "text-rose-400" },
  mcp: { bg: "bg-orange-500/20", text: "text-orange-400" },
};

export default function GlossaryPage() {
  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ category: "all", component: "all" });
  const [search, setSearch] = useState("");
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newTerm, setNewTerm] = useState({
    term: "",
    definition: "",
    category: "general",
    component: "",
    related_terms: [] as string[],
  });
  const [relatedInput, setRelatedInput] = useState("");

  useEffect(() => {
    loadGlossary();
  }, [filter]);

  const loadGlossary = async () => {
    try {
      const filters: { category?: string; component?: string } = {};
      if (filter.category !== "all") filters.category = filter.category;
      if (filter.component !== "all") filters.component = filter.component;
      const data = await getGlossary(filters);
      setTerms(data);
    } catch (error) {
      console.error("Failed to load glossary:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddTerm = async () => {
    try {
      await createGlossaryTerm(newTerm);
      setAddDialogOpen(false);
      setNewTerm({
        term: "",
        definition: "",
        category: "general",
        component: "",
        related_terms: [],
      });
      setRelatedInput("");
      await loadGlossary();
    } catch (error) {
      console.error("Failed to create term:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this term?")) return;
    try {
      await deleteGlossaryTerm(id);
      await loadGlossary();
    } catch (error) {
      console.error("Failed to delete term:", error);
    }
  };

  const handleAddRelated = () => {
    if (relatedInput.trim() && !newTerm.related_terms.includes(relatedInput.trim())) {
      setNewTerm((p) => ({ ...p, related_terms: [...p.related_terms, relatedInput.trim()] }));
      setRelatedInput("");
    }
  };

  const handleRemoveRelated = (term: string) => {
    setNewTerm((p) => ({ ...p, related_terms: p.related_terms.filter((t) => t !== term) }));
  };

  const filteredTerms = terms.filter((t) =>
    t.term.toLowerCase().includes(search.toLowerCase()) ||
    t.definition.toLowerCase().includes(search.toLowerCase())
  );

  // Group by first letter
  const groupedTerms = filteredTerms.reduce((acc, term) => {
    const letter = term.term[0].toUpperCase();
    if (!acc[letter]) acc[letter] = [];
    acc[letter].push(term);
    return acc;
  }, {} as Record<string, GlossaryTerm[]>);

  const sortedLetters = Object.keys(groupedTerms).sort();

  return (
    <PageLayout
      title="Technical Glossary"
      description="Definitions of technical terms used in the project"
      badges={[{ label: `${terms.length} Terms` }]}
      gradientOrbs={[
        { color: "amber", position: "top-1/4 -left-32" },
        { color: "orange", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Category Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {Object.entries(categoryColors).map(([cat, config]) => {
          const count = terms.filter((t) => t.category === cat).length;
          return (
            <Card
              key={cat}
              className={`bg-slate-900/50 border-slate-800 cursor-pointer hover:border-slate-700 transition-all ${filter.category === cat ? "ring-1 ring-slate-600" : ""}`}
              onClick={() => setFilter((p) => ({ ...p, category: p.category === cat ? "all" : cat }))}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-white">{count}</p>
                    <p className={`text-sm ${config.text}`}>{config.label}</p>
                  </div>
                  <div className={`w-3 h-3 rounded-full ${config.text.replace("text-", "bg-")}`} />
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Search and Filters */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="relative">
            <svg className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10 w-64 bg-slate-800 border-slate-700 text-white"
              placeholder="Search terms..."
            />
          </div>
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
          {filter.category !== "all" && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFilter((p) => ({ ...p, category: "all" }))}
              className="text-slate-400"
            >
              Clear category
            </Button>
          )}
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-amber-500 to-orange-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Term
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700">
            <DialogHeader>
              <DialogTitle className="text-white">Add Glossary Term</DialogTitle>
              <DialogDescription className="text-slate-400">
                Add a new technical term definition
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-slate-300">Term</Label>
                <Input
                  value={newTerm.term}
                  onChange={(e) => setNewTerm((p) => ({ ...p, term: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="e.g., STFT"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Definition</Label>
                <Textarea
                  value={newTerm.definition}
                  onChange={(e) => setNewTerm((p) => ({ ...p, definition: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="What does this term mean?"
                  rows={3}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Category</Label>
                  <Select
                    value={newTerm.category}
                    onValueChange={(v) => setNewTerm((p) => ({ ...p, category: v }))}
                  >
                    <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="ml">Machine Learning</SelectItem>
                      <SelectItem value="audio">Audio/DSP</SelectItem>
                      <SelectItem value="graphics">Graphics/3D</SelectItem>
                      <SelectItem value="general">General</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Component</Label>
                  <Select
                    value={newTerm.component}
                    onValueChange={(v) => setNewTerm((p) => ({ ...p, component: v }))}
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
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Related Terms</Label>
                <div className="flex gap-2">
                  <Input
                    value={relatedInput}
                    onChange={(e) => setRelatedInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddRelated())}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="Add related term..."
                  />
                  <Button type="button" variant="outline" onClick={handleAddRelated} className="border-slate-700 text-slate-300">
                    Add
                  </Button>
                </div>
                {newTerm.related_terms.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {newTerm.related_terms.map((term) => (
                      <Badge key={term} variant="secondary" className="bg-slate-800 text-slate-300">
                        {term}
                        <button
                          type="button"
                          onClick={() => handleRemoveRelated(term)}
                          className="ml-1 hover:text-red-400"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                Cancel
              </Button>
              <Button onClick={handleAddTerm} className="bg-gradient-to-r from-amber-500 to-orange-600 text-white">
                Add Term
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Alphabet Navigation */}
      <div className="flex flex-wrap gap-1 mb-6">
        {sortedLetters.map((letter) => (
          <a
            key={letter}
            href={`#letter-${letter}`}
            className="w-8 h-8 flex items-center justify-center rounded bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white transition-colors text-sm font-medium"
          >
            {letter}
          </a>
        ))}
      </div>

      {/* Glossary */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Terms</CardTitle>
          <CardDescription className="text-slate-400">
            {filteredTerms.length} {filteredTerms.length === 1 ? "term" : "terms"} found
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px] pr-4">
            {loading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="animate-pulse p-4 bg-slate-800/50 rounded-lg">
                    <div className="h-5 bg-slate-700 rounded w-1/4 mb-2" />
                    <div className="h-4 bg-slate-700 rounded w-3/4" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-8">
                {sortedLetters.map((letter) => (
                  <div key={letter} id={`letter-${letter}`}>
                    <h3 className="text-2xl font-bold text-white mb-4 sticky top-0 bg-slate-900/50 py-2">
                      {letter}
                    </h3>
                    <div className="space-y-3">
                      {groupedTerms[letter].map((term) => {
                        const catColors = categoryColors[term.category || "general"] || categoryColors.general;
                        return (
                          <div
                            key={term.id}
                            className="p-4 bg-slate-800/30 rounded-lg hover:bg-slate-800/50 transition-colors group"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="text-lg font-semibold text-white">{term.term}</h4>
                                  <Badge className={`${catColors.bg} ${catColors.text} text-xs`}>
                                    {catColors.label}
                                  </Badge>
                                  {term.component && (
                                    <Badge className={`${componentColors[term.component]?.bg || "bg-slate-500/20"} ${componentColors[term.component]?.text || "text-slate-400"} text-xs`}>
                                      {term.component === "tts-lipsync" ? "TTS" : (term.component?.toUpperCase() || "")}
                                    </Badge>
                                  )}
                                </div>
                                <p className="text-slate-400">{term.definition}</p>
                                {term.related_terms.length > 0 && (
                                  <div className="flex items-center gap-2 mt-2">
                                    <span className="text-xs text-slate-500">Related:</span>
                                    {term.related_terms.map((rt) => (
                                      <span key={rt} className="text-xs text-cyan-400 hover:underline cursor-pointer">
                                        {rt}
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDelete(term.id)}
                                className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                {filteredTerms.length === 0 && (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium text-white mb-2">No terms found</h3>
                    <p className="text-slate-400">
                      {search ? "Try a different search" : "Add your first glossary term"}
                    </p>
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
