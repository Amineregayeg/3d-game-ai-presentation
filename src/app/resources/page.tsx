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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getResources, createResource, deleteResource, type Resource } from "@/lib/api";

const categoryConfig: Record<string, { icon: string; color: string; label: string }> = {
  paper: {
    icon: "M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253",
    color: "text-purple-400 bg-purple-500/20",
    label: "Research Papers",
  },
  tutorial: {
    icon: "M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z",
    color: "text-blue-400 bg-blue-500/20",
    label: "Tutorials",
  },
  tool: {
    icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z",
    color: "text-amber-400 bg-amber-500/20",
    label: "Tools",
  },
  library: {
    icon: "M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z",
    color: "text-emerald-400 bg-emerald-500/20",
    label: "Libraries",
  },
  docs: {
    icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
    color: "text-cyan-400 bg-cyan-500/20",
    label: "Documentation",
  },
};

const componentColors: Record<string, { bg: string; text: string }> = {
  stt: { bg: "bg-cyan-500/20", text: "text-cyan-400" },
  rag: { bg: "bg-emerald-500/20", text: "text-emerald-400" },
  "tts-lipsync": { bg: "bg-rose-500/20", text: "text-rose-400" },
  mcp: { bg: "bg-orange-500/20", text: "text-orange-400" },
};

export default function ResourcesPage() {
  const [resources, setResources] = useState<Resource[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ category: "all", component: "all" });
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newResource, setNewResource] = useState({
    title: "",
    url: "",
    category: "paper" as Resource["category"],
    description: "",
    component: "",
    tags: [] as string[],
  });
  const [tagInput, setTagInput] = useState("");

  useEffect(() => {
    loadResources();
  }, [filter]);

  const loadResources = async () => {
    try {
      const filters: { category?: string; component?: string } = {};
      if (filter.category !== "all") filters.category = filter.category;
      if (filter.component !== "all") filters.component = filter.component;
      const data = await getResources(filters);
      setResources(data);
    } catch (error) {
      console.error("Failed to load resources:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddResource = async () => {
    try {
      await createResource(newResource);
      setAddDialogOpen(false);
      setNewResource({
        title: "",
        url: "",
        category: "paper",
        description: "",
        component: "",
        tags: [],
      });
      setTagInput("");
      await loadResources();
    } catch (error) {
      console.error("Failed to create resource:", error);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this resource?")) return;
    try {
      await deleteResource(id);
      await loadResources();
    } catch (error) {
      console.error("Failed to delete resource:", error);
    }
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !newResource.tags.includes(tagInput.trim())) {
      setNewResource((p) => ({ ...p, tags: [...p.tags, tagInput.trim()] }));
      setTagInput("");
    }
  };

  const handleRemoveTag = (tag: string) => {
    setNewResource((p) => ({ ...p, tags: p.tags.filter((t) => t !== tag) }));
  };

  const groupedResources = resources.reduce((acc, r) => {
    const cat = r.category;
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(r);
    return acc;
  }, {} as Record<string, Resource[]>);

  return (
    <PageLayout
      title="Resources"
      description="Curated collection of papers, tools, tutorials, and documentation"
      badges={[{ label: `${resources.length} Resources` }]}
      gradientOrbs={[
        { color: "teal", position: "top-1/4 -left-32" },
        { color: "cyan", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Category Stats */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        {Object.entries(categoryConfig).map(([cat, config]) => {
          const count = resources.filter((r) => r.category === cat).length;
          return (
            <Card
              key={cat}
              className={`bg-slate-900/50 border-slate-800 cursor-pointer hover:border-slate-700 transition-all ${filter.category === cat ? "ring-1 ring-slate-600" : ""}`}
              onClick={() => setFilter((p) => ({ ...p, category: p.category === cat ? "all" : cat }))}
            >
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${config.color}`}>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={config.icon} />
                    </svg>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-white">{count}</p>
                    <p className="text-xs text-slate-400">{config.label}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

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
          {filter.category !== "all" && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFilter((p) => ({ ...p, category: "all" }))}
              className="text-slate-400"
            >
              Clear category filter
            </Button>
          )}
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-teal-500 to-cyan-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Resource
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700">
            <DialogHeader>
              <DialogTitle className="text-white">Add Resource</DialogTitle>
              <DialogDescription className="text-slate-400">
                Add a new resource to the collection
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-slate-300">Title</Label>
                <Input
                  value={newResource.title}
                  onChange={(e) => setNewResource((p) => ({ ...p, title: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="Resource title"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">URL</Label>
                <Input
                  value={newResource.url}
                  onChange={(e) => setNewResource((p) => ({ ...p, url: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="https://..."
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-slate-300">Category</Label>
                  <Select
                    value={newResource.category}
                    onValueChange={(v: Resource["category"]) => setNewResource((p) => ({ ...p, category: v }))}
                  >
                    <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="paper">Paper</SelectItem>
                      <SelectItem value="tutorial">Tutorial</SelectItem>
                      <SelectItem value="tool">Tool</SelectItem>
                      <SelectItem value="library">Library</SelectItem>
                      <SelectItem value="docs">Documentation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-300">Component</Label>
                  <Select
                    value={newResource.component}
                    onValueChange={(v) => setNewResource((p) => ({ ...p, component: v }))}
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
                <Label className="text-slate-300">Description</Label>
                <Textarea
                  value={newResource.description}
                  onChange={(e) => setNewResource((p) => ({ ...p, description: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="Brief description"
                  rows={2}
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Tags</Label>
                <div className="flex gap-2">
                  <Input
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddTag())}
                    className="bg-slate-800 border-slate-700 text-white"
                    placeholder="Add tag..."
                  />
                  <Button type="button" variant="outline" onClick={handleAddTag} className="border-slate-700 text-slate-300">
                    Add
                  </Button>
                </div>
                {newResource.tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {newResource.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="bg-slate-800 text-slate-300">
                        {tag}
                        <button
                          type="button"
                          onClick={() => handleRemoveTag(tag)}
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
              <Button onClick={handleAddResource} className="bg-gradient-to-r from-teal-500 to-cyan-600 text-white">
                Add Resource
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Resources Grid */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Resource Library</CardTitle>
          <CardDescription className="text-slate-400">
            {filter.category !== "all"
              ? categoryConfig[filter.category]?.label || "Resources"
              : "All curated resources"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="animate-pulse p-4 bg-slate-800/50 rounded-lg">
                  <div className="h-5 bg-slate-700 rounded w-3/4 mb-2" />
                  <div className="h-4 bg-slate-700 rounded w-full" />
                </div>
              ))}
            </div>
          ) : filter.category !== "all" ? (
            <ScrollArea className="h-[500px]">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {resources.map((resource) => (
                  <ResourceCard
                    key={resource.id}
                    resource={resource}
                    onDelete={() => handleDelete(resource.id)}
                  />
                ))}
                {resources.length === 0 && (
                  <div className="col-span-2 text-center py-12 text-slate-400">
                    No resources in this category
                  </div>
                )}
              </div>
            </ScrollArea>
          ) : (
            <Tabs defaultValue="all" className="w-full">
              <TabsList className="bg-slate-800/50 mb-4">
                <TabsTrigger value="all" className="data-[state=active]:bg-slate-700">All</TabsTrigger>
                {Object.entries(categoryConfig).map(([cat, config]) => (
                  <TabsTrigger key={cat} value={cat} className="data-[state=active]:bg-slate-700">
                    {config.label}
                  </TabsTrigger>
                ))}
              </TabsList>

              <TabsContent value="all">
                <ScrollArea className="h-[500px]">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {resources.map((resource) => (
                      <ResourceCard
                        key={resource.id}
                        resource={resource}
                        onDelete={() => handleDelete(resource.id)}
                      />
                    ))}
                    {resources.length === 0 && (
                      <div className="col-span-2 text-center py-12">
                        <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                          <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                          </svg>
                        </div>
                        <h3 className="text-lg font-medium text-white mb-2">No resources yet</h3>
                        <p className="text-slate-400">Add your first resource to build your library</p>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </TabsContent>

              {Object.keys(categoryConfig).map((cat) => (
                <TabsContent key={cat} value={cat}>
                  <ScrollArea className="h-[500px]">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {(groupedResources[cat] || []).map((resource) => (
                        <ResourceCard
                          key={resource.id}
                          resource={resource}
                          onDelete={() => handleDelete(resource.id)}
                        />
                      ))}
                      {(!groupedResources[cat] || groupedResources[cat].length === 0) && (
                        <div className="col-span-2 text-center py-12 text-slate-400">
                          No {categoryConfig[cat]?.label.toLowerCase()} yet
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>
              ))}
            </Tabs>
          )}
        </CardContent>
      </Card>
    </PageLayout>
  );
}

// Helper to extract YouTube video ID from URL
function getYouTubeVideoId(url: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\s?]+)/,
    /youtube\.com\/watch\?.*v=([^&\s]+)/,
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

function ResourceCard({ resource, onDelete }: { resource: Resource; onDelete: () => void }) {
  const [showVideo, setShowVideo] = useState(false);
  const config = categoryConfig[resource.category] || categoryConfig.docs;
  const youtubeId = getYouTubeVideoId(resource.url);
  const isYouTube = !!youtubeId;

  return (
    <div className={`p-4 bg-slate-800/30 rounded-lg hover:bg-slate-800/50 transition-colors group ${isYouTube && showVideo ? "col-span-2" : ""}`}>
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${config.color} shrink-0`}>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={config.icon} />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2">
              <a
                href={resource.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-white font-medium hover:text-cyan-400 transition-colors truncate block"
              >
                {resource.title}
              </a>
              {isYouTube && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowVideo(!showVideo)}
                  className="text-red-500 hover:text-red-400 hover:bg-red-500/10"
                >
                  <svg className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                  </svg>
                  {showVideo ? "Hide" : "Watch"}
                </Button>
              )}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </Button>
          </div>
          {resource.description && (
            <p className="text-sm text-slate-400 mt-1 line-clamp-2">{resource.description}</p>
          )}

          {/* YouTube Embed */}
          {isYouTube && showVideo && (
            <div className="mt-4 rounded-lg overflow-hidden bg-black aspect-video">
              <iframe
                src={`https://www.youtube.com/embed/${youtubeId}`}
                title={resource.title}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-full"
              />
            </div>
          )}

          <div className="flex items-center gap-2 mt-2 flex-wrap">
            {resource.component && (
              <Badge className={`${componentColors[resource.component]?.bg || "bg-slate-500/20"} ${componentColors[resource.component]?.text || "text-slate-400"} text-xs`}>
                {resource.component === "tts-lipsync" ? "TTS" : (resource.component?.toUpperCase() || "")}
              </Badge>
            )}
            {resource.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="secondary" className="bg-slate-700 text-slate-400 text-xs">
                {tag}
              </Badge>
            ))}
            {resource.tags.length > 3 && (
              <Badge variant="secondary" className="bg-slate-700 text-slate-500 text-xs">
                +{resource.tags.length - 3}
              </Badge>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
