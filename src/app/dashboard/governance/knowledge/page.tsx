"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  BookOpen,
  Plus,
  Edit,
  Trash2,
  ArrowLeft,
  Search,
  Tag,
  Filter
} from "lucide-react";

interface KnowledgeEntry {
  id: number;
  category: string;
  title: string;
  content: Record<string, unknown>;
  tags: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export default function KnowledgePage() {
  const [entries, setEntries] = useState<KnowledgeEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newEntry, setNewEntry] = useState({
    title: "",
    category: "context",
    content: { text: "" },
    tags: [] as string[]
  });
  const [tagInput, setTagInput] = useState("");

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  useEffect(() => {
    fetchEntries();
  }, [categoryFilter]);

  async function fetchEntries() {
    try {
      const token = localStorage.getItem("access_token");
      const params = new URLSearchParams();
      if (categoryFilter) params.append("category", categoryFilter);

      const res = await fetch(`${apiUrl}/api/governance/knowledge?${params}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setEntries(data.entries);
      }
    } catch (error) {
      console.error("Failed to fetch knowledge entries:", error);
    } finally {
      setLoading(false);
    }
  }

  async function createEntry() {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${apiUrl}/api/governance/knowledge`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newEntry)
      });
      if (res.ok) {
        setShowCreateModal(false);
        setNewEntry({ title: "", category: "context", content: { text: "" }, tags: [] });
        fetchEntries();
      }
    } catch (error) {
      console.error("Failed to create entry:", error);
    }
  }

  async function deleteEntry(id: number) {
    if (!confirm("Are you sure you want to delete this knowledge entry?")) return;
    try {
      const token = localStorage.getItem("access_token");
      await fetch(`${apiUrl}/api/governance/knowledge/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchEntries();
    } catch (error) {
      console.error("Failed to delete entry:", error);
    }
  }

  function addTag() {
    if (tagInput.trim() && !newEntry.tags.includes(tagInput.trim())) {
      setNewEntry({ ...newEntry, tags: [...newEntry.tags, tagInput.trim()] });
      setTagInput("");
    }
  }

  function removeTag(tag: string) {
    setNewEntry({ ...newEntry, tags: newEntry.tags.filter((t) => t !== tag) });
  }

  const filteredEntries = entries.filter((e) =>
    e.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const categoryColors: Record<string, string> = {
    context: "bg-blue-500/20 text-blue-400",
    standard: "bg-emerald-500/20 text-emerald-400",
    constraint: "bg-amber-500/20 text-amber-400",
    reference: "bg-purple-500/20 text-purple-400"
  };

  const categories = ["context", "standard", "constraint", "reference"];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Link href="/dashboard/governance">
            <button className="p-2 text-slate-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              <BookOpen className="w-7 h-7 text-emerald-400" />
              Knowledge Base
            </h1>
            <p className="text-slate-400 mt-1">Manage AI context, standards, and constraints</p>
          </div>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-500 rounded-lg font-medium text-white hover:bg-emerald-600 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Entry
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
          <input
            type="text"
            placeholder="Search knowledge..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-3 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          >
            <option value="">All Categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Entries Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="bg-white/5 border border-white/10 rounded-xl p-6 animate-pulse">
              <div className="h-5 bg-white/10 rounded w-1/3 mb-3" />
              <div className="h-6 bg-white/10 rounded w-2/3 mb-2" />
              <div className="h-4 bg-white/10 rounded w-full" />
            </div>
          ))}
        </div>
      ) : filteredEntries.length === 0 ? (
        <div className="text-center py-12">
          <BookOpen className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">No knowledge entries found. Add your first entry to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredEntries.map((entry, index) => (
            <motion.div
              key={entry.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white/5 border border-white/10 rounded-xl p-6 hover:bg-white/10 transition-all group"
            >
              <div className="flex items-start justify-between mb-3">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${categoryColors[entry.category] || "bg-slate-500/20 text-slate-400"}`}>
                  {entry.category.toUpperCase()}
                </span>
                <button
                  onClick={() => deleteEntry(entry.id)}
                  className="p-1 text-slate-500 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{entry.title}</h3>
              {entry.content && typeof entry.content === "object" && "text" in entry.content && (
                <p className="text-slate-400 text-sm line-clamp-2 mb-3">
                  {String(entry.content.text)}
                </p>
              )}
              {entry.tags && entry.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {entry.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 bg-white/5 rounded text-xs text-slate-400"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
              <p className="text-slate-500 text-xs mt-3">
                Updated {new Date(entry.updated_at).toLocaleDateString()}
              </p>
            </motion.div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-900 border border-white/10 rounded-2xl p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Add Knowledge Entry</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Title</label>
                <input
                  type="text"
                  value={newEntry.title}
                  onChange={(e) => setNewEntry({ ...newEntry, title: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                  placeholder="Enter title"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Category</label>
                <select
                  value={newEntry.category}
                  onChange={(e) => setNewEntry({ ...newEntry, category: e.target.value })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                >
                  <option value="context">Context</option>
                  <option value="standard">Standard</option>
                  <option value="constraint">Constraint</option>
                  <option value="reference">Reference</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Content</label>
                <textarea
                  value={newEntry.content.text || ""}
                  onChange={(e) => setNewEntry({ ...newEntry, content: { text: e.target.value } })}
                  rows={4}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/50 resize-none"
                  placeholder="Enter content"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Tags</label>
                <div className="flex gap-2 mb-2">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addTag())}
                    className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
                    placeholder="Add tag and press Enter"
                  />
                  <button
                    onClick={addTag}
                    className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors"
                  >
                    <Tag className="w-4 h-4" />
                  </button>
                </div>
                {newEntry.tags.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {newEntry.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-sm flex items-center gap-1"
                      >
                        {tag}
                        <button onClick={() => removeTag(tag)} className="hover:text-white">
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white hover:bg-white/10 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={createEntry}
                disabled={!newEntry.title}
                className="flex-1 px-4 py-2 bg-emerald-500 rounded-lg font-medium text-white hover:bg-emerald-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Create Entry
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
