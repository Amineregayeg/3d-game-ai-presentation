"use client";

import { useState, useEffect } from "react";
import PageLayout from "@/components/page-layout";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getActivity, type Activity } from "@/lib/api";
import { formatDistanceToNow } from "date-fns";

const activityIcons: Record<string, { icon: string; color: string }> = {
  task_created: { icon: "M12 4v16m8-8H4", color: "text-emerald-400 bg-emerald-500/20" },
  task_status_changed: { icon: "M5 13l4 4L19 7", color: "text-blue-400 bg-blue-500/20" },
  task_updated: { icon: "M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z", color: "text-amber-400 bg-amber-500/20" },
  milestone_created: { icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z", color: "text-purple-400 bg-purple-500/20" },
  milestone_completed: { icon: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z", color: "text-emerald-400 bg-emerald-500/20" },
  decision_created: { icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z", color: "text-cyan-400 bg-cyan-500/20" },
  team_member_added: { icon: "M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z", color: "text-indigo-400 bg-indigo-500/20" },
  secret_created: { icon: "M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z", color: "text-red-400 bg-red-500/20" },
  secret_revealed: { icon: "M15 12a3 3 0 11-6 0 3 3 0 016 0z", color: "text-amber-400 bg-amber-500/20" },
  vault_access: { icon: "M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z", color: "text-rose-400 bg-rose-500/20" },
  changelog_added: { icon: "M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z", color: "text-teal-400 bg-teal-500/20" },
  project_created: { icon: "M13 10V3L4 14h7v7l9-11h-7z", color: "text-yellow-400 bg-yellow-500/20" },
  default: { icon: "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z", color: "text-slate-400 bg-slate-500/20" },
};

const componentColors: Record<string, string> = {
  stt: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  rag: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  "tts-lipsync": "bg-rose-500/20 text-rose-400 border-rose-500/30",
  mcp: "bg-orange-500/20 text-orange-400 border-orange-500/30",
};

export default function ActivityPage() {
  const [activity, setActivity] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ component: "all", type: "all" });

  useEffect(() => {
    loadActivity();
  }, [filter]);

  const loadActivity = async () => {
    try {
      const filters: { component?: string; type?: string; limit: number } = { limit: 100 };
      if (filter.component !== "all") filters.component = filter.component;
      if (filter.type !== "all") filters.type = filter.type;
      const data = await getActivity(filters);
      setActivity(data);
    } catch (error) {
      console.error("Failed to load activity:", error);
    } finally {
      setLoading(false);
    }
  };

  const getIcon = (type: string) => activityIcons[type] || activityIcons.default;

  const groupByDate = (activities: Activity[]) => {
    const groups: Record<string, Activity[]> = {};
    activities.forEach((a) => {
      const date = new Date(a.created_at).toLocaleDateString("en-US", {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
      });
      if (!groups[date]) groups[date] = [];
      groups[date].push(a);
    });
    return groups;
  };

  const grouped = groupByDate(activity);

  return (
    <PageLayout
      title="Activity Feed"
      description="Track all project changes and updates"
      badges={[{ label: `${activity.length} Events` }]}
      gradientOrbs={[
        { color: "emerald", position: "top-1/4 -left-32" },
        { color: "teal", position: "bottom-1/4 -right-32" },
      ]}
    >
      {/* Filters */}
      <div className="flex items-center gap-4 mb-6">
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
        <Select value={filter.type} onValueChange={(v) => setFilter((p) => ({ ...p, type: v }))}>
          <SelectTrigger className="w-40 bg-slate-800 border-slate-700 text-slate-300">
            <SelectValue placeholder="Type" />
          </SelectTrigger>
          <SelectContent className="bg-slate-800 border-slate-700">
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="task_created">Tasks Created</SelectItem>
            <SelectItem value="task_status_changed">Status Changes</SelectItem>
            <SelectItem value="milestone_created">Milestones</SelectItem>
            <SelectItem value="decision_created">Decisions</SelectItem>
            <SelectItem value="team_member_added">Team Changes</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Activity Timeline */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Recent Activity</CardTitle>
          <CardDescription className="text-slate-400">
            All project events and changes
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[600px] pr-4">
            {loading ? (
              <div className="space-y-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="flex gap-4 animate-pulse">
                    <div className="w-10 h-10 rounded-full bg-slate-700" />
                    <div className="flex-1 space-y-2">
                      <div className="h-4 bg-slate-700 rounded w-3/4" />
                      <div className="h-3 bg-slate-700 rounded w-1/2" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-8">
                {Object.entries(grouped).map(([date, activities]) => (
                  <div key={date}>
                    <h3 className="text-sm font-medium text-slate-400 mb-4 sticky top-0 bg-slate-900/50 py-2">
                      {date}
                    </h3>
                    <div className="space-y-4 relative">
                      <div className="absolute left-5 top-0 bottom-0 w-px bg-slate-800" />
                      {activities.map((item) => {
                        const iconData = getIcon(item.type);
                        return (
                          <div key={item.id} className="flex gap-4 relative">
                            <div className={`w-10 h-10 rounded-full ${iconData.color} flex items-center justify-center z-10 shrink-0`}>
                              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={iconData.icon} />
                              </svg>
                            </div>
                            <div className="flex-1 bg-slate-800/30 rounded-lg p-4 hover:bg-slate-800/50 transition-colors">
                              <div className="flex items-start justify-between">
                                <div>
                                  <p className="text-white font-medium">{item.title}</p>
                                  {item.description && (
                                    <p className="text-sm text-slate-400 mt-1">{item.description}</p>
                                  )}
                                </div>
                                <span className="text-xs text-slate-500">
                                  {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                                </span>
                              </div>
                              <div className="flex items-center gap-2 mt-2">
                                {item.component && (
                                  <Badge className={componentColors[item.component] || "bg-slate-500/20 text-slate-400"}>
                                    {item.component === "tts-lipsync" ? "TTS" : item.component.toUpperCase()}
                                  </Badge>
                                )}
                                {item.user && (
                                  <span className="text-xs text-slate-500">by {item.user}</span>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                {activity.length === 0 && (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
                      <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium text-white mb-2">No activity yet</h3>
                    <p className="text-slate-400">Project events will appear here as they happen</p>
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
