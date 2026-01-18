"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ExternalLink,
  Maximize2,
  Minimize2,
  RefreshCw,
  Monitor,
  Loader2,
  CheckCircle,
  PlusCircle,
  Edit3,
  Trash2,
  Database,
  User,
  Building2,
  Mail,
  Phone,
  Globe,
  LayoutDashboard,
  Table2,
  Code,
  Search,
  FileJson,
  ChevronDown,
  ChevronRight,
  ArrowUpDown,
  Download,
  Copy,
  Play,
  AlertCircle,
  HardDrive,
  Users,
  Briefcase,
  TrendingUp,
  Clock,
  Zap,
  Tag,
  Link2,
  Eye,
  X,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SalesforceConnectionStatus, SalesforceOperation } from "./types";

interface SalesforceEmbedProps {
  connectionStatus: SalesforceConnectionStatus;
  screenshotUrl?: string;
  onRefresh?: () => void;
  operations?: SalesforceOperation[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SalesforceRecord = Record<string, any>;

interface DashboardData {
  org: {
    id: string;
    name: string;
    type: string;
    isSandbox: boolean;
    instance: string;
    locale: string;
    instanceUrl: string;
  };
  stats: {
    activeUsers: number;
    recordCounts: Record<string, number>;
  };
  limits: {
    apiRequests: { Max: number; Remaining: number };
    dataStorage: { Max: number; Remaining: number };
    fileStorage: { Max: number; Remaining: number };
  };
  recent: {
    accounts: SalesforceRecord[];
    contacts: SalesforceRecord[];
  };
}

interface ObjectInfo {
  name: string;
  label: string;
  custom: boolean;
  queryable: boolean;
}

interface FieldInfo {
  name: string;
  label: string;
  type: string;
  length?: number;
  required: boolean;
  unique: boolean;
  updateable: boolean;
  createable: boolean;
  custom: boolean;
  picklistValues?: { value: string; label: string; active: boolean }[];
  referenceTo?: string[];
  relationshipName?: string;
}

interface ObjectMetadata {
  object: {
    name: string;
    label: string;
    labelPlural: string;
    keyPrefix: string;
    custom: boolean;
    createable: boolean;
    updateable: boolean;
    deletable: boolean;
    queryable: boolean;
    searchable: boolean;
  };
  fields: FieldInfo[];
  relationships: FieldInfo[];
  childRelationships: { name: string; childObject: string; field: string }[];
  fieldCount: number;
  relationshipCount: number;
}

type TabType = "dashboard" | "results" | "explorer" | "console" | "inspector";

export function SalesforceEmbed({
  connectionStatus,
  screenshotUrl,
  onRefresh,
  operations = [],
}: SalesforceEmbedProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>("dashboard");

  // Dashboard state
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);

  // Object Explorer state
  const [objects, setObjects] = useState<ObjectInfo[]>([]);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [objectMetadata, setObjectMetadata] = useState<ObjectMetadata | null>(null);
  const [objectsLoading, setObjectsLoading] = useState(false);
  const [fieldSearch, setFieldSearch] = useState("");

  // SOQL Console state
  const [soqlQuery, setSoqlQuery] = useState("SELECT Id, Name FROM Account LIMIT 10");
  const [queryResult, setQueryResult] = useState<{ records: SalesforceRecord[]; totalSize: number } | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);

  // Record Inspector state
  const [inspectedRecord, setInspectedRecord] = useState<{ object: string; id: string; fields: { name: string; label: string; type: string; value: unknown }[] } | null>(null);
  const [inspectorLoading, setInspectorLoading] = useState(false);

  // Data table state
  const [sortField, setSortField] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  const handleRefresh = async () => {
    if (onRefresh) {
      setIsLoading(true);
      await onRefresh();
      setIsLoading(false);
    }
  };

  const handleOpenInNewTab = () => {
    if (connectionStatus.org?.instanceUrl) {
      window.open(connectionStatus.org.instanceUrl, "_blank");
    }
  };

  // Fetch dashboard data
  const fetchDashboard = useCallback(async () => {
    if (!connectionStatus.connected) return;
    setDashboardLoading(true);
    try {
      const response = await fetch("/api/salesforce/mcp/dashboard");
      const data = await response.json();
      if (data.success) {
        setDashboardData(data);
      }
    } catch (error) {
      console.error("Failed to fetch dashboard:", error);
    } finally {
      setDashboardLoading(false);
    }
  }, [connectionStatus.connected]);

  // Fetch objects list
  const fetchObjects = useCallback(async () => {
    if (!connectionStatus.connected) return;
    setObjectsLoading(true);
    try {
      const response = await fetch("/api/salesforce/mcp/objects");
      const data = await response.json();
      if (data.success) {
        setObjects(data.objects);
      }
    } catch (error) {
      console.error("Failed to fetch objects:", error);
    } finally {
      setObjectsLoading(false);
    }
  }, [connectionStatus.connected]);

  // Fetch object metadata
  const fetchObjectMetadata = useCallback(async (objectName: string) => {
    setObjectsLoading(true);
    try {
      const response = await fetch(`/api/salesforce/mcp/describe/${objectName}/full`);
      const data = await response.json();
      if (data.success) {
        setObjectMetadata(data);
      }
    } catch (error) {
      console.error("Failed to fetch object metadata:", error);
    } finally {
      setObjectsLoading(false);
    }
  }, []);

  // Execute SOQL query
  const executeQuery = async () => {
    setQueryLoading(true);
    setQueryError(null);
    try {
      const response = await fetch("/api/salesforce/mcp/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ soql: soqlQuery }),
      });
      const data = await response.json();
      if (data.success) {
        setQueryResult({ records: data.records, totalSize: data.totalSize });
        setActiveTab("results");
      } else {
        setQueryError(data.error || "Query failed");
      }
    } catch (error) {
      setQueryError(error instanceof Error ? error.message : "Query failed");
    } finally {
      setQueryLoading(false);
    }
  };

  // Fetch record for inspector
  const fetchRecord = async (objectName: string, recordId: string) => {
    setInspectorLoading(true);
    try {
      const response = await fetch(`/api/salesforce/mcp/record/${objectName}/${recordId}`);
      const data = await response.json();
      if (data.success) {
        setInspectedRecord({
          object: data.object,
          id: data.id,
          fields: data.fields,
        });
        setActiveTab("inspector");
      }
    } catch (error) {
      console.error("Failed to fetch record:", error);
    } finally {
      setInspectorLoading(false);
    }
  };

  // Load data on tab change
  useEffect(() => {
    if (connectionStatus.connected) {
      if (activeTab === "dashboard" && !dashboardData) {
        fetchDashboard();
      } else if (activeTab === "explorer" && objects.length === 0) {
        fetchObjects();
      }
    }
  }, [activeTab, connectionStatus.connected, dashboardData, objects.length, fetchDashboard, fetchObjects]);

  // Load object metadata when selected
  useEffect(() => {
    if (selectedObject) {
      fetchObjectMetadata(selectedObject);
    }
  }, [selectedObject, fetchObjectMetadata]);

  // Extract data from operations
  const displayData = useMemo(() => {
    const queryResults: { object: string; records: SalesforceRecord[]; soql?: string }[] = [];
    const recentChanges: { type: string; object: string; id: string; details: SalesforceRecord; timestamp: number }[] = [];

    operations.forEach((op) => {
      if (op.status !== "success" || !op.result) return;

      const result = op.result as SalesforceRecord;

      if (op.type === "query" && result.records) {
        queryResults.push({
          object: result.object || "Records",
          records: result.records,
          soql: result.soql,
        });
      } else if (op.type === "insert" || op.type === "update" || op.type === "delete") {
        recentChanges.push({
          type: op.type,
          object: result.object || "Record",
          id: result.id || "",
          details: result,
          timestamp: op.timestamp,
        });
      }
    });

    return { queryResults, recentChanges };
  }, [operations]);

  // Sort records
  const sortedRecords = useMemo(() => {
    const records = queryResult?.records || displayData.queryResults[0]?.records || [];
    if (!sortField) return records;

    return [...records].sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      const result = aVal < bVal ? -1 : 1;
      return sortDirection === "asc" ? result : -result;
    });
  }, [queryResult?.records, displayData.queryResults, sortField, sortDirection]);

  // Get icon for object type
  const getObjectIcon = (objectName: string) => {
    const name = objectName.toLowerCase();
    if (name.includes("account")) return <Building2 className="w-4 h-4 text-[#0176D3]" />;
    if (name.includes("contact")) return <User className="w-4 h-4 text-purple-400" />;
    if (name.includes("lead")) return <User className="w-4 h-4 text-amber-400" />;
    if (name.includes("opportunity")) return <TrendingUp className="w-4 h-4 text-emerald-400" />;
    if (name.includes("case")) return <Briefcase className="w-4 h-4 text-orange-400" />;
    return <Database className="w-4 h-4 text-slate-400" />;
  };

  // Get field type badge color
  const getFieldTypeBadge = (type: string) => {
    const colors: Record<string, string> = {
      string: "bg-blue-500",
      id: "bg-purple-500",
      reference: "bg-emerald-500",
      boolean: "bg-amber-500",
      datetime: "bg-cyan-500",
      date: "bg-cyan-400",
      double: "bg-pink-500",
      currency: "bg-green-500",
      picklist: "bg-indigo-500",
      email: "bg-red-400",
      phone: "bg-orange-400",
      url: "bg-blue-400",
      textarea: "bg-slate-500",
    };
    return colors[type] || "bg-slate-400";
  };

  // Export to CSV
  const exportToCSV = () => {
    const records = sortedRecords;
    if (records.length === 0) return;

    const headers = Object.keys(records[0]).filter((k) => k !== "attributes");
    const csv = [
      headers.join(","),
      ...records.map((r) => headers.map((h) => JSON.stringify(r[h] ?? "")).join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "salesforce_export.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Copy SOQL to clipboard
  const copySOQL = () => {
    navigator.clipboard.writeText(soqlQuery);
  };

  const hasData = displayData.queryResults.length > 0 || displayData.recentChanges.length > 0;

  const tabs = [
    { id: "dashboard" as TabType, label: "Dashboard", icon: LayoutDashboard },
    { id: "results" as TabType, label: "Results", icon: Table2, count: queryResult?.totalSize || displayData.queryResults.reduce((s, q) => s + q.records.length, 0) },
    { id: "explorer" as TabType, label: "Objects", icon: Database },
    { id: "console" as TabType, label: "SOQL", icon: Code },
    { id: "inspector" as TabType, label: "Inspector", icon: Eye, hidden: !inspectedRecord },
  ];

  return (
    <Card
      className={`bg-slate-800/50 border-white/10 backdrop-blur-sm overflow-hidden ${
        isFullscreen ? "fixed inset-4 z-50" : ""
      }`}
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <Monitor className="w-4 h-4 text-[#0176D3]" />
          Salesforce View
          <div className="ml-auto flex items-center gap-2">
            {connectionStatus.connected && connectionStatus.org && (
              <Badge variant="outline" className="text-[10px] border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse" />
                {connectionStatus.org.name}
              </Badge>
            )}
            <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={!connectionStatus.connected || isLoading} className="h-7 w-7">
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleOpenInNewTab} disabled={!connectionStatus.connected} className="h-7 w-7">
              <ExternalLink className="w-3.5 h-3.5" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setIsFullscreen(!isFullscreen)} className="h-7 w-7">
              {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className={`relative bg-slate-900 ${isFullscreen ? "h-[calc(100vh-8rem)]" : "h-[400px]"}`}>
          {!connectionStatus.connected ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="w-16 h-16 rounded-lg bg-slate-800 flex items-center justify-center mb-3">
                <Monitor className="w-8 h-8 text-slate-600" />
              </div>
              <p className="text-slate-500 text-sm">Salesforce not connected</p>
              <p className="text-slate-600 text-xs mt-1">Connect to view your org</p>
            </div>
          ) : (
            <div className="absolute inset-0 flex flex-col">
              {/* Tab Navigation */}
              <div className="h-10 bg-[#032D60] flex items-center px-2 gap-1 border-b border-white/10">
                {tabs.filter((t) => !t.hidden).map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`px-3 py-1.5 text-xs rounded flex items-center gap-1.5 transition-all ${
                        activeTab === tab.id
                          ? "bg-[#0176D3] text-white font-medium shadow-lg"
                          : "text-white/70 hover:text-white hover:bg-white/10"
                      }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      {tab.label}
                      {tab.count !== undefined && tab.count > 0 && (
                        <Badge className="ml-1 h-4 px-1 text-[10px] bg-white/20">{tab.count}</Badge>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Content Area */}
              <div className="flex-1 overflow-hidden">
                <AnimatePresence mode="wait">
                  {/* Dashboard Tab */}
                  {activeTab === "dashboard" && (
                    <motion.div
                      key="dashboard"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="h-full overflow-auto p-4 bg-gradient-to-br from-slate-900 to-slate-800"
                    >
                      {dashboardLoading ? (
                        <div className="flex items-center justify-center h-full">
                          <Loader2 className="w-8 h-8 text-[#0176D3] animate-spin" />
                        </div>
                      ) : dashboardData ? (
                        <div className="space-y-4">
                          {/* Org Info Header */}
                          <div className="bg-gradient-to-r from-[#0176D3] to-[#032D60] rounded-lg p-4">
                            <div className="flex items-center gap-3">
                              <div className="w-12 h-12 rounded-lg bg-white/20 flex items-center justify-center">
                                <Database className="w-6 h-6 text-white" />
                              </div>
                              <div>
                                <h3 className="text-white font-semibold text-lg">{dashboardData.org.name}</h3>
                                <p className="text-white/70 text-sm">{dashboardData.org.type} • {dashboardData.org.instance}</p>
                              </div>
                              {dashboardData.org.isSandbox && (
                                <Badge className="ml-auto bg-amber-500 text-white">Sandbox</Badge>
                              )}
                            </div>
                          </div>

                          {/* Stats Grid */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="bg-slate-800/50 rounded-lg p-3 border border-white/5">
                              <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                                <Users className="w-3.5 h-3.5" />
                                Active Users
                              </div>
                              <p className="text-2xl font-bold text-white">{dashboardData.stats.activeUsers}</p>
                            </div>
                            {Object.entries(dashboardData.stats.recordCounts).slice(0, 3).map(([obj, count]) => (
                              <div key={obj} className="bg-slate-800/50 rounded-lg p-3 border border-white/5">
                                <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
                                  {getObjectIcon(obj)}
                                  {obj}s
                                </div>
                                <p className="text-2xl font-bold text-white">{count.toLocaleString()}</p>
                              </div>
                            ))}
                          </div>

                          {/* Limits */}
                          <div className="bg-slate-800/50 rounded-lg p-4 border border-white/5">
                            <h4 className="text-white font-medium text-sm mb-3 flex items-center gap-2">
                              <Zap className="w-4 h-4 text-amber-400" />
                              API & Storage Limits
                            </h4>
                            <div className="space-y-3">
                              {[
                                { label: "API Requests", data: dashboardData.limits.apiRequests, icon: Globe },
                                { label: "Data Storage", data: dashboardData.limits.dataStorage, icon: HardDrive, unit: "MB" },
                                { label: "File Storage", data: dashboardData.limits.fileStorage, icon: FileJson, unit: "MB" },
                              ].map((limit) => {
                                const used = limit.data.Max - limit.data.Remaining;
                                const percentage = (used / limit.data.Max) * 100;
                                return (
                                  <div key={limit.label}>
                                    <div className="flex items-center justify-between text-xs mb-1">
                                      <span className="text-slate-400 flex items-center gap-1.5">
                                        <limit.icon className="w-3.5 h-3.5" />
                                        {limit.label}
                                      </span>
                                      <span className="text-white">
                                        {used.toLocaleString()} / {limit.data.Max.toLocaleString()} {limit.unit || ""}
                                      </span>
                                    </div>
                                    <Progress value={percentage} className="h-2 bg-slate-700" />
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          {/* Recent Records */}
                          <div className="grid grid-cols-2 gap-3">
                            <div className="bg-slate-800/50 rounded-lg p-3 border border-white/5">
                              <h4 className="text-white font-medium text-xs mb-2 flex items-center gap-2">
                                <Building2 className="w-3.5 h-3.5 text-[#0176D3]" />
                                Recent Accounts
                              </h4>
                              <div className="space-y-1">
                                {dashboardData.recent.accounts.slice(0, 3).map((acc) => (
                                  <div key={acc.Id} className="text-xs text-slate-300 truncate hover:text-white cursor-pointer" onClick={() => fetchRecord("Account", acc.Id)}>
                                    {acc.Name}
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-3 border border-white/5">
                              <h4 className="text-white font-medium text-xs mb-2 flex items-center gap-2">
                                <User className="w-3.5 h-3.5 text-purple-400" />
                                Recent Contacts
                              </h4>
                              <div className="space-y-1">
                                {dashboardData.recent.contacts.slice(0, 3).map((con) => (
                                  <div key={con.Id} className="text-xs text-slate-300 truncate hover:text-white cursor-pointer" onClick={() => fetchRecord("Contact", con.Id)}>
                                    {con.FirstName} {con.LastName}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <Button onClick={fetchDashboard} variant="outline" size="sm">
                            <RefreshCw className="w-4 h-4 mr-2" />
                            Load Dashboard
                          </Button>
                        </div>
                      )}
                    </motion.div>
                  )}

                  {/* Results Tab */}
                  {activeTab === "results" && (
                    <motion.div
                      key="results"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="h-full flex flex-col"
                    >
                      {/* Toolbar */}
                      <div className="h-10 bg-slate-800 border-b border-white/10 flex items-center px-3 gap-2">
                        <span className="text-xs text-slate-400">
                          {sortedRecords.length} records
                          {displayData.recentChanges.length > 0 && (
                            <span className="ml-2 text-emerald-400">
                              • {displayData.recentChanges.length} changes
                            </span>
                          )}
                        </span>
                        <div className="flex-1" />
                        <Button variant="ghost" size="sm" onClick={exportToCSV} className="h-7 text-xs">
                          <Download className="w-3.5 h-3.5 mr-1" />
                          Export CSV
                        </Button>
                      </div>

                      {/* Data Table */}
                      <div className="flex-1 overflow-auto">
                        {/* Recent Changes Section */}
                        {displayData.recentChanges.length > 0 && (
                          <div className="p-3 border-b border-white/10 bg-emerald-500/5">
                            <h4 className="text-xs font-medium text-emerald-400 mb-2 flex items-center gap-2">
                              <CheckCircle className="w-3.5 h-3.5" />
                              Recent Operations
                            </h4>
                            <div className="space-y-2">
                              {displayData.recentChanges.slice(0, 5).map((change, idx) => (
                                <div
                                  key={`change-${idx}`}
                                  className="bg-slate-800/50 rounded-lg p-2 border border-emerald-500/20"
                                >
                                  <div className="flex items-center gap-2">
                                    <Badge
                                      className={`text-[9px] ${
                                        change.type === "insert"
                                          ? "bg-emerald-500/20 text-emerald-400"
                                          : change.type === "update"
                                          ? "bg-amber-500/20 text-amber-400"
                                          : "bg-red-500/20 text-red-400"
                                      }`}
                                    >
                                      {change.type === "insert" ? (
                                        <PlusCircle className="w-2.5 h-2.5 mr-1" />
                                      ) : change.type === "update" ? (
                                        <Edit3 className="w-2.5 h-2.5 mr-1" />
                                      ) : (
                                        <Trash2 className="w-2.5 h-2.5 mr-1" />
                                      )}
                                      {change.type.toUpperCase()}
                                    </Badge>
                                    <span className="text-xs text-white font-medium">{change.object}</span>
                                    <span className="text-[10px] text-slate-400 font-mono">{change.id}</span>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="ml-auto h-5 text-[10px] px-2"
                                      onClick={() => fetchRecord(change.object, change.id)}
                                    >
                                      <Eye className="w-3 h-3 mr-1" />
                                      View
                                    </Button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Query Results */}
                        {sortedRecords.length > 0 ? (
                          <table className="w-full text-xs">
                            <thead className="bg-slate-800 sticky top-0">
                              <tr>
                                {Object.keys(sortedRecords[0])
                                  .filter((k) => k !== "attributes")
                                  .slice(0, 8)
                                  .map((key) => (
                                    <th
                                      key={key}
                                      className="px-3 py-2 text-left text-slate-400 font-medium cursor-pointer hover:bg-slate-700"
                                      onClick={() => {
                                        if (sortField === key) {
                                          setSortDirection(sortDirection === "asc" ? "desc" : "asc");
                                        } else {
                                          setSortField(key);
                                          setSortDirection("asc");
                                        }
                                      }}
                                    >
                                      <div className="flex items-center gap-1">
                                        {key}
                                        {sortField === key && (
                                          <ArrowUpDown className="w-3 h-3" />
                                        )}
                                      </div>
                                    </th>
                                  ))}
                                <th className="px-3 py-2 text-left text-slate-400 font-medium w-20">Actions</th>
                              </tr>
                            </thead>
                            <tbody>
                              {sortedRecords.map((record, idx) => (
                                <tr key={record.Id || idx} className="border-b border-white/5 hover:bg-slate-800/50">
                                  {Object.entries(record)
                                    .filter(([k]) => k !== "attributes")
                                    .slice(0, 8)
                                    .map(([key, value]) => (
                                      <td key={key} className="px-3 py-2 text-slate-300 truncate max-w-[200px]">
                                        {typeof value === "object" ? JSON.stringify(value) : String(value ?? "")}
                                      </td>
                                    ))}
                                  <td className="px-3 py-2">
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="h-6 w-6 p-0"
                                      onClick={() => {
                                        const objType = record.attributes?.type || "Account";
                                        fetchRecord(objType, record.Id);
                                      }}
                                    >
                                      <Eye className="w-3.5 h-3.5" />
                                    </Button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : displayData.recentChanges.length === 0 ? (
                          <div className="flex flex-col items-center justify-center h-full text-slate-500">
                            <Table2 className="w-12 h-12 mb-3 opacity-50" />
                            <p className="text-sm">No query results yet</p>
                            <p className="text-xs mt-1">Use the SOQL tab to run a query</p>
                          </div>
                        ) : null}
                      </div>
                    </motion.div>
                  )}

                  {/* Object Explorer Tab */}
                  {activeTab === "explorer" && (
                    <motion.div
                      key="explorer"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="h-full flex"
                    >
                      {/* Object List */}
                      <div className="w-48 bg-slate-800/50 border-r border-white/10 overflow-auto">
                        <div className="p-2 border-b border-white/10">
                          <div className="relative">
                            <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-slate-500" />
                            <input
                              type="text"
                              placeholder="Search objects..."
                              className="w-full bg-slate-900 border border-white/10 rounded px-2 py-1 pl-7 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-[#0176D3]"
                              value={fieldSearch}
                              onChange={(e) => setFieldSearch(e.target.value)}
                            />
                          </div>
                        </div>
                        <div className="p-1">
                          {objectsLoading && objects.length === 0 ? (
                            <div className="flex items-center justify-center py-8">
                              <Loader2 className="w-5 h-5 text-[#0176D3] animate-spin" />
                            </div>
                          ) : (
                            objects
                              .filter((o) => o.name.toLowerCase().includes(fieldSearch.toLowerCase()) || o.label.toLowerCase().includes(fieldSearch.toLowerCase()))
                              .slice(0, 50)
                              .map((obj) => (
                                <button
                                  key={obj.name}
                                  onClick={() => setSelectedObject(obj.name)}
                                  className={`w-full px-2 py-1.5 text-left text-xs rounded flex items-center gap-2 ${
                                    selectedObject === obj.name
                                      ? "bg-[#0176D3] text-white"
                                      : "text-slate-300 hover:bg-white/5"
                                  }`}
                                >
                                  {getObjectIcon(obj.name)}
                                  <span className="truncate">{obj.label}</span>
                                  {obj.custom && <Badge className="h-4 px-1 text-[8px] bg-purple-500 ml-auto">Custom</Badge>}
                                </button>
                              ))
                          )}
                        </div>
                      </div>

                      {/* Object Details */}
                      <div className="flex-1 overflow-auto p-4">
                        {selectedObject && objectMetadata ? (
                          <div className="space-y-4">
                            {/* Object Header */}
                            <div className="bg-slate-800/50 rounded-lg p-4 border border-white/5">
                              <div className="flex items-center gap-3">
                                {getObjectIcon(objectMetadata.object.name)}
                                <div>
                                  <h3 className="text-white font-semibold">{objectMetadata.object.label}</h3>
                                  <p className="text-slate-400 text-xs">{objectMetadata.object.name} • {objectMetadata.fieldCount} fields</p>
                                </div>
                                <div className="ml-auto flex gap-1">
                                  {objectMetadata.object.createable && <Badge className="bg-emerald-500 text-[10px]">Createable</Badge>}
                                  {objectMetadata.object.updateable && <Badge className="bg-blue-500 text-[10px]">Updateable</Badge>}
                                  {objectMetadata.object.deletable && <Badge className="bg-red-500 text-[10px]">Deletable</Badge>}
                                </div>
                              </div>
                            </div>

                            {/* Fields */}
                            <div className="bg-slate-800/50 rounded-lg border border-white/5 overflow-hidden">
                              <div className="px-4 py-2 bg-slate-800 border-b border-white/10">
                                <h4 className="text-white font-medium text-sm">Fields ({objectMetadata.fields.length})</h4>
                              </div>
                              <div className="max-h-48 overflow-auto">
                                {objectMetadata.fields.map((field) => (
                                  <div key={field.name} className="px-4 py-2 border-b border-white/5 flex items-center gap-3 hover:bg-white/5">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-white text-xs font-medium truncate">{field.label}</p>
                                      <p className="text-slate-500 text-[10px]">{field.name}</p>
                                    </div>
                                    <Badge className={`${getFieldTypeBadge(field.type)} text-[10px]`}>{field.type}</Badge>
                                    {field.required && <Badge className="bg-red-500/20 text-red-400 text-[10px]">Required</Badge>}
                                    {field.custom && <Tag className="w-3 h-3 text-purple-400" />}
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Relationships */}
                            {objectMetadata.relationships.length > 0 && (
                              <div className="bg-slate-800/50 rounded-lg border border-white/5 overflow-hidden">
                                <div className="px-4 py-2 bg-slate-800 border-b border-white/10">
                                  <h4 className="text-white font-medium text-sm flex items-center gap-2">
                                    <Link2 className="w-4 h-4 text-emerald-400" />
                                    Relationships ({objectMetadata.relationships.length})
                                  </h4>
                                </div>
                                <div className="max-h-32 overflow-auto">
                                  {objectMetadata.relationships.map((rel) => (
                                    <div key={rel.name} className="px-4 py-2 border-b border-white/5 flex items-center gap-3 hover:bg-white/5">
                                      <div className="flex-1">
                                        <p className="text-white text-xs">{rel.label}</p>
                                        <p className="text-slate-500 text-[10px]">→ {rel.referenceTo?.join(", ")}</p>
                                      </div>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="h-6 text-[10px]"
                                        onClick={() => setSelectedObject(rel.referenceTo?.[0] || "")}
                                      >
                                        View
                                      </Button>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="flex flex-col items-center justify-center h-full text-slate-500">
                            <Database className="w-12 h-12 mb-3 opacity-50" />
                            <p className="text-sm">Select an object to view details</p>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}

                  {/* SOQL Console Tab */}
                  {activeTab === "console" && (
                    <motion.div
                      key="console"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="h-full flex flex-col p-4"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <Code className="w-4 h-4 text-[#0176D3]" />
                        <span className="text-white font-medium text-sm">SOQL Query Console</span>
                      </div>

                      {/* Query Editor */}
                      <div className="relative flex-1 min-h-[120px] mb-3">
                        <textarea
                          value={soqlQuery}
                          onChange={(e) => setSoqlQuery(e.target.value)}
                          className="w-full h-full bg-slate-800 border border-white/10 rounded-lg p-3 text-sm text-white font-mono resize-none focus:outline-none focus:border-[#0176D3]"
                          placeholder="SELECT Id, Name FROM Account LIMIT 10"
                          spellCheck={false}
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          className="absolute top-2 right-2 h-7"
                          onClick={copySOQL}
                        >
                          <Copy className="w-3.5 h-3.5" />
                        </Button>
                      </div>

                      {/* Error Message */}
                      {queryError && (
                        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-3 mb-3 flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 text-red-400 mt-0.5" />
                          <p className="text-red-400 text-xs flex-1">{queryError}</p>
                          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setQueryError(null)}>
                            <X className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      )}

                      {/* Execute Button */}
                      <div className="flex items-center gap-2">
                        <Button
                          onClick={executeQuery}
                          disabled={queryLoading || !soqlQuery.trim()}
                          className="bg-[#0176D3] hover:bg-[#0176D3]/80"
                        >
                          {queryLoading ? (
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          ) : (
                            <Play className="w-4 h-4 mr-2" />
                          )}
                          Execute Query
                        </Button>
                        {queryResult && (
                          <span className="text-slate-400 text-xs">
                            Last query returned {queryResult.totalSize} records
                          </span>
                        )}
                      </div>

                      {/* Quick Queries */}
                      <div className="mt-4">
                        <p className="text-slate-400 text-xs mb-2">Quick Queries:</p>
                        <div className="flex flex-wrap gap-2">
                          {[
                            "SELECT Id, Name FROM Account LIMIT 10",
                            "SELECT Id, FirstName, LastName, Email FROM Contact LIMIT 10",
                            "SELECT Id, Name, Amount, StageName FROM Opportunity LIMIT 10",
                            "SELECT Id, Subject, Status FROM Case LIMIT 10",
                          ].map((query, idx) => (
                            <Button
                              key={idx}
                              variant="outline"
                              size="sm"
                              className="text-[10px] h-6"
                              onClick={() => setSoqlQuery(query)}
                            >
                              {query.split(" FROM ")[1]?.split(" ")[0] || "Query"}
                            </Button>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* Record Inspector Tab */}
                  {activeTab === "inspector" && inspectedRecord && (
                    <motion.div
                      key="inspector"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="h-full overflow-auto p-4"
                    >
                      {inspectorLoading ? (
                        <div className="flex items-center justify-center h-full">
                          <Loader2 className="w-8 h-8 text-[#0176D3] animate-spin" />
                        </div>
                      ) : (
                        <div className="space-y-4">
                          {/* Record Header */}
                          <div className="bg-gradient-to-r from-[#0176D3] to-[#032D60] rounded-lg p-4">
                            <div className="flex items-center gap-3">
                              {getObjectIcon(inspectedRecord.object)}
                              <div>
                                <h3 className="text-white font-semibold">{inspectedRecord.object} Record</h3>
                                <p className="text-white/70 text-xs font-mono">{inspectedRecord.id}</p>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="ml-auto text-white/70 hover:text-white"
                                onClick={() => {
                                  window.open(`${connectionStatus.org?.instanceUrl}/${inspectedRecord.id}`, "_blank");
                                }}
                              >
                                <ExternalLink className="w-4 h-4 mr-1" />
                                View in Salesforce
                              </Button>
                            </div>
                          </div>

                          {/* Field Values */}
                          <div className="bg-slate-800/50 rounded-lg border border-white/5 overflow-hidden">
                            <div className="px-4 py-2 bg-slate-800 border-b border-white/10">
                              <h4 className="text-white font-medium text-sm">Field Values</h4>
                            </div>
                            <div className="divide-y divide-white/5">
                              {inspectedRecord.fields
                                .filter((f) => f.value !== null && f.value !== undefined)
                                .map((field) => (
                                  <div key={field.name} className="px-4 py-3 flex items-start gap-4 hover:bg-white/5">
                                    <div className="w-1/3 min-w-0">
                                      <p className="text-slate-400 text-xs truncate">{field.label}</p>
                                      <p className="text-slate-600 text-[10px] font-mono">{field.name}</p>
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-white text-sm break-words">
                                        {typeof field.value === "object"
                                          ? JSON.stringify(field.value, null, 2)
                                          : String(field.value)}
                                      </p>
                                    </div>
                                    <Badge className={`${getFieldTypeBadge(field.type)} text-[10px] shrink-0`}>
                                      {field.type}
                                    </Badge>
                                  </div>
                                ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Loading Overlay */}
              {isLoading && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <Loader2 className="w-8 h-8 text-white animate-spin" />
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
