"use client";

import { useState, useEffect } from "react";
import PageLayout from "@/components/page-layout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  authenticateVault,
  getSecrets,
  revealSecret,
  createSecret,
  deleteSecret,
  getVaultToken,
  clearVaultToken,
  type Secret,
} from "@/lib/api";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const categoryColors: Record<string, { bg: string; text: string }> = {
  api_key: { bg: "bg-cyan-500/20", text: "text-cyan-400" },
  token: { bg: "bg-purple-500/20", text: "text-purple-400" },
  credential: { bg: "bg-amber-500/20", text: "text-amber-400" },
  env: { bg: "bg-emerald-500/20", text: "text-emerald-400" },
  other: { bg: "bg-slate-500/20", text: "text-slate-400" },
};

export default function SecretsPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [secrets, setSecrets] = useState<Secret[]>([]);
  const [loading, setLoading] = useState(false);
  const [revealedSecrets, setRevealedSecrets] = useState<Record<number, string>>({});
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [newSecret, setNewSecret] = useState({
    name: "",
    category: "api_key",
    value: "",
    description: "",
  });

  useEffect(() => {
    const token = getVaultToken();
    if (token) {
      // Verify token is still valid by trying to load secrets
      getSecrets()
        .then((data) => {
          setIsAuthenticated(true);
          setSecrets(data);
        })
        .catch(() => {
          // Token is expired or invalid, clear it
          clearVaultToken();
          setIsAuthenticated(false);
        });
    }
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await authenticateVault(password);
      setIsAuthenticated(true);
      await loadSecrets();
    } catch {
      setError("Invalid password");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    clearVaultToken();
    setIsAuthenticated(false);
    setSecrets([]);
    setRevealedSecrets({});
    setPassword("");
  };

  const loadSecrets = async () => {
    try {
      const data = await getSecrets();
      setSecrets(data);
      setError("");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load secrets";
      if (message.includes("token") || message.includes("unauthorized")) {
        // Token expired or invalid - log out
        clearVaultToken();
        setIsAuthenticated(false);
        setSecrets([]);
      } else {
        setError(message);
      }
    }
  };

  const handleReveal = async (id: number) => {
    if (revealedSecrets[id]) {
      setRevealedSecrets((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      return;
    }

    try {
      const { value } = await revealSecret(id);
      setRevealedSecrets((prev) => ({ ...prev, [id]: value }));
    } catch {
      setError("Failed to reveal secret");
    }
  };

  const handleCopy = async (id: number) => {
    try {
      // Always fetch the real value to copy
      let value = revealedSecrets[id];
      if (!value) {
        const result = await revealSecret(id);
        value = result.value;
      }

      // Try modern clipboard API first, fallback to legacy method
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
      } else {
        // Fallback for HTTP sites
        const textArea = document.createElement("textarea");
        textArea.value = value;
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        textArea.style.top = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand("copy");
        textArea.remove();
      }

      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      setError("Failed to copy secret");
    }
  };

  const handleAddSecret = async () => {
    try {
      await createSecret(newSecret);
      setAddDialogOpen(false);
      setNewSecret({ name: "", category: "api_key", value: "", description: "" });
      await loadSecrets();
    } catch {
      setError("Failed to create secret");
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this secret?")) return;
    try {
      await deleteSecret(id);
      await loadSecrets();
    } catch {
      setError("Failed to delete secret");
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        {/* Background Effects */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 -left-32 w-96 h-96 bg-red-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl" />
        </div>

        <Card className="w-full max-w-md bg-slate-900/80 border-slate-800 backdrop-blur-xl relative z-10">
          <CardHeader className="text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-gradient-to-br from-red-500 to-amber-600 flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <CardTitle className="text-2xl text-white">Secrets Vault</CardTitle>
            <CardDescription className="text-slate-400">
              Enter password to access project secrets
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="password" className="text-slate-300">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="Enter vault password"
                  autoFocus
                />
              </div>
              {error && (
                <p className="text-red-400 text-sm">{error}</p>
              )}
              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-red-500 to-amber-600 text-white hover:opacity-90"
                disabled={loading}
              >
                {loading ? "Authenticating..." : "Unlock Vault"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Navigation Dock */}
        <PresentationDock items={dockItems} />
      </div>
    );
  }

  return (
    <PageLayout
      title="Secrets Vault"
      description="Secure storage for API keys, tokens, and credentials"
      badges={[{ label: `${secrets.length} Secrets` }]}
      gradientOrbs={[
        { color: "red", position: "top-1/4 -left-32" },
        { color: "amber", position: "bottom-1/4 -right-32" },
      ]}
      actions={
        <Button
          variant="outline"
          size="sm"
          onClick={handleLogout}
          className="bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700"
        >
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
          Lock Vault
        </Button>
      }
    >
      {/* Actions Bar */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Vault Unlocked
          </div>
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-red-500 to-amber-600 text-white hover:opacity-90">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Secret
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-slate-900 border-slate-700">
            <DialogHeader>
              <DialogTitle className="text-white">Add New Secret</DialogTitle>
              <DialogDescription className="text-slate-400">
                Store a new secret securely in the vault
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-slate-300">Name</Label>
                <Input
                  value={newSecret.name}
                  onChange={(e) => setNewSecret((p) => ({ ...p, name: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="e.g., ELEVENLABS_API_KEY"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Category</Label>
                <Select
                  value={newSecret.category}
                  onValueChange={(v) => setNewSecret((p) => ({ ...p, category: v }))}
                >
                  <SelectTrigger className="bg-slate-800 border-slate-700 text-slate-300">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700">
                    <SelectItem value="api_key">API Key</SelectItem>
                    <SelectItem value="token">Token</SelectItem>
                    <SelectItem value="credential">Credential</SelectItem>
                    <SelectItem value="env">Environment Variable</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Value</Label>
                <Textarea
                  value={newSecret.value}
                  onChange={(e) => setNewSecret((p) => ({ ...p, value: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white font-mono"
                  placeholder="Secret value"
                  rows={3}
                />
              </div>
              <div className="space-y-2">
                <Label className="text-slate-300">Description (optional)</Label>
                <Input
                  value={newSecret.description}
                  onChange={(e) => setNewSecret((p) => ({ ...p, description: e.target.value }))}
                  className="bg-slate-800 border-slate-700 text-white"
                  placeholder="What this secret is used for"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)} className="border-slate-700 text-slate-300">
                Cancel
              </Button>
              <Button onClick={handleAddSecret} className="bg-gradient-to-r from-red-500 to-amber-600 text-white">
                Add Secret
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Secrets Table */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white">Stored Secrets</CardTitle>
          <CardDescription className="text-slate-400">
            Click to reveal secret values. Values are masked by default.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px]">
            <Table>
              <TableHeader>
                <TableRow className="border-slate-800 hover:bg-slate-800/50">
                  <TableHead className="text-slate-300">Name</TableHead>
                  <TableHead className="text-slate-300">Category</TableHead>
                  <TableHead className="text-slate-300">Value</TableHead>
                  <TableHead className="text-slate-300">Description</TableHead>
                  <TableHead className="text-slate-300 text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {secrets.map((secret) => (
                  <TableRow key={secret.id} className="border-slate-800 hover:bg-slate-800/30">
                    <TableCell className="font-mono text-white">{secret.name}</TableCell>
                    <TableCell>
                      <Badge className={`${categoryColors[secret.category]?.bg || categoryColors.other.bg} ${categoryColors[secret.category]?.text || categoryColors.other.text}`}>
                        {secret.category.replace("_", " ")}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-slate-400 max-w-xs truncate">
                      {revealedSecrets[secret.id] || secret.value}
                    </TableCell>
                    <TableCell className="text-slate-500 max-w-xs truncate">
                      {secret.description || "-"}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleReveal(secret.id)}
                          className="text-slate-400 hover:text-white"
                        >
                          {revealedSecrets[secret.id] ? (
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                            </svg>
                          ) : (
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleCopy(secret.id)}
                          className="text-slate-400 hover:text-white relative"
                        >
                          {copiedId === secret.id ? (
                            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 text-xs">
                              Copied!
                            </Badge>
                          ) : (
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(secret.id)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {secrets.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-slate-500 py-8">
                      No secrets stored yet. Add your first secret to get started.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Quick Reference */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-white">API Keys</p>
                <p className="text-xs text-slate-500">{secrets.filter(s => s.category === 'api_key').length} stored</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-white">Tokens</p>
                <p className="text-xs text-slate-500">{secrets.filter(s => s.category === 'token').length} stored</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-white">Environment</p>
                <p className="text-xs text-slate-500">{secrets.filter(s => s.category === 'env').length} stored</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  );
}
