"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Building2,
  ChevronDown,
  Plus,
  Settings,
  Users,
  Check,
  Crown
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

interface Organization {
  id: number;
  name: string;
  slug: string;
  plan: string;
  role: string;
  member_count: number;
}

interface OrgSelectorProps {
  onSettingsClick?: () => void;
}

export function OrgSelector({ onSettingsClick }: OrgSelectorProps) {
  const { user } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchOrganizations();
  }, []);

  const fetchOrganizations = async () => {
    try {
      const token = localStorage.getItem('auth_access_token');
      if (!token) return;

      const response = await fetch('/api/organizations', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setOrganizations(data.organizations || []);
      }
    } catch (error) {
      console.error('Failed to fetch organizations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const currentOrg = user?.organization;

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg animate-pulse">
        <div className="w-4 h-4 bg-white/20 rounded" />
        <div className="w-24 h-4 bg-white/20 rounded" />
      </div>
    );
  }

  if (!currentOrg) {
    return (
      <button
        onClick={() => {/* TODO: Open create org modal */}}
        className="flex items-center gap-2 px-3 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors"
      >
        <Plus className="w-4 h-4" />
        <span className="text-sm font-medium">Create Organization</span>
      </button>
    );
  }

  const planColors: Record<string, string> = {
    free: 'bg-slate-500/20 text-slate-400',
    starter: 'bg-blue-500/20 text-blue-400',
    pro: 'bg-purple-500/20 text-purple-400',
    enterprise: 'bg-amber-500/20 text-amber-400',
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
      >
        <Building2 className="w-4 h-4 text-slate-400" />
        <span className="text-sm text-white font-medium max-w-[120px] truncate">
          {currentOrg.name}
        </span>
        <span className={`text-xs px-1.5 py-0.5 rounded capitalize ${planColors[currentOrg.plan] || planColors.free}`}>
          {currentOrg.plan}
        </span>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />

            {/* Dropdown */}
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute left-0 top-full mt-2 w-72 bg-slate-900 border border-white/10 rounded-xl overflow-hidden shadow-xl z-50"
            >
              {/* Current Org Header */}
              <div className="p-4 border-b border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-white font-semibold">{currentOrg.name}</p>
                  {user?.role === 'owner' && (
                    <Crown className="w-4 h-4 text-amber-400" />
                  )}
                </div>
                <div className="flex items-center gap-3 text-sm text-slate-400">
                  <span className="flex items-center gap-1">
                    <Users className="w-3 h-3" />
                    {organizations[0]?.member_count || 1} members
                  </span>
                  <span className={`px-1.5 py-0.5 rounded text-xs capitalize ${planColors[currentOrg.plan] || planColors.free}`}>
                    {currentOrg.plan}
                  </span>
                </div>
              </div>

              {/* Organization List */}
              {organizations.length > 1 && (
                <div className="p-2 border-b border-white/10">
                  <p className="px-2 py-1 text-xs text-slate-500 uppercase tracking-wider">
                    Switch Organization
                  </p>
                  {organizations.map((org) => (
                    <button
                      key={org.id}
                      onClick={() => {
                        // TODO: Implement org switching
                        setIsOpen(false);
                      }}
                      className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
                        org.id === currentOrg.id
                          ? 'bg-blue-500/10 text-blue-400'
                          : 'text-slate-300 hover:bg-white/5'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Building2 className="w-4 h-4" />
                        <span className="text-sm truncate max-w-[150px]">{org.name}</span>
                      </div>
                      {org.id === currentOrg.id && (
                        <Check className="w-4 h-4" />
                      )}
                    </button>
                  ))}
                </div>
              )}

              {/* Actions */}
              <div className="p-2">
                <button
                  onClick={() => {
                    setIsOpen(false);
                    onSettingsClick?.();
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-slate-300 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <Settings className="w-4 h-4" />
                  <span className="text-sm">Organization Settings</span>
                </button>
                <button
                  onClick={() => {
                    // TODO: Open create org modal
                    setIsOpen(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-slate-300 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span className="text-sm">Create New Organization</span>
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
