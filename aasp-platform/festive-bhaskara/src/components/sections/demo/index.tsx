'use client';

import * as React from 'react';
import { motion } from 'motion/react';
import { Play, Zap, Shield, Clock, FileText, RotateCcw, ExternalLink } from 'lucide-react';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { SimulatorSection } from './simulator';
import { PoliciesSection } from './policies';
import { ApprovalsSection } from './approvals';
import { AuditSection } from './audit';

const demoTabs = [
  { id: 'simulator', label: 'Simulator', icon: <Zap className="h-4 w-4" /> },
  { id: 'policies', label: 'Policies', icon: <Shield className="h-4 w-4" /> },
  { id: 'approvals', label: 'Approvals', icon: <Clock className="h-4 w-4" /> },
  { id: 'audit', label: 'Audit Trail', icon: <FileText className="h-4 w-4" /> },
];

export function DemoSection() {
  const [activeTab, setActiveTab] = React.useState('simulator');
  const [isResetting, setIsResetting] = React.useState(false);

  const handleReset = async () => {
    setIsResetting(true);
    try {
      await fetch('/api/demo/reset', { method: 'POST' });
      // Reload to refresh all state
      window.location.reload();
    } catch (error) {
      console.error('Failed to reset demo:', error);
    } finally {
      setIsResetting(false);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'simulator':
        return <SimulatorSection />;
      case 'policies':
        return <PoliciesSection />;
      case 'approvals':
        return <ApprovalsSection />;
      case 'audit':
        return <AuditSection />;
      default:
        return <SimulatorSection />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header with Video Placeholder */}
      <div className="text-center max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 text-sm text-primary mb-4"
        >
          <Play className="h-4 w-4" />
          Interactive Demo
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-4xl font-bold text-foreground mb-4"
        >
          Experience AASP in Action
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg text-muted-foreground mb-8"
        >
          Fire AI agent actions, create policies, experience approval workflows, and explore the audit trail
        </motion.p>

        {/* Video Placeholder */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="relative aspect-video max-w-3xl mx-auto rounded-2xl border border-border/50 bg-card/30 backdrop-blur-sm overflow-hidden mb-8"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5" />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary/20 border border-primary/30 mb-4 cursor-pointer hover:bg-primary/30 transition-colors">
              <Play className="h-8 w-8 text-primary ml-1" />
            </div>
            <p className="text-lg font-medium text-foreground">Product Walkthrough</p>
            <p className="text-sm text-muted-foreground">Coming soon - Demo video placeholder</p>
          </div>
          {/* Video embed placeholder */}
          <div className="absolute bottom-4 right-4 flex items-center gap-2 text-xs text-muted-foreground">
            <ExternalLink className="h-3 w-3" />
            Full demo available
          </div>
        </motion.div>
      </div>

      {/* Demo Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex items-center justify-between border-b border-border/50 pb-4"
      >
        <div>
          <AnimatedTabs
            tabs={demoTabs}
            activeTab={activeTab}
            onTabChange={setActiveTab}
            className="bg-transparent"
          />
        </div>

        <button
          onClick={handleReset}
          disabled={isResetting}
          className="flex items-center gap-2 rounded-lg bg-secondary/50 px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors disabled:opacity-50"
        >
          <RotateCcw className={`h-4 w-4 ${isResetting ? 'animate-spin' : ''}`} />
          Reset Demo
        </button>
      </motion.div>

      {/* Demo Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
      >
        {renderContent()}
      </motion.div>
    </div>
  );
}
