'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Building2,
  Cpu,
  Rocket,
  BarChart3,
  Mail,
  Shield,
  Play,
} from 'lucide-react';

import { Dock, DockItem, DockSeparator } from '@/components/ui/magnified-dock';
import { AnimatedTabs } from '@/components/ui/animated-tabs';

// Business Sections
import { HeroSection } from '@/components/sections/business/hero';
import { ProblemSection } from '@/components/sections/business/problem';
import { MarketSection } from '@/components/sections/business/market';
import { CompetitiveSection } from '@/components/sections/business/competitive';
import { BusinessModelSection } from '@/components/sections/business/business-model';
import { GoToMarketSection } from '@/components/sections/business/go-to-market';

// Technical Sections
import { ArchitectureSection } from '@/components/sections/technical/architecture';
import { CoreLayersSection } from '@/components/sections/technical/core-layers';
import { IntegrationsSection } from '@/components/sections/technical/integrations';

// MVP Sections
import { MVPScopeSection } from '@/components/sections/mvp/scope';
import { FeaturesSection } from '@/components/sections/mvp/features';
import { RoadmapSection } from '@/components/sections/mvp/roadmap';
import { MetricsSection } from '@/components/sections/mvp/metrics';
import { CTASection } from '@/components/sections/mvp/cta';

// Demo Section
import { DemoSection } from '@/components/sections/demo';

type Section = 'business' | 'technical' | 'mvp' | 'demo';

const businessTabs = [
  { id: 'hero', label: 'Overview' },
  { id: 'problem', label: 'Problem' },
  { id: 'market', label: 'Market' },
  { id: 'competitive', label: 'Competitive' },
  { id: 'model', label: 'Business Model' },
  { id: 'gtm', label: 'Go-to-Market' },
];

const technicalTabs = [
  { id: 'architecture', label: 'Architecture' },
  { id: 'layers', label: 'Core Layers' },
  { id: 'integrations', label: 'Integrations' },
];

const mvpTabs = [
  { id: 'scope', label: 'Scope' },
  { id: 'features', label: 'Features' },
  { id: 'roadmap', label: 'Roadmap' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'cta', label: 'Get Started' },
];

export default function Home() {
  const [activeSection, setActiveSection] = React.useState<Section>('business');
  const [businessTab, setBusinessTab] = React.useState('hero');
  const [technicalTab, setTechnicalTab] = React.useState('architecture');
  const [mvpTab, setMvpTab] = React.useState('scope');

  const renderBusinessContent = () => {
    switch (businessTab) {
      case 'hero':
        return <HeroSection />;
      case 'problem':
        return <ProblemSection />;
      case 'market':
        return <MarketSection />;
      case 'competitive':
        return <CompetitiveSection />;
      case 'model':
        return <BusinessModelSection />;
      case 'gtm':
        return <GoToMarketSection />;
      default:
        return <HeroSection />;
    }
  };

  const renderTechnicalContent = () => {
    switch (technicalTab) {
      case 'architecture':
        return <ArchitectureSection />;
      case 'layers':
        return <CoreLayersSection />;
      case 'integrations':
        return <IntegrationsSection />;
      default:
        return <ArchitectureSection />;
    }
  };

  const renderMVPContent = () => {
    switch (mvpTab) {
      case 'scope':
        return <MVPScopeSection />;
      case 'features':
        return <FeaturesSection />;
      case 'roadmap':
        return <RoadmapSection />;
      case 'metrics':
        return <MetricsSection />;
      case 'cta':
        return <CTASection />;
      default:
        return <MVPScopeSection />;
    }
  };

  return (
    <div className="relative min-h-screen bg-background">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5 pointer-events-none" />

      {/* Grid pattern overlay */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.02]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '50px 50px',
        }}
      />

      {/* Main content */}
      <main className="relative pb-32">
        {/* Header */}
        <header className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-xl">
          <div className="mx-auto max-w-7xl px-6 py-4">
            <div className="flex items-center justify-between">
              {/* Logo */}
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent">
                  <Shield className="h-5 w-5 text-white" />
                </div>
                <div>
                  <span className="text-lg font-bold text-foreground">AASP</span>
                  <span className="ml-2 text-sm text-muted-foreground">AI Agent Security Platform</span>
                </div>
              </div>

              {/* Section Tabs */}
              <AnimatedTabs
                tabs={[
                  { id: 'business', label: 'Business', icon: <Building2 className="h-4 w-4" /> },
                  { id: 'technical', label: 'Technical', icon: <Cpu className="h-4 w-4" /> },
                  { id: 'mvp', label: 'MVP', icon: <Rocket className="h-4 w-4" /> },
                  { id: 'demo', label: 'Demo', icon: <Play className="h-4 w-4" /> },
                ]}
                activeTab={activeSection}
                onTabChange={(id) => setActiveSection(id as Section)}
              />

              {/* Status Badge */}
              <div className="flex items-center gap-2 rounded-full bg-green-500/10 px-3 py-1.5 text-sm text-green-400">
                <span className="h-2 w-2 rounded-full bg-green-400 animate-pulse" />
                Building
              </div>
            </div>
          </div>

          {/* Sub-navigation */}
          <div className="border-t border-border/30 bg-secondary/20">
            <div className="mx-auto max-w-7xl px-6 py-2">
              <AnimatePresence mode="wait">
                {activeSection === 'business' && (
                  <motion.div
                    key="business-tabs"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <AnimatedTabs
                      tabs={businessTabs}
                      activeTab={businessTab}
                      onTabChange={setBusinessTab}
                      className="bg-transparent"
                    />
                  </motion.div>
                )}
                {activeSection === 'technical' && (
                  <motion.div
                    key="technical-tabs"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <AnimatedTabs
                      tabs={technicalTabs}
                      activeTab={technicalTab}
                      onTabChange={setTechnicalTab}
                      className="bg-transparent"
                    />
                  </motion.div>
                )}
                {activeSection === 'mvp' && (
                  <motion.div
                    key="mvp-tabs"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <AnimatedTabs
                      tabs={mvpTabs}
                      activeTab={mvpTab}
                      onTabChange={setMvpTab}
                      className="bg-transparent"
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <div className="mx-auto max-w-7xl px-6 py-12">
          <AnimatePresence mode="wait">
            {activeSection === 'business' && (
              <motion.div
                key={`business-${businessTab}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {renderBusinessContent()}
              </motion.div>
            )}
            {activeSection === 'technical' && (
              <motion.div
                key={`technical-${technicalTab}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {renderTechnicalContent()}
              </motion.div>
            )}
            {activeSection === 'mvp' && (
              <motion.div
                key={`mvp-${mvpTab}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {renderMVPContent()}
              </motion.div>
            )}
            {activeSection === 'demo' && (
              <motion.div
                key="demo"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <DemoSection />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Bottom Dock Navigation */}
      <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2">
        <Dock magnification={70} distance={150}>
          <DockItem
            label="Business"
            onClick={() => setActiveSection('business')}
            isActive={activeSection === 'business'}
          >
            <Building2 className="h-6 w-6" />
          </DockItem>
          <DockItem
            label="Technical"
            onClick={() => setActiveSection('technical')}
            isActive={activeSection === 'technical'}
          >
            <Cpu className="h-6 w-6" />
          </DockItem>
          <DockItem
            label="MVP"
            onClick={() => setActiveSection('mvp')}
            isActive={activeSection === 'mvp'}
          >
            <Rocket className="h-6 w-6" />
          </DockItem>
          <DockItem
            label="Demo"
            onClick={() => setActiveSection('demo')}
            isActive={activeSection === 'demo'}
          >
            <Play className="h-6 w-6" />
          </DockItem>

          <DockSeparator />

          <DockItem
            label="Metrics"
            onClick={() => {
              setActiveSection('mvp');
              setMvpTab('metrics');
            }}
            isActive={activeSection === 'mvp' && mvpTab === 'metrics'}
          >
            <BarChart3 className="h-6 w-6" />
          </DockItem>
          <DockItem
            label="Contact"
            onClick={() => {
              setActiveSection('mvp');
              setMvpTab('cta');
            }}
            isActive={activeSection === 'mvp' && mvpTab === 'cta'}
          >
            <Mail className="h-6 w-6" />
          </DockItem>
        </Dock>
      </div>
    </div>
  );
}
