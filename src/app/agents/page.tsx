"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "motion/react";
import Image from "next/image";
import {
  Sparkles,
  Zap,
  BookOpen,
  Code,
  Building2,
  CheckCircle2,
  Rocket,
  Calendar,
  Globe2,
  ArrowRight,
  LayoutGrid,
  Languages,
  GraduationCap,
  Lock,
  Clock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MagicCard } from "@/components/ui/magic-card";

// Consultant data from landing page - exact match
const CONSULTANTS = [
  {
    id: "marcus",
    src: "/avatars/manEN.png",
    fallback: "MR",
    name: "Marcus Reynolds",
    language: "en",
    level: "expert",
    levelBadge: "Enterprise Architect",
    hasMCP: true,
    intro: "Your expert for complex Salesforce architecture and enterprise-level integrations. Marcus handles the most challenging implementations with precision.",
    features: [
      "Multi-cloud Salesforce architectures",
      "Complex API integrations",
      "Performance optimization",
      "MCP direct execution",
    ],
  },
  {
    id: "sarah",
    src: "/avatars/womenEN.png",
    fallback: "SC",
    name: "Sarah Chen",
    language: "en",
    level: "intermediate",
    levelBadge: "Solutions Consultant",
    hasMCP: true,
    intro: "Your go-to consultant for workflows, automations, and reporting. Sarah streamlines your processes and makes Salesforce work smarter.",
    features: [
      "Automated workflows",
      "Custom reports & dashboards",
      "Process optimization",
      "MCP integration",
    ],
  },
  {
    id: "david",
    src: "/avatars/Man2EN.png",
    fallback: "DK",
    name: "David Kim",
    language: "en",
    level: "beginner",
    levelBadge: "Learning Guide",
    hasMCP: false,
    intro: "Your patient guide to mastering Salesforce fundamentals. David explains concepts clearly and helps you build confidence step by step.",
    features: [
      "Salesforce basics",
      "Navigation guidance",
      "Core concepts",
      "Step-by-step tutorials",
    ],
  },
  {
    id: "jean",
    src: "/avatars/male1.png",
    fallback: "JD",
    name: "Jean Dupont",
    language: "fr",
    level: "expert",
    levelBadge: "Architecte Entreprise",
    hasMCP: true,
    intro: "Votre expert pour les architectures Salesforce complexes et les integrations d'entreprise. Jean gere les implementations les plus exigeantes.",
    features: [
      "Architectures multi-cloud",
      "Integrations API complexes",
      "Optimisation performances",
      "Execution MCP directe",
    ],
  },
  {
    id: "claire",
    src: "/avatars/Women.png",
    fallback: "CB",
    name: "Claire Bernard",
    language: "fr",
    level: "intermediate",
    levelBadge: "Consultante Solutions",
    hasMCP: true,
    intro: "Votre consultante pour les workflows, automatisations et rapports. Claire optimise vos processus et rend Salesforce plus efficace.",
    features: [
      "Workflows automatises",
      "Rapports personnalises",
      "Optimisation processus",
      "Integration MCP",
    ],
  },
  {
    id: "marie",
    src: "/avatars/women2.png",
    fallback: "ML",
    name: "Marie Laurent",
    language: "fr",
    level: "beginner",
    levelBadge: "Guide d'Apprentissage",
    hasMCP: false,
    intro: "Votre guide patiente pour maitriser les fondamentaux de Salesforce. Marie explique les concepts clairement et vous aide a progresser.",
    features: [
      "Bases de Salesforce",
      "Navigation guidee",
      "Concepts fondamentaux",
      "Tutoriels pas a pas",
    ],
  },
];

// Coming Soon Consultants
const COMING_SOON_CONSULTANTS = [
  {
    id: "coming-1",
    name: "Analytics Expert",
    levelBadge: "Data Specialist",
    language: "en",
    teaser: "Advanced Einstein Analytics & Tableau CRM specialist",
  },
  {
    id: "coming-2",
    name: "Integration Pro",
    levelBadge: "API Architect",
    language: "en",
    teaser: "MuleSoft & external system integration expert",
  },
  {
    id: "coming-3",
    name: "Security Advisor",
    levelBadge: "Compliance Expert",
    language: "en",
    teaser: "Shield, encryption & compliance specialist",
  },
];

type FilterMode = "all" | "language" | "level";
type Language = "en" | "fr";
type Level = "beginner" | "intermediate" | "expert";

const levelConfig = {
  beginner: {
    icon: BookOpen,
    label: "Beginner",
    labelFr: "Debutant",
    color: "#10B981",
  },
  intermediate: {
    icon: Code,
    label: "Intermediate",
    labelFr: "Intermediaire",
    color: "#F59E0B",
  },
  expert: {
    icon: Building2,
    label: "Expert",
    labelFr: "Expert",
    color: "#8B5CF6",
  },
};

const filterTabs = [
  { id: "all" as FilterMode, label: "All Consultants", icon: LayoutGrid },
  { id: "language" as FilterMode, label: "By Language", icon: Languages },
  { id: "level" as FilterMode, label: "By Level", icon: GraduationCap },
];

export default function AgentsPage() {
  const router = useRouter();
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [selectedLanguage, setSelectedLanguage] = useState<Language | null>(null);
  const [selectedLevel, setSelectedLevel] = useState<Level | null>(null);
  const [selectedConsultant, setSelectedConsultant] = useState<typeof CONSULTANTS[0] | null>(null);
  const [hoveredConsultant, setHoveredConsultant] = useState<string | null>(null);

  // Filter consultants based on mode and selection
  const filteredConsultants = useMemo(() => {
    if (filterMode === "all") return CONSULTANTS;
    if (filterMode === "language" && selectedLanguage) {
      return CONSULTANTS.filter((c) => c.language === selectedLanguage);
    }
    if (filterMode === "level" && selectedLevel) {
      return CONSULTANTS.filter((c) => c.level === selectedLevel);
    }
    return CONSULTANTS;
  }, [filterMode, selectedLanguage, selectedLevel]);

  const handleStartSession = () => {
    if (selectedConsultant) {
      const avatarMap: Record<string, string> = {
        marcus: "morgan-en",
        sarah: "jordan-en",
        david: "alex-en",
        jean: "dominique-fr",
        claire: "robin-fr",
        marie: "camille-fr",
      };
      router.push(`/dashboard/product?avatar=${avatarMap[selectedConsultant.id] || selectedConsultant.id}`);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Hero Section Background - Exact match */}
      <div className="fixed inset-0">
        {/* Background - Dark blue gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />

        {/* Grid Pattern - Salesforce blue */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#00A1E015_1px,transparent_1px),linear-gradient(to_bottom,#00A1E015_1px,transparent_1px)] bg-[size:3rem_3rem]" />

        {/* Subtle blue orbs */}
        <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-[#00A1E0]/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-[#00A1E0]/5 rounded-full blur-[120px] translate-x-1/2 translate-y-1/2" />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 border-b border-white/5 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div
              className="flex items-center gap-3 cursor-pointer"
              onClick={() => router.push("/")}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00A1E0] to-[#0176D3] flex items-center justify-center shadow-lg shadow-[#00A1E0]/20">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">
                SF Consultant<span className="text-[#00A1E0]">AI</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                className="text-[#00A1E0] bg-[#00A1E0]/10 hover:bg-[#00A1E0]/20"
              >
                Agents
              </Button>
              <Button
                variant="ghost"
                className="text-slate-400 hover:text-white hover:bg-white/5"
                onClick={() => router.push("/dashboard")}
              >
                Dashboard
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10 max-w-6xl mx-auto px-6 py-16">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <Badge className="mb-4 bg-[#00A1E0]/10 text-[#00A1E0] border-[#00A1E0]/20 px-4 py-1.5">
            <Globe2 className="w-3.5 h-3.5 mr-1.5" />
            6 AI Consultants • 3 Coming Soon
          </Badge>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4">
            Choose Your{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#00A1E0] to-[#00D4FF]">
              AI Consultant
            </span>
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Select your preferred filter to find the perfect Salesforce consultant for your needs
          </p>
        </motion.div>

        {/* Animated Filter Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-10"
        >
          <div className="flex justify-center">
            <div className="inline-flex p-1.5 rounded-2xl bg-slate-900/80 border border-slate-800/50 backdrop-blur-sm">
              {filterTabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = filterMode === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setFilterMode(tab.id);
                      setSelectedLanguage(null);
                      setSelectedLevel(null);
                      setSelectedConsultant(null);
                    }}
                    className="relative px-6 py-3 rounded-xl text-sm font-medium transition-colors duration-200"
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeFilter"
                        className="absolute inset-0 bg-gradient-to-r from-[#00A1E0] to-[#0176D3] rounded-xl shadow-lg shadow-[#00A1E0]/25"
                        transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                    <span className={`relative z-10 flex items-center gap-2 ${isActive ? "text-white" : "text-slate-400 hover:text-slate-200"}`}>
                      <Icon className="w-4 h-4" />
                      {tab.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* Sub-filters for Language/Level modes */}
        <AnimatePresence mode="wait">
          {filterMode === "language" && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 overflow-hidden"
            >
              <div className="flex justify-center gap-4">
                {[
                  { code: "en" as Language, flag: "🇬🇧", label: "English" },
                  { code: "fr" as Language, flag: "🇫🇷", label: "Francais" },
                ].map((lang) => (
                  <motion.button
                    key={lang.code}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      setSelectedLanguage(lang.code);
                      setSelectedConsultant(null);
                    }}
                    className={`relative px-6 py-3 rounded-xl border-2 transition-all duration-300 ${
                      selectedLanguage === lang.code
                        ? "border-[#00A1E0] bg-[#00A1E0]/10 shadow-lg shadow-[#00A1E0]/10"
                        : "border-slate-800 bg-slate-900/50 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{lang.flag}</span>
                      <span className={`font-medium ${selectedLanguage === lang.code ? "text-white" : "text-slate-400"}`}>
                        {lang.label}
                      </span>
                    </div>
                    {selectedLanguage === lang.code && (
                      <motion.div
                        layoutId="lang-check"
                        className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-[#00A1E0] flex items-center justify-center"
                      >
                        <CheckCircle2 className="w-3 h-3 text-white" />
                      </motion.div>
                    )}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}

          {filterMode === "level" && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 overflow-hidden"
            >
              <div className="flex justify-center gap-4">
                {(["beginner", "intermediate", "expert"] as Level[]).map((level) => {
                  const config = levelConfig[level];
                  const Icon = config.icon;
                  return (
                    <motion.button
                      key={level}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => {
                        setSelectedLevel(level);
                        setSelectedConsultant(null);
                      }}
                      className={`relative px-6 py-3 rounded-xl border-2 transition-all duration-300 ${
                        selectedLevel === level
                          ? "border-[#00A1E0] bg-[#00A1E0]/10 shadow-lg shadow-[#00A1E0]/10"
                          : "border-slate-800 bg-slate-900/50 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-8 h-8 rounded-lg flex items-center justify-center"
                          style={{ backgroundColor: `${config.color}20` }}
                        >
                          <Icon className="w-4 h-4" style={{ color: config.color }} />
                        </div>
                        <span className={`font-medium ${selectedLevel === level ? "text-white" : "text-slate-400"}`}>
                          {config.label}
                        </span>
                      </div>
                      {selectedLevel === level && (
                        <motion.div
                          layoutId="level-check"
                          className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-[#00A1E0] flex items-center justify-center"
                        >
                          <CheckCircle2 className="w-3 h-3 text-white" />
                        </motion.div>
                      )}
                    </motion.button>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Consultant Grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-12"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence mode="popLayout">
              {filteredConsultants.map((consultant, index) => {
                const levelConf = levelConfig[consultant.level as Level];
                const LevelIcon = levelConf.icon;
                const isSelected = selectedConsultant?.id === consultant.id;
                const isHovered = hoveredConsultant === consultant.id;

                return (
                  <motion.div
                    key={consultant.id}
                    layout
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <MagicCard
                      gradientColor="#00A1E0"
                      gradientOpacity={0.3}
                      className="rounded-2xl"
                    >
                      <motion.button
                        whileHover={{ y: -4 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedConsultant(consultant)}
                        onMouseEnter={() => setHoveredConsultant(consultant.id)}
                        onMouseLeave={() => setHoveredConsultant(null)}
                        className={`relative w-full p-6 rounded-2xl border-2 text-left transition-all duration-300 overflow-hidden ${
                          isSelected
                            ? "border-[#00A1E0] bg-slate-900/80"
                            : "border-slate-800/50 bg-slate-900/50 hover:border-slate-700"
                        }`}
                      >
                        {/* Level Badge & Language */}
                        <div className="flex items-center justify-between mb-4">
                          <div
                            className="px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5"
                            style={{
                              backgroundColor: `${levelConf.color}20`,
                              color: levelConf.color
                            }}
                          >
                            <LevelIcon className="w-3 h-3" />
                            {levelConf.label}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-lg">{consultant.language === "en" ? "🇬🇧" : "🇫🇷"}</span>
                            {consultant.hasMCP && (
                              <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30 text-[10px]">
                                <Zap className="w-2.5 h-2.5 mr-1" />
                                MCP
                              </Badge>
                            )}
                          </div>
                        </div>

                        {/* Avatar */}
                        <div className="relative mb-4">
                          <div className={`relative w-24 h-24 mx-auto rounded-full overflow-hidden border-4 transition-all duration-300 ${
                            isSelected ? "border-[#00A1E0] shadow-lg shadow-[#00A1E0]/30" : "border-slate-700"
                          }`}>
                            <Image
                              src={consultant.src}
                              alt={consultant.name}
                              fill
                              className="object-cover"
                            />
                          </div>
                          {isSelected && (
                            <motion.div
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#00A1E0] flex items-center justify-center shadow-lg"
                            >
                              <CheckCircle2 className="w-5 h-5 text-white" />
                            </motion.div>
                          )}
                        </div>

                        {/* Info */}
                        <div className="text-center">
                          <h3 className="text-xl font-bold text-white mb-1">
                            {consultant.name}
                          </h3>
                          <p className="text-sm text-[#00A1E0] font-medium mb-3">
                            {consultant.levelBadge}
                          </p>
                          <p className="text-xs text-slate-400 line-clamp-2">
                            {consultant.intro}
                          </p>
                        </div>

                        {/* Features on hover */}
                        <AnimatePresence>
                          {(isHovered || isSelected) && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className="mt-4 pt-4 border-t border-slate-800"
                            >
                              <div className="grid grid-cols-2 gap-2">
                                {consultant.features.slice(0, 4).map((feature, idx) => (
                                  <div
                                    key={idx}
                                    className="flex items-center gap-1.5 text-[11px] text-slate-400"
                                  >
                                    <CheckCircle2 className="w-3 h-3 text-[#00A1E0] flex-shrink-0" />
                                    <span className="truncate">{feature}</span>
                                  </div>
                                ))}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.button>
                    </MagicCard>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {/* Coming Soon Cards - Only show in "all" mode */}
            {filterMode === "all" && COMING_SOON_CONSULTANTS.map((consultant, index) => (
              <motion.div
                key={consultant.id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: (filteredConsultants.length + index) * 0.05 }}
              >
                <div className="relative">
                  {/* Blur overlay */}
                  <div className="absolute inset-0 backdrop-blur-[2px] bg-slate-950/30 rounded-2xl z-10 pointer-events-none" />

                  <MagicCard
                    gradientColor="#64748b"
                    gradientOpacity={0.15}
                    className="rounded-2xl"
                  >
                    <div className="relative w-full p-6 rounded-2xl border-2 border-slate-800/30 bg-slate-900/30 text-left overflow-hidden opacity-60">
                      {/* Coming Soon Badge */}
                      <div className="flex items-center justify-between mb-4">
                        <Badge className="bg-slate-700/50 text-slate-400 border-slate-600/50 text-xs">
                          <Clock className="w-3 h-3 mr-1" />
                          Coming Soon
                        </Badge>
                        <span className="text-lg opacity-50">🌐</span>
                      </div>

                      {/* Placeholder Avatar */}
                      <div className="relative mb-4">
                        <div className="relative w-24 h-24 mx-auto rounded-full overflow-hidden border-4 border-slate-700/50 bg-gradient-to-br from-slate-800 to-slate-900">
                          <div className="absolute inset-0 flex items-center justify-center">
                            <Lock className="w-8 h-8 text-slate-600" />
                          </div>
                        </div>
                      </div>

                      {/* Info */}
                      <div className="text-center">
                        <h3 className="text-xl font-bold text-slate-500 mb-1">
                          {consultant.name}
                        </h3>
                        <p className="text-sm text-slate-600 font-medium mb-3">
                          {consultant.levelBadge}
                        </p>
                        <p className="text-xs text-slate-600 line-clamp-2">
                          {consultant.teaser}
                        </p>
                      </div>

                      {/* Coming Soon Footer */}
                      <div className="mt-4 pt-4 border-t border-slate-800/50 text-center">
                        <span className="text-xs text-slate-600 flex items-center justify-center gap-1.5">
                          <Sparkles className="w-3 h-3" />
                          Launching Q2 2025
                        </span>
                      </div>
                    </div>
                  </MagicCard>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Selected Consultant Detail & Action */}
        <AnimatePresence mode="wait">
          {selectedConsultant && (
            <motion.div
              key={selectedConsultant.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-12"
            >
              <div className="relative rounded-3xl border border-[#00A1E0]/30 bg-gradient-to-br from-slate-900 via-slate-900/95 to-slate-950 p-8 overflow-hidden">
                {/* Glow effect */}
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-[#00A1E0] to-transparent" />

                <div className="flex flex-col lg:flex-row items-center gap-8">
                  {/* Avatar */}
                  <div className="relative">
                    <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-[#00A1E0] shadow-2xl shadow-[#00A1E0]/30">
                      <Image
                        src={selectedConsultant.src}
                        alt={selectedConsultant.name}
                        width={128}
                        height={128}
                        className="object-cover"
                      />
                    </div>
                    {selectedConsultant.hasMCP && (
                      <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-full bg-purple-500 flex items-center justify-center border-4 border-slate-900 shadow-lg">
                        <Zap className="w-5 h-5 text-white" />
                      </div>
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 text-center lg:text-left">
                    <div className="flex flex-wrap items-center justify-center lg:justify-start gap-2 mb-2">
                      <Badge
                        className="text-xs"
                        style={{
                          backgroundColor: `${levelConfig[selectedConsultant.level as Level].color}20`,
                          color: levelConfig[selectedConsultant.level as Level].color,
                          borderColor: `${levelConfig[selectedConsultant.level as Level].color}30`
                        }}
                      >
                        {selectedConsultant.levelBadge}
                      </Badge>
                      <Badge className="bg-slate-800 text-slate-300 border-slate-700">
                        {selectedConsultant.language === "en" ? "🇬🇧 English" : "🇫🇷 Francais"}
                      </Badge>
                    </div>
                    <h2 className="text-3xl font-bold text-white mb-2">
                      {selectedConsultant.name}
                    </h2>
                    <p className="text-slate-400 max-w-xl">
                      {selectedConsultant.intro}
                    </p>
                  </div>

                  {/* Action */}
                  <div className="flex flex-col gap-3">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleStartSession}
                      className="group relative px-8 py-4 rounded-xl bg-gradient-to-r from-[#00A1E0] to-[#0176D3] text-white font-semibold shadow-lg shadow-[#00A1E0]/30 hover:shadow-xl hover:shadow-[#00A1E0]/40 transition-all"
                    >
                      <span className="flex items-center gap-2">
                        <Rocket className="w-5 h-5" />
                        Start Consultation
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                      </span>
                    </motion.button>
                    <button
                      className="px-8 py-3 rounded-xl border border-slate-700 text-slate-400 hover:text-white hover:border-slate-600 transition-all text-sm flex items-center justify-center gap-2"
                    >
                      <Calendar className="w-4 h-4" />
                      Schedule for Later
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Prompt to select */}
        {!selectedConsultant && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8"
          >
            <p className="text-slate-500">
              Select a consultant above to continue
            </p>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-800/50 mt-16 py-8">
        <div className="max-w-6xl mx-auto px-6 text-center text-slate-600 text-sm">
          &copy; 2025 SF Consultant AI. All rights reserved.
        </div>
      </footer>
    </div>
  );
}
