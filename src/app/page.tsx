"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { motion, useScroll } from "motion/react";
import {
  ArrowRight,
  Shield,
  Check,
  Play,
  Users,
  Calendar,
  Zap,
} from "lucide-react";
import { AnimatedAvatarGroup } from "@/components/ui/animated-avatar-group";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

// Dithered Pattern Divider Component
function DitheredDivider({ fromDark = true }: { fromDark?: boolean }) {
  const darkColor = "#0f172a"; // slate-900
  const lightColor = "#ffffff";
  const topColor = fromDark ? darkColor : lightColor;
  const bottomColor = fromDark ? lightColor : darkColor;

  // Each row has a different density of pixels
  // Using CSS background patterns for crisp, square pixels
  const pixelSize = 4; // 4px squares

  return (
    <div className="relative w-full flex flex-col">
      {/* Row 1: 100% top */}
      <div className="h-2" style={{ backgroundColor: topColor }} />

      {/* Row 2: ~90% top - very sparse */}
      <div
        className="h-2"
        style={{
          backgroundColor: topColor,
          backgroundImage: `repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent ${pixelSize * 3}px,
            ${bottomColor} ${pixelSize * 3}px,
            ${bottomColor} ${pixelSize * 4}px
          )`,
          backgroundSize: `${pixelSize * 4}px ${pixelSize}px`
        }}
      />

      {/* Row 3: ~75% top */}
      <div
        className="h-2"
        style={{
          backgroundColor: topColor,
          backgroundImage: `repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent ${pixelSize * 2}px,
            ${bottomColor} ${pixelSize * 2}px,
            ${bottomColor} ${pixelSize * 3}px
          )`,
          backgroundSize: `${pixelSize * 3}px ${pixelSize}px`
        }}
      />

      {/* Row 4: 50% checkerboard */}
      <div
        className="h-3"
        style={{
          backgroundColor: topColor,
          backgroundImage: `
            linear-gradient(45deg, ${bottomColor} 25%, transparent 25%),
            linear-gradient(-45deg, ${bottomColor} 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, ${bottomColor} 75%),
            linear-gradient(-45deg, transparent 75%, ${bottomColor} 75%)
          `,
          backgroundSize: `${pixelSize * 2}px ${pixelSize * 2}px`,
          backgroundPosition: `0 0, 0 ${pixelSize}px, ${pixelSize}px -${pixelSize}px, -${pixelSize}px 0px`
        }}
      />

      {/* Row 5: ~25% top */}
      <div
        className="h-2"
        style={{
          backgroundColor: bottomColor,
          backgroundImage: `repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent ${pixelSize * 2}px,
            ${topColor} ${pixelSize * 2}px,
            ${topColor} ${pixelSize * 3}px
          )`,
          backgroundSize: `${pixelSize * 3}px ${pixelSize}px`
        }}
      />

      {/* Row 6: ~10% top - very sparse */}
      <div
        className="h-2"
        style={{
          backgroundColor: bottomColor,
          backgroundImage: `repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent ${pixelSize * 3}px,
            ${topColor} ${pixelSize * 3}px,
            ${topColor} ${pixelSize * 4}px
          )`,
          backgroundSize: `${pixelSize * 4}px ${pixelSize}px`
        }}
      />

      {/* Row 7: 100% bottom */}
      <div className="h-2" style={{ backgroundColor: bottomColor }} />
    </div>
  );
}

// Consultant avatars for hero
const CONSULTANT_AVATARS = [
  { src: "/avatars/manEN.png", fallback: "MR", name: "Marcus Reynolds" },
  { src: "/avatars/womenEN.png", fallback: "SC", name: "Sarah Chen" },
  { src: "/avatars/Man2EN.png", fallback: "DK", name: "David Kim" },
  { src: "/avatars/male1.png", fallback: "JD", name: "Jean Dupont" },
  { src: "/avatars/Women.png", fallback: "CB", name: "Claire Bernard" },
  { src: "/avatars/women2.png", fallback: "ML", name: "Marie Laurent" },
];

// Detailed consultant data for selection section
const CONSULTANTS_DETAILED = [
  {
    id: "marcus",
    src: "/avatars/manEN.png",
    fallback: "MR",
    name: "Marcus Reynolds",
    language: "English",
    level: "Expert",
    levelBadge: "Enterprise Architect",
    hasMCP: true,
    intro: "Your expert for complex Salesforce architecture and enterprise-level integrations. Marcus handles the most challenging implementations with precision.",
    features: [
      "Design multi-cloud Salesforce architectures",
      "Complex API integrations & data migrations",
      "Performance optimization & scalability planning",
      "Execute changes directly via MCP integration",
      "Security & compliance configurations",
      "Custom Apex & Lightning development guidance"
    ],
    scenario: "You need to integrate Salesforce with 5 external systems",
    scenarioAnswer: "Marcus analyzes your data flows, designs the integration architecture, and implements the connections directly in your org via MCP."
  },
  {
    id: "sarah",
    src: "/avatars/womenEN.png",
    fallback: "SC",
    name: "Sarah Chen",
    language: "English",
    level: "Intermediate",
    levelBadge: "Solutions Consultant",
    hasMCP: true,
    intro: "Your go-to consultant for workflows, automations, and reporting. Sarah streamlines your processes and makes Salesforce work smarter for you.",
    features: [
      "Build automated workflows & approval processes",
      "Create custom reports & dashboards",
      "Configure Sales & Service Cloud features",
      "Implement changes via MCP integration",
      "Process Builder & Flow optimization",
      "User training & best practices"
    ],
    scenario: "You want to automate your lead assignment process",
    scenarioAnswer: "Sarah designs the assignment rules, builds the automation flow, and deploys it to your org—all in one conversation."
  },
  {
    id: "david",
    src: "/avatars/Man2EN.png",
    fallback: "DK",
    name: "David Kim",
    language: "English",
    level: "Beginner",
    levelBadge: "Learning Guide",
    hasMCP: false,
    intro: "Your patient guide to mastering Salesforce fundamentals. David explains concepts clearly and helps you build confidence step by step.",
    features: [
      "Salesforce basics & navigation",
      "Understanding objects, fields & relationships",
      "Creating basic reports & list views",
      "Learning Sales Cloud fundamentals",
      "Best practices for data entry",
      "Guided tutorials & explanations"
    ],
    scenario: "You're new to Salesforce and feel overwhelmed",
    scenarioAnswer: "David walks you through the interface, explains core concepts at your pace, and provides practice exercises to build your skills."
  },
  {
    id: "jean",
    src: "/avatars/male1.png",
    fallback: "JD",
    name: "Jean Dupont",
    language: "Français",
    level: "Expert",
    levelBadge: "Architecte Entreprise",
    hasMCP: true,
    intro: "Votre expert pour les architectures Salesforce complexes et les intégrations d'entreprise. Jean gère les implémentations les plus exigeantes.",
    features: [
      "Conception d'architectures multi-cloud",
      "Intégrations API & migrations de données",
      "Optimisation des performances",
      "Exécution directe via intégration MCP",
      "Configurations sécurité & conformité",
      "Développement Apex & Lightning"
    ],
    scenario: "Vous devez intégrer Salesforce avec 5 systèmes externes",
    scenarioAnswer: "Jean analyse vos flux de données, conçoit l'architecture d'intégration et implémente les connexions directement dans votre org via MCP."
  },
  {
    id: "claire",
    src: "/avatars/Women.png",
    fallback: "CB",
    name: "Claire Bernard",
    language: "Français",
    level: "Intermediate",
    levelBadge: "Consultante Solutions",
    hasMCP: true,
    intro: "Votre consultante pour les workflows, automatisations et rapports. Claire optimise vos processus et rend Salesforce plus efficace.",
    features: [
      "Création de workflows & processus d'approbation",
      "Rapports personnalisés & tableaux de bord",
      "Configuration Sales & Service Cloud",
      "Implémentation via intégration MCP",
      "Optimisation Process Builder & Flow",
      "Formation utilisateurs & bonnes pratiques"
    ],
    scenario: "Vous voulez automatiser l'attribution de vos leads",
    scenarioAnswer: "Claire conçoit les règles d'attribution, construit le flux d'automatisation et le déploie dans votre org—le tout en une conversation."
  },
  {
    id: "marie",
    src: "/avatars/women2.png",
    fallback: "ML",
    name: "Marie Laurent",
    language: "Français",
    level: "Beginner",
    levelBadge: "Guide d'Apprentissage",
    hasMCP: false,
    intro: "Votre guide patiente pour maîtriser les fondamentaux de Salesforce. Marie explique les concepts clairement et vous aide à progresser à votre rythme.",
    features: [
      "Bases de Salesforce & navigation",
      "Objets, champs & relations",
      "Création de rapports basiques",
      "Fondamentaux Sales Cloud",
      "Bonnes pratiques de saisie",
      "Tutoriels guidés & explications"
    ],
    scenario: "Vous débutez sur Salesforce et vous sentez perdu",
    scenarioAnswer: "Marie vous guide dans l'interface, explique les concepts à votre rythme et propose des exercices pratiques."
  }
];


// Hero Section
function HeroSection() {
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background - Dark blue gradient like /technical */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />

      {/* Grid Pattern - Salesforce blue */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#00A1E015_1px,transparent_1px),linear-gradient(to_bottom,#00A1E015_1px,transparent_1px)] bg-[size:3rem_3rem]" />

      {/* Subtle blue orb */}
      <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-[#00A1E0]/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-[#00A1E0]/5 rounded-full blur-[120px] translate-x-1/2 translate-y-1/2" />

      <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
        {/* Animated Avatar Group - 20% smaller */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex flex-col items-center gap-3 mb-8 scale-[0.8]"
        >
          <AnimatedAvatarGroup
            avatars={CONSULTANT_AVATARS}
            maxVisible={6}
            size="lg"
          />
          <span className="text-sm text-slate-400">Meet your AI consultants</span>
        </motion.div>

        {/* Main Headline - 15% smaller, all white */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-4xl md:text-6xl font-bold mb-10 text-white"
        >
          Your Personal
          <br />
          Salesforce Expert
        </motion.h1>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Link href="/login">
            <button className="group flex items-center gap-2 px-8 py-4 bg-[#00A1E0] rounded-xl font-semibold text-white hover:bg-[#0087be] hover:shadow-lg hover:shadow-[#00A1E0]/25 transition-all duration-300">
              Get Started Free
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </Link>
          <button className="flex items-center gap-2 px-8 py-4 bg-white/5 border border-white/20 rounded-xl font-semibold text-white hover:bg-white/10 transition-all duration-300">
            <Play className="w-5 h-5" />
            Watch Demo
          </button>
        </motion.div>
      </div>
    </div>
  );
}

// Features Section with Sticky Video
// Feature avatar configurations for each step
const FEATURE_AVATARS = [
  { // Step 1: 3 avatars on left
    position: "left",
    avatars: [
      { src: "/avatars/male1.png", fallback: "JM" },
      { src: "/avatars/womenEN.png", fallback: "SC" },
      { src: "/avatars/manEN.png", fallback: "MR" },
    ]
  },
  { // Step 2: 1 avatar on right
    position: "right",
    avatars: [
      { src: "/avatars/Women.png", fallback: "EP" },
    ]
  },
  { // Step 3: 1 avatar on left
    position: "left",
    avatars: [
      { src: "/avatars/Man2EN.png", fallback: "DK" },
    ]
  },
  { // Step 4: 1 avatar on right
    position: "right",
    avatars: [
      { src: "/avatars/women2.png", fallback: "LM" },
    ]
  },
];

function FeaturesSection() {
  const features = [
    {
      title: "Choose Your Expert",
      description: "6 specialized AI consultants tailored to your expertise level. From beginners learning the basics to enterprise architects tackling complex integrations.",
    },
    {
      title: "Natural Conversations",
      description: "Voice-powered interactions with real-time responses. Ask questions naturally and get expert guidance as if talking to a human consultant.",
    },
    {
      title: "Multilingual Support",
      description: "Available in English and French with native-speaking AI voices. Get help in the language you're most comfortable with.",
    },
    {
      title: "Powered by RAG",
      description: "Advanced Retrieval-Augmented Generation ensures accurate, up-to-date answers sourced from official Salesforce documentation.",
    }
  ];

  const containerRef = useRef<HTMLDivElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  useEffect(() => {
    return scrollYProgress.on("change", (latest) => {
      const newIndex = Math.min(
        Math.floor(latest * features.length),
        features.length - 1
      );
      setActiveIndex(newIndex);
    });
  }, [scrollYProgress, features.length]);

  return (
    <div ref={containerRef} className="relative bg-white" style={{ height: `${features.length * 100}vh` }}>
      <div className="sticky top-0 min-h-screen flex items-center">
        <div className="max-w-7xl mx-auto px-6 w-full">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left Side - Text with Animated Avatars */}
            <div className="relative h-[200px] flex items-center">
              {features.map((feature, index) => {
                const avatarConfig = FEATURE_AVATARS[index];
                const avatarsOnLeft = avatarConfig.position === "left";

                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{
                      opacity: activeIndex === index ? 1 : 0,
                      y: activeIndex === index ? 0 : 20,
                    }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                    className={`absolute inset-0 flex items-center ${
                      activeIndex === index ? 'pointer-events-auto' : 'pointer-events-none'
                    }`}
                  >
                    <div className={`flex items-center gap-6 w-full ${avatarsOnLeft ? 'flex-row' : 'flex-row-reverse'}`}>
                      {/* Avatar Circles */}
                      <motion.div
                        className="flex -space-x-3"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{
                          opacity: activeIndex === index ? 1 : 0,
                          scale: activeIndex === index ? 1 : 0.8,
                        }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                      >
                        {avatarConfig.avatars.map((avatar, avatarIndex) => (
                          <motion.div
                            key={avatarIndex}
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{
                              opacity: activeIndex === index ? 1 : 0,
                              scale: activeIndex === index ? 1 : 0,
                            }}
                            transition={{
                              duration: 0.4,
                              delay: activeIndex === index ? 0.15 + (avatarIndex * 0.1) : 0
                            }}
                            className="relative"
                            style={{ zIndex: avatarConfig.avatars.length - avatarIndex }}
                          >
                            <div className="w-16 h-16 rounded-full border-3 border-white ring-2 ring-[#00A1E0]/30 overflow-hidden bg-slate-100">
                              <img
                                src={avatar.src}
                                alt={avatar.fallback}
                                className="w-full h-full object-cover"
                              />
                            </div>
                          </motion.div>
                        ))}
                      </motion.div>

                      {/* Text Content */}
                      <div className={`flex-1 ${avatarsOnLeft ? 'text-left' : 'text-right'}`}>
                        <h3 className="text-2xl md:text-3xl font-bold text-slate-900 mb-3">
                          {feature.title}
                        </h3>
                        <p className="text-base md:text-lg text-slate-600 leading-relaxed">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* Right Side - Persistent Video Placeholder (15% bigger) */}
            <div className="relative lg:pl-8">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8 }}
                className="relative scale-[1.15] origin-center"
              >
                {/* Video Container */}
                <div className="aspect-video rounded-2xl bg-slate-900 p-1">
                  <div className="w-full h-full rounded-xl bg-slate-900 flex flex-col items-center justify-center relative overflow-hidden">
                    {/* Grid Pattern */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#00A1E010_1px,transparent_1px),linear-gradient(to_bottom,#00A1E010_1px,transparent_1px)] bg-[size:2rem_2rem]" />

                    {/* Play Button */}
                    <motion.div
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.95 }}
                      className="relative z-10 w-20 h-20 rounded-full bg-[#00A1E0] flex items-center justify-center cursor-pointer shadow-lg shadow-[#00A1E0]/25"
                    >
                      <Play className="w-8 h-8 text-white ml-1" />
                    </motion.div>

                    {/* Video Label */}
                    <p className="mt-6 text-slate-400 text-sm font-medium relative z-10">
                      Watch Product Demo
                    </p>
                    <p className="text-slate-500 text-xs mt-1 relative z-10">
                      2 min overview
                    </p>

                    {/* Subtle blue orb */}
                    <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-[#00A1E0]/20 rounded-full blur-3xl" />
                  </div>
                </div>

                {/* Feature Indicator Dots */}
                <div className="flex justify-center gap-2 mt-6">
                  {features.map((_, index) => (
                    <motion.div
                      key={index}
                      className={`h-2 rounded-full transition-all duration-300 ${
                        activeIndex === index
                          ? 'w-8 bg-[#00A1E0]'
                          : 'w-2 bg-slate-300'
                      }`}
                    />
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Consultants Section
function ConsultantsSection() {
  const [selectedId, setSelectedId] = useState("marcus");
  const selectedConsultant = CONSULTANTS_DETAILED.find(c => c.id === selectedId) || CONSULTANTS_DETAILED[0];

  return (
    <section className="py-24 relative">
      {/* Dark blue background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />
      {/* Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#00A1E010_1px,transparent_1px),linear-gradient(to_bottom,#00A1E010_1px,transparent_1px)] bg-[size:3rem_3rem]" />

      <div className="relative max-w-6xl mx-auto px-6">
        {/* 24/7 Subtitle */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-4"
        >
          <span className="inline-block px-4 py-1.5 bg-[#00A1E0]/10 border border-[#00A1E0]/30 rounded-full text-[#00A1E0] text-sm font-medium">
            Available 24/7
          </span>
        </motion.div>

        {/* Main Hook */}
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-3xl md:text-5xl font-bold text-white text-center mb-6"
        >
          Meet Your AI Consultants
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-slate-400 text-center text-lg mb-8 max-w-2xl mx-auto"
        >
          Choose from 6 specialized experts tailored to your skill level and language
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12"
        >
          <Link href="/login">
            <button className="group flex items-center gap-2 px-8 py-4 bg-[#00A1E0] rounded-xl font-semibold text-white hover:bg-[#0087be] hover:shadow-lg hover:shadow-[#00A1E0]/25 transition-all duration-300">
              Get Started Free
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </Link>
          <button className="flex items-center gap-2 px-8 py-4 bg-white/5 border border-white/20 rounded-xl font-semibold text-white hover:bg-white/10 transition-all duration-300">
            <Play className="w-5 h-5" />
            Watch Demo
          </button>
        </motion.div>

        {/* Avatar Selection Row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="flex justify-center gap-6 md:gap-8 mb-12"
        >
          {CONSULTANTS_DETAILED.map((consultant) => (
            <motion.button
              key={consultant.id}
              onClick={() => setSelectedId(consultant.id)}
              whileHover={{ scale: 1.1, y: -4 }}
              whileTap={{ scale: 0.95 }}
              className={`relative transition-all duration-300 ${
                selectedId === consultant.id
                  ? 'opacity-100'
                  : 'opacity-40 hover:opacity-70'
              }`}
            >
              <div className={`w-16 h-16 md:w-20 md:h-20 rounded-full overflow-hidden border-3 transition-all duration-300 ${
                selectedId === consultant.id
                  ? 'border-[#00A1E0] ring-4 ring-[#00A1E0]/30'
                  : 'border-white/20'
              }`}>
                <img
                  src={consultant.src}
                  alt={consultant.name}
                  className="w-full h-full object-cover"
                />
              </div>
              {/* Language indicator */}
              <div className={`absolute -bottom-1 -right-1 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                consultant.language === 'English'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-slate-900'
              }`}>
                {consultant.language === 'English' ? 'EN' : 'FR'}
              </div>
            </motion.button>
          ))}
        </motion.div>

        {/* Selected Consultant Card */}
        <motion.div
          key={selectedId}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="rounded-2xl overflow-hidden"
          style={{ backgroundColor: '#d5dee4' }}
        >
          <div className="grid md:grid-cols-[300px_1fr] lg:grid-cols-[350px_1fr]">
            {/* Left - Avatar Image */}
            <div className="relative h-64 md:h-auto">
              <img
                src={selectedConsultant.src}
                alt={selectedConsultant.name}
                className="w-full h-full object-cover object-top"
              />
              {/* Gradient overlay for text readability on mobile */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent md:hidden" />
              <div className="absolute bottom-4 left-4 md:hidden">
                <h3 className="text-xl font-bold text-white">{selectedConsultant.name}</h3>
                <p className="text-white/80 text-sm">{selectedConsultant.levelBadge}</p>
              </div>
            </div>

            {/* Right - Content */}
            <div className="p-6 md:p-8">
              {/* Header */}
              <div className="hidden md:block mb-4">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-2xl font-bold text-slate-900">{selectedConsultant.name}</h3>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    selectedConsultant.level === 'Expert'
                      ? 'bg-amber-100 text-amber-800'
                      : selectedConsultant.level === 'Intermediate'
                      ? 'bg-blue-100 text-blue-800'
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {selectedConsultant.levelBadge}
                  </span>
                  {selectedConsultant.hasMCP && (
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                      MCP Enabled
                    </span>
                  )}
                </div>
                <p className="text-slate-600 flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    selectedConsultant.language === 'English' ? 'bg-blue-500' : 'bg-slate-400'
                  }`} />
                  {selectedConsultant.language}
                </p>
              </div>

              {/* Intro */}
              <p className="text-slate-700 mb-6 leading-relaxed">
                {selectedConsultant.intro}
              </p>

              {/* Features */}
              <div className="grid sm:grid-cols-2 gap-2 mb-6">
                {selectedConsultant.features.map((feature, index) => (
                  <div key={index} className="flex items-start gap-2">
                    <Check className="w-5 h-5 text-[#00A1E0] flex-shrink-0 mt-0.5" />
                    <span className="text-slate-700 text-sm">{feature}</span>
                  </div>
                ))}
              </div>

              {/* Scenario */}
              <div className="bg-white/60 rounded-xl p-4">
                <p className="text-slate-500 text-xs uppercase tracking-wide mb-2 font-medium">
                  {selectedConsultant.language === 'English' ? 'Example Scenario' : 'Exemple de scénario'}
                </p>
                <p className="text-slate-800 font-medium mb-2">{selectedConsultant.scenario}</p>
                <p className="text-slate-600 text-sm">{selectedConsultant.scenarioAnswer}</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Magic Card Component with mouse-following spotlight (White bg, dark blue accents)
function MagicCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return (
    <motion.div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ y: -8, scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={`relative overflow-hidden rounded-2xl bg-white border border-slate-200 ${className}`}
      style={{
        boxShadow: isHovered
          ? "0 25px 50px -12px rgba(15, 23, 42, 0.15), 0 0 0 1px rgba(15, 23, 42, 0.1)"
          : "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 0 0 1px rgba(0, 0, 0, 0.02)",
      }}
    >
      {/* Spotlight gradient that follows mouse - dark blue tint */}
      <motion.div
        className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(600px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(15, 23, 42, 0.03), transparent 40%)`,
        }}
      />
      {/* Border glow effect - dark blue */}
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-2xl opacity-0 transition-opacity duration-300"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(400px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(15, 23, 42, 0.08), transparent 40%)`,
          mask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
          maskComposite: "xor",
          WebkitMaskComposite: "xor",
          padding: "1px",
        }}
      />
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}

// Animated Number Component
function AnimatedNumber({ value }: { value: string }) {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.5 }}
      whileInView={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 200, damping: 15 }}
      className="inline-block"
    >
      {value}
    </motion.span>
  );
}

// How It Works Section
function HowItWorksSection() {
  const steps = [
    {
      number: "01",
      title: "Consult",
      subtitle: "Expert Guidance On-Demand",
      description: "Access specialized AI consultants for strategic advice, best practices, and solutions tailored to your Salesforce challenges. Get answers instantly, 24/7.",
      icon: Users,
    },
    {
      number: "02",
      title: "Execute",
      subtitle: "MCP-Powered Actions",
      description: "Go beyond conversation. Our MCP integration lets your consultant implement configurations, automations, and changes directly in your Salesforce org—in real time.",
      icon: Zap,
    },
    {
      number: "03",
      title: "Control",
      subtitle: "Salesforce View Dashboard",
      description: "Monitor, query, and manage your Salesforce data from a unified interface. View records, run reports, and perform operations without leaving the platform.",
      icon: Shield,
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden bg-white">
      {/* Subtle grid pattern - dark blue */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a08_1px,transparent_1px),linear-gradient(to_bottom,#0f172a08_1px,transparent_1px)] bg-[size:4rem_4rem]" />

      <div className="relative max-w-6xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <motion.span
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-block px-4 py-1.5 bg-slate-900/5 border border-slate-900/10 rounded-full text-slate-700 text-sm font-medium mb-4"
          >
            How It Works
          </motion.span>
          <h2 className="text-3xl md:text-5xl font-bold text-slate-900 mb-4">
            Three Pillars of{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-slate-700 to-slate-900">
              Salesforce Mastery
            </span>
          </h2>
          <p className="text-slate-600 text-lg max-w-2xl mx-auto">
            From expert consultation to direct execution—everything you need in one platform
          </p>
        </motion.div>

        {/* Connecting Line - Horizontal on desktop */}
        <div className="hidden md:block absolute top-[380px] left-1/2 -translate-x-1/2 w-[55%] h-[2px]">
          <motion.div
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="h-full bg-gradient-to-r from-transparent via-slate-300 to-transparent origin-left"
          />
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-3 gap-8 md:gap-10 pt-6">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.15 }}
                className="relative"
              >
                {/* Step Number Badge - z-20 so it's above card (z-10) but below navbar (z-50) */}
                <motion.div
                  initial={{ scale: 0 }}
                  whileInView={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 300, damping: 15, delay: index * 0.15 + 0.2 }}
                  className="absolute -top-5 left-1/2 -translate-x-1/2 z-20"
                >
                  <div className="w-11 h-11 rounded-full bg-slate-900 flex items-center justify-center text-white font-bold text-sm shadow-xl shadow-slate-900/30 border-4 border-white">
                    <AnimatedNumber value={step.number} />
                  </div>
                </motion.div>

                <MagicCard className="h-full min-h-[340px] relative z-10">
                  <div className="p-8 pt-10 text-center h-full flex flex-col">
                    {/* Icon */}
                    <motion.div
                      whileHover={{ rotate: [0, -10, 10, 0], scale: 1.1 }}
                      transition={{ duration: 0.5 }}
                      className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-900 mb-6 mx-auto shadow-lg shadow-slate-900/20"
                    >
                      <Icon className="w-8 h-8 text-white" />
                    </motion.div>

                    {/* Title */}
                    <h3 className="text-2xl font-bold text-slate-900 mb-2">
                      {step.title}
                    </h3>
                    <p className="text-slate-500 font-medium mb-4 text-sm">
                      {step.subtitle}
                    </p>

                    {/* Description */}
                    <p className="text-slate-600 leading-relaxed text-sm flex-grow">
                      {step.description}
                    </p>
                  </div>
                </MagicCard>
              </motion.div>
            );
          })}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center mt-16"
        >
          <Link href="/login">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="group relative inline-flex items-center gap-2 px-8 py-4 bg-slate-900 rounded-xl font-semibold text-white overflow-hidden shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/30 transition-shadow duration-300"
            >
              {/* Shimmer effect */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full"
                animate={{ translateX: ["100%", "-100%"] }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
              />
              <span className="relative z-10">Get Started Now</span>
              <ArrowRight className="relative z-10 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </section>
  );
}

// Pricing Section
function PricingSection() {
  const plans = [
    {
      name: "Starter",
      price: "$29",
      period: "/month",
      description: "Perfect for individuals learning Salesforce",
      features: [
        "Access to Beginner AI Consultants",
        "50 conversations/month",
        "Basic RAG knowledge base",
        "Email support",
        "1 language"
      ],
      cta: "Start Free Trial",
      popular: false
    },
    {
      name: "Pro",
      price: "$99",
      period: "/month",
      description: "For power users and small teams",
      features: [
        "All AI Consultant levels",
        "Unlimited conversations",
        "Full RAG knowledge base",
        "Priority support",
        "2 languages (EN/FR)",
        "Conversation history",
        "Export transcripts"
      ],
      cta: "Start Free Trial",
      popular: true
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "",
      description: "For organizations with advanced needs",
      features: [
        "Everything in Pro",
        "Custom AI training",
        "Salesforce MCP integration",
        "SSO & advanced security",
        "Dedicated support",
        "Custom languages",
        "API access"
      ],
      cta: "Contact Sales",
      popular: false
    }
  ];

  return (
    <section className="py-32 relative">
      {/* Dark blue background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />
      {/* Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#00A1E010_1px,transparent_1px),linear-gradient(to_bottom,#00A1E010_1px,transparent_1px)] bg-[size:3rem_3rem]" />

      <div className="relative max-w-6xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Simple, Transparent Pricing
          </h2>
          <p className="text-xl text-slate-400">
            Choose the plan that fits your needs
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className={`relative rounded-2xl p-8 ${
                plan.popular
                  ? 'bg-[#00A1E0]/10 border-2 border-[#00A1E0]/50'
                  : 'bg-white/5 border border-white/10'
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-[#00A1E0] rounded-full text-sm font-medium text-white">
                  Most Popular
                </div>
              )}

              <h3 className="text-2xl font-bold text-white mb-2">{plan.name}</h3>
              <p className="text-slate-400 mb-6">{plan.description}</p>

              <div className="mb-6">
                <span className="text-4xl font-bold text-white">{plan.price}</span>
                <span className="text-slate-400">{plan.period}</span>
              </div>

              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-center gap-3 text-slate-300">
                    <Check className="w-5 h-5 text-[#00A1E0] flex-shrink-0" />
                    {feature}
                  </li>
                ))}
              </ul>

              <Link href="/login">
                <button className={`w-full py-3 rounded-xl font-semibold transition-all duration-300 ${
                  plan.popular
                    ? 'bg-[#00A1E0] text-white hover:bg-[#0087be] hover:shadow-lg hover:shadow-[#00A1E0]/25'
                    : 'bg-white/10 text-white hover:bg-white/20'
                }`}>
                  {plan.cta}
                </button>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// FAQ Section
const FAQ_ITEMS = [
  {
    question: "How is this different from reading Salesforce documentation?",
    answer: "Our AI consultants understand context and provide personalized guidance based on your expertise level. Instead of searching through thousands of pages, you get instant, conversational answers tailored to your specific use case—like having a senior consultant on-demand 24/7."
  },
  {
    question: "What expertise levels are the AI consultants available in?",
    answer: "We offer 6 specialized AI consultants ranging from Beginner (for those new to Salesforce) to Enterprise Architect (for complex integration and architecture questions). Each consultant adapts their communication style and technical depth to match your needs."
  },
  {
    question: "Can I interact using voice instead of typing?",
    answer: "Yes! Our platform features voice-powered interactions with real-time responses. Simply speak your questions naturally and receive expert guidance as if you're talking to a human consultant. Voice support is available in both English and French."
  },
  {
    question: "How accurate are the AI responses?",
    answer: "Our platform uses advanced RAG (Retrieval-Augmented Generation) technology that sources answers directly from official Salesforce documentation. This ensures responses are accurate, up-to-date, and grounded in verified information rather than hallucinated content."
  },
  {
    question: "Is my data and conversation history secure?",
    answer: "Absolutely. We're SOC 2 compliant with enterprise-grade security. Your conversations are encrypted, and we never use your data to train our models. Enterprise customers get additional security features including SSO and dedicated infrastructure."
  },
  {
    question: "Can I cancel or change my plan anytime?",
    answer: "Yes, all plans are month-to-month with no long-term commitments. You can upgrade, downgrade, or cancel your subscription at any time from your account settings. If you cancel, you'll retain access until the end of your billing period."
  }
];

function FAQSection() {
  return (
    <section className="py-24 relative bg-white">
      <div className="relative max-w-3xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">
            Frequently Asked Questions
          </h2>
          <p className="text-slate-600">
            Everything you need to know about SF Consultant AI
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Accordion type="single" collapsible className="w-full">
            {FAQ_ITEMS.map((item, index) => (
              <AccordionItem
                key={index}
                value={`item-${index}`}
                className="border-b border-slate-200"
              >
                <AccordionTrigger className="text-left text-slate-900 hover:text-[#00A1E0] hover:no-underline py-5 text-base md:text-lg font-medium">
                  {item.question}
                </AccordionTrigger>
                <AccordionContent className="text-slate-600 text-base leading-relaxed pb-5">
                  {item.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="text-center mt-12"
        >
          <p className="text-slate-500 text-sm">
            Still have questions?{" "}
            <Link href="/contact" className="text-[#00A1E0] hover:text-[#0087be] transition-colors">
              Contact our team
            </Link>
          </p>
        </motion.div>
      </div>
    </section>
  );
}

// Footer
function Footer() {
  return (
    <footer className="py-16 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border-t border-white/10">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          <div>
            <h3 className="text-xl font-bold text-white mb-4">SF Consultant AI</h3>
            <p className="text-slate-400">
              AI-powered Salesforce consulting at your fingertips.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-4">Product</h4>
            <ul className="space-y-2 text-slate-400">
              <li><Link href="/dashboard/marketplace" className="hover:text-[#00A1E0] transition-colors">Consultants</Link></li>
              <li><Link href="#pricing" className="hover:text-[#00A1E0] transition-colors">Pricing</Link></li>
              <li><Link href="/dashboard/docs" className="hover:text-[#00A1E0] transition-colors">Documentation</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-4">Company</h4>
            <ul className="space-y-2 text-slate-400">
              <li><Link href="/about" className="hover:text-[#00A1E0] transition-colors">About</Link></li>
              <li><Link href="/contact" className="hover:text-[#00A1E0] transition-colors">Contact</Link></li>
              <li><Link href="/careers" className="hover:text-[#00A1E0] transition-colors">Careers</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-4">Legal</h4>
            <ul className="space-y-2 text-slate-400">
              <li><Link href="/privacy" className="hover:text-[#00A1E0] transition-colors">Privacy</Link></li>
              <li><Link href="/terms" className="hover:text-[#00A1E0] transition-colors">Terms</Link></li>
              <li><Link href="/security" className="hover:text-[#00A1E0] transition-colors">Security</Link></li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-slate-500 text-sm">
            &copy; 2026 SF Consultant AI. All rights reserved.
          </p>
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <Shield className="w-4 h-4" />
            <span>SOC 2 Compliant</span>
          </div>
        </div>
      </div>
    </footer>
  );
}

// Navbar
function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-slate-950/90 backdrop-blur-xl border-b border-white/10' : ''
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="text-xl font-bold text-white">
          SF Consultant<span className="text-[#00A1E0]">AI</span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          <Link href="#features" className="text-slate-400 hover:text-white transition-colors">Features</Link>
          <Link href="/agents" className="text-slate-400 hover:text-white transition-colors">Agents</Link>
          <Link href="#pricing" className="text-slate-400 hover:text-white transition-colors">Pricing</Link>
          <Link href="/dashboard/docs" className="text-slate-400 hover:text-white transition-colors">Docs</Link>
        </div>

        <div className="flex items-center gap-4">
          <Link href="/login" className="text-slate-400 hover:text-white transition-colors">
            Log in
          </Link>
          <Link href="/login">
            <button className="px-4 py-2 bg-[#00A1E0] rounded-lg font-medium text-white hover:bg-[#0087be] hover:shadow-lg hover:shadow-[#00A1E0]/25 transition-all duration-300">
              Get Started
            </button>
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}

// Main Page Component
export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <Navbar />
      <HeroSection />
      <DitheredDivider fromDark={true} />
      <div id="features">
        <FeaturesSection />
      </div>
      <DitheredDivider fromDark={false} />
      <div id="consultants">
        <ConsultantsSection />
      </div>
      <DitheredDivider fromDark={true} />
      <div id="how-it-works">
        <HowItWorksSection />
      </div>
      <DitheredDivider fromDark={false} />
      <div id="pricing">
        <PricingSection />
      </div>
      <DitheredDivider fromDark={true} />
      <div id="faq">
        <FAQSection />
      </div>
      <DitheredDivider fromDark={false} />
      <Footer />
    </div>
  );
}
