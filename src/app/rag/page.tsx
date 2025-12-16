"use client";

import { Suspense } from "react";
import {
  RAGTitleSlide,
  RAGArchitectureSlide,
  HybridRetrievalSlide,
  EmbeddingVectorSlide,
  RerankingSlide,
  AgenticValidationSlide,
  EvaluationFrameworkSlide,
  RAGRoadmapSlide
} from "@/components/rag-slides";
import Link from "next/link";
import { useSlideNavigation } from "@/hooks/useSlideNavigation";
import { SlideNavigation, PresenterNotes } from "@/components/presentation";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const slides = [
  RAGTitleSlide,
  RAGArchitectureSlide,
  HybridRetrievalSlide,
  EmbeddingVectorSlide,
  RerankingSlide,
  AgenticValidationSlide,
  EvaluationFrameworkSlide,
  RAGRoadmapSlide
];

// Slide metadata for navigation and presenter notes
const slideInfo = [
  {
    title: "Advanced RAG Introduction",
    subtitle: "Retrieval-Augmented Generation",
    notes: [
      "RAG combines retrieval systems with language models for grounded responses",
      "Key advantage: responses are based on actual documents, reducing hallucination",
      "Use case: game NPCs accessing lore databases, character backstories",
    ],
  },
  {
    title: "System Architecture",
    subtitle: "End-to-end RAG pipeline",
    notes: [
      "Three main stages: Retrieval -> Reranking -> Generation",
      "Hybrid retrieval combines dense vectors with sparse BM25",
      "PostgreSQL + pgvector for production-grade vector storage",
      "MiniLM cross-encoder for semantic reranking",
    ],
  },
  {
    title: "Hybrid Retrieval",
    subtitle: "Dense + Sparse search",
    notes: [
      "Dense retrieval: BGE-M3 embeddings (4096 dimensions)",
      "Sparse retrieval: BM25 for keyword matching and rare terms",
      "RRF (Reciprocal Rank Fusion) merges results from both",
      "k=60 constant in RRF formula balances ranking positions",
    ],
  },
  {
    title: "Embedding & Vector Store",
    subtitle: "BGE-M3 and pgvector",
    notes: [
      "BGE-M3: multilingual, multi-granularity embeddings",
      "4096 dimensions capture rich semantic information",
      "HNSW indexing for sub-linear search complexity",
      "ef_construction=128, m=16 for balanced speed/recall",
    ],
  },
  {
    title: "Reranking Pipeline",
    subtitle: "Cross-encoder refinement",
    notes: [
      "First stage retrieves top-k candidates (k=100)",
      "Cross-encoder scores each query-document pair",
      "MiniLM model: fast inference with strong accuracy",
      "Final top-10 documents passed to LLM",
    ],
  },
  {
    title: "Agentic Validation",
    subtitle: "Query transformation & loops",
    notes: [
      "Query understanding: decomposition, expansion, reformulation",
      "Self-RAG: model critiques its own retrieved context",
      "Validation loop ensures answer quality before returning",
      "Fallback strategies for low-confidence scenarios",
    ],
  },
  {
    title: "Evaluation Framework",
    subtitle: "RAGAS metrics",
    notes: [
      "Faithfulness: Is the answer grounded in retrieved context?",
      "Answer Relevancy: Does it address the user's question?",
      "Context Precision: Are retrieved docs actually useful?",
      "Context Recall: Did we find all relevant information?",
    ],
  },
  {
    title: "Implementation Roadmap",
    subtitle: "16-week development plan",
    notes: [
      "Phase 1: Vector store setup with pgvector (Weeks 1-4)",
      "Phase 2: Embedding pipeline with BGE-M3 (Weeks 5-8)",
      "Phase 3: Retrieval and reranking (Weeks 9-12)",
      "Phase 4: Agentic features and evaluation (Weeks 13-16)",
    ],
  },
];

function RAGPresentationContent() {
  const totalSlides = slides.length;

  const {
    currentSlide,
    goToSlide,
    goToNextSlide,
    goToPrevSlide,
    isFirstSlide,
    isLastSlide,
  } = useSlideNavigation({ totalSlides });

  const CurrentSlideComponent = slides[currentSlide];
  const currentSlideInfo = slideInfo[currentSlide];

  return (
    <div className="dark">
      {/* Back to Overview Link */}
      <Link
        href="/"
        className="fixed top-4 left-4 z-50 px-3 py-1.5 bg-slate-800/80 border border-slate-700 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-700 transition-all backdrop-blur-sm flex items-center gap-2"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        Overview
      </Link>

      {/* Slide Navigation */}
      <SlideNavigation
        currentSlide={currentSlide}
        totalSlides={totalSlides}
        slides={slideInfo}
        onSlideChange={goToSlide}
        onPrev={goToPrevSlide}
        onNext={goToNextSlide}
        isFirstSlide={isFirstSlide}
        isLastSlide={isLastSlide}
        presentationTitle="Advanced RAG Technical"
        accentGradient="from-emerald-500 to-cyan-500"
      />

      {/* Presenter Notes */}
      <PresenterNotes
        notes={currentSlideInfo.notes}
        slideNumber={currentSlide + 1}
        totalSlides={totalSlides}
        slideTitle={currentSlideInfo.title}
      />

      {/* Current Slide */}
      <CurrentSlideComponent slideNumber={currentSlide + 1} totalSlides={totalSlides} />

      {/* Navigation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}

export default function RAGPresentation() {
  return (
    <Suspense fallback={
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading presentation...</div>
      </div>
    }>
      <RAGPresentationContent />
    </Suspense>
  );
}
