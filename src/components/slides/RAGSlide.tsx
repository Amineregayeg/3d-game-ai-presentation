"use client";

import { SlideWrapper } from "./SlideWrapper";

interface RAGSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RAGSlide({ slideNumber, totalSlides }: RAGSlideProps) {
  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/10 border border-purple-500/30 rounded-full">
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
            <span className="text-purple-400 text-sm font-medium">Module 2</span>
          </div>
          <h2 className="text-5xl font-bold text-white">
            Advanced <span className="bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">RAG</span> System
          </h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            Retrieval-Augmented Generation for context-aware, accurate responses
          </p>
        </div>

        {/* RAG Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Knowledge Base */}
          <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white">Knowledge Base</h3>
            </div>
            <ul className="space-y-2 text-slate-400 text-sm">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                Blender API documentation
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                3D modeling best practices
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                Game asset specifications
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                Material & texture libraries
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                User conversation history
              </li>
            </ul>
          </div>

          {/* Retrieval Engine */}
          <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-pink-500 to-rose-600 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white">Smart Retrieval</h3>
            </div>
            <ul className="space-y-2 text-slate-400 text-sm">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-pink-400 rounded-full" />
                Semantic vector search
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-pink-400 rounded-full" />
                Hybrid keyword + embedding
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-pink-400 rounded-full" />
                Context window optimization
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-pink-400 rounded-full" />
                Multi-query decomposition
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-pink-400 rounded-full" />
                Re-ranking for relevance
              </li>
            </ul>
          </div>

          {/* Generation */}
          <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white">Augmented Gen</h3>
            </div>
            <ul className="space-y-2 text-slate-400 text-sm">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                Context injection
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                Chain-of-thought reasoning
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                Tool/function calling
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                Structured output
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                Response validation
              </li>
            </ul>
          </div>
        </div>

        {/* Tech stack */}
        <div className="p-6 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-slate-700/50 rounded-2xl">
          <h4 className="text-lg font-semibold text-white mb-4 text-center">Technology Stack</h4>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              "LangChain",
              "LlamaIndex",
              "ChromaDB / Pinecone",
              "OpenAI Embeddings",
              "Claude / GPT-4",
              "FAISS",
              "Sentence Transformers"
            ].map((tech) => (
              <span key={tech} className="px-4 py-2 bg-slate-800/50 border border-purple-500/30 rounded-lg text-sm text-purple-300">
                {tech}
              </span>
            ))}
          </div>
        </div>
      </div>
    </SlideWrapper>
  );
}
