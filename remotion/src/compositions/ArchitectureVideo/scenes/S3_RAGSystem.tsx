import React from 'react';
import { AbsoluteFill, interpolate, Easing } from 'remotion';
import { COLORS, THEMES, GRADIENTS } from '../../../utils/colors';
import { EASING, highlightDim, cameraZoom, getMicroSceneProgress } from '../../../utils/animations';
import { secondsToFrames, MICRO_SCENES } from '../../../utils/timings';
import { GradientText } from '../components';
import { Cursor, createCursorPath, CursorKeyframe } from '../../../components/Cursor';

interface S3_RAGSystemProps {
  frame: number;
}

const METRICS = [
  { label: 'Documents', value: 3885, unit: '' },
  { label: 'RAGAS Score', value: 0.82, unit: '', isDecimal: true },
  { label: 'Hallucination', value: 5, unit: '%', prefix: '<' },
];

// RAG Pipeline stages
const PIPELINE_STAGES = [
  { id: 'query', label: 'User Query', icon: '💬', color: COLORS.cyan500 },
  { id: 'retrieval', label: 'Hybrid Retrieval', icon: '🔍', color: COLORS.emerald500 },
  { id: 'rerank', label: 'Cross-Encoder', icon: '📊', color: COLORS.purple500 },
  { id: 'context', label: 'Context Assembly', icon: '📄', color: COLORS.amber500 },
  { id: 'generate', label: 'LLM Generation', icon: '🤖', color: COLORS.rose500 },
];

// Scene duration: 40s = 1200 frames
const SCENE_MICRO = {
  title: { start: 0, duration: 90 },        // 0-3s
  pipeline: { start: 90, duration: 240 },   // 3-11s
  architecture: { start: 330, duration: 300 }, // 11-21s
  retrieval: { start: 630, duration: 300 },  // 21-31s
  reranking: { start: 930, duration: 210 },  // 31-38s
  outro: { start: 1140, duration: 60 },      // 38-40s
};

// Helper for spring-based animations
const springIn = (
  frame: number,
  start: number,
  duration: number = 15
): { opacity: number; scale: number; y: number } => {
  const progress = interpolate(frame, [start, start + duration], [0, 1], {
    easing: EASING.spring,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return {
    opacity: progress,
    scale: interpolate(progress, [0, 1], [0.8, 1]),
    y: interpolate(progress, [0, 1], [20, 0]),
  };
};

// Animated count-up with easing
const countUp = (frame: number, start: number, duration: number, from: number, to: number): number => {
  const progress = interpolate(frame, [start, start + duration], [0, 1], {
    easing: EASING.snappy,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return Math.round(from + (to - from) * progress);
};

// Draw path animation
const drawPath = (frame: number, start: number, duration: number): number => {
  return interpolate(frame, [start, start + duration], [0, 1], {
    easing: EASING.snappy,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

// Moving token along path
const getTokenPosition = (
  frame: number,
  start: number,
  duration: number,
  path: { x1: number; y1: number; x2: number; y2: number }
): { x: number; y: number; opacity: number } => {
  const progress = interpolate(frame, [start, start + duration], [0, 1], {
    easing: Easing.inOut(Easing.quad),
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const opacity = interpolate(frame, [start, start + 5, start + duration - 5, start + duration], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return {
    x: path.x1 + (path.x2 - path.x1) * progress,
    y: path.y1 + (path.y2 - path.y1) * progress,
    opacity,
  };
};

export const S3_RAGSystem: React.FC<S3_RAGSystemProps> = ({ frame }) => {
  const theme = THEMES.rag;

  // Determine current micro-scene
  const getMicroScene = (): string => {
    if (frame < SCENE_MICRO.pipeline.start) return 'title';
    if (frame < SCENE_MICRO.architecture.start) return 'pipeline';
    if (frame < SCENE_MICRO.retrieval.start) return 'architecture';
    if (frame < SCENE_MICRO.reranking.start) return 'retrieval';
    if (frame < SCENE_MICRO.outro.start) return 'reranking';
    return 'outro';
  };

  const microScene = getMicroScene();

  // Create cursor path for the entire scene
  const cursorKeyframes: CursorKeyframe[] = [
    // Pipeline navigation
    { frame: 100, x: 240, y: 300, state: 'moving' },
    { frame: 130, x: 240, y: 300, state: 'hover' },
    { frame: 145, x: 240, y: 300, state: 'clicking' },
    { frame: 160, x: 380, y: 300, state: 'moving' },
    { frame: 190, x: 380, y: 300, state: 'hover' },
    { frame: 205, x: 380, y: 300, state: 'clicking' },
    { frame: 220, x: 520, y: 300, state: 'moving' },
    { frame: 250, x: 520, y: 300, state: 'hover' },
    { frame: 265, x: 520, y: 300, state: 'clicking' },
    { frame: 280, x: 660, y: 300, state: 'moving' },
    { frame: 300, x: 800, y: 300, state: 'moving' },
    // Architecture navigation
    { frame: 350, x: 150, y: 350, state: 'moving' },
    { frame: 380, x: 150, y: 350, state: 'hover' },
    { frame: 420, x: 400, y: 280, state: 'moving' },
    { frame: 450, x: 400, y: 280, state: 'hover' },
    { frame: 500, x: 600, y: 350, state: 'moving' },
    { frame: 530, x: 600, y: 350, state: 'clicking' },
    { frame: 580, x: 800, y: 350, state: 'moving' },
    // Retrieval section
    { frame: 650, x: 350, y: 350, state: 'moving' },
    { frame: 700, x: 350, y: 350, state: 'hover' },
    { frame: 730, x: 350, y: 350, state: 'clicking' },
    { frame: 780, x: 750, y: 350, state: 'moving' },
    { frame: 830, x: 750, y: 350, state: 'hover' },
    { frame: 860, x: 750, y: 350, state: 'clicking' },
    // Reranking section
    { frame: 950, x: 300, y: 300, state: 'moving' },
    { frame: 1000, x: 550, y: 300, state: 'moving' },
    { frame: 1050, x: 800, y: 300, state: 'moving' },
    { frame: 1080, x: 800, y: 300, state: 'hover' },
  ];

  // Global camera zoom based on micro-scene
  const getCameraTransform = (): { scale: number; x: number; y: number } => {
    switch (microScene) {
      case 'architecture':
        const archProgress = getMicroSceneProgress(frame, SCENE_MICRO.architecture.start, SCENE_MICRO.architecture.duration);
        const zoomTarget = archProgress < 0.3 ? 0 : archProgress < 0.6 ? 1 : 2;
        const zoomScale = interpolate(archProgress, [0, 0.3, 0.6, 1], [1, 1.15, 1.15, 1], {
          easing: EASING.spring,
          extrapolateRight: 'clamp',
        });
        const zoomX = interpolate(zoomTarget, [0, 1, 2], [0, -100, 100]);
        const zoomY = interpolate(zoomTarget, [0, 1, 2], [0, -50, -50]);
        return { scale: zoomScale, x: zoomX, y: zoomY };
      default:
        return { scale: 1, x: 0, y: 0 };
    }
  };

  const camera = getCameraTransform();

  const renderTitle = () => {
    const localFrame = frame;
    const titleAnim = springIn(localFrame, 0);
    const subtitleAnim = springIn(localFrame, 15);
    const metricsAnim = springIn(localFrame, 30);

    // Exit animation
    const exitProgress = interpolate(localFrame, [70, 90], [0, 1], {
      easing: EASING.snappy,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 24,
          opacity: 1 - exitProgress,
          transform: `scale(${1 + exitProgress * 0.1}) translateY(${-exitProgress * 50}px)`,
        }}
      >
        <div
          style={{
            padding: '12px 24px',
            backgroundColor: `${theme.primary}20`,
            border: `2px solid ${theme.primary}`,
            borderRadius: 50,
            opacity: titleAnim.opacity,
            transform: `scale(${titleAnim.scale}) translateY(${titleAnim.y}px)`,
          }}
        >
          <span
            style={{
              fontSize: 18,
              fontWeight: 600,
              color: theme.primary,
              fontFamily: 'system-ui, -apple-system, sans-serif',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}
          >
            Component 2
          </span>
        </div>

        <div
          style={{
            opacity: titleAnim.opacity,
            transform: `scale(${titleAnim.scale}) translateY(${titleAnim.y}px)`,
          }}
        >
          <GradientText gradient={GRADIENTS.rag} fontSize={80} fontWeight={800}>
            Advanced RAG System
          </GradientText>
        </div>

        <div style={{ opacity: subtitleAnim.opacity, transform: `translateY(${subtitleAnim.y}px)` }}>
          <span
            style={{
              fontSize: 28,
              color: COLORS.slate600,
              fontFamily: 'system-ui, -apple-system, sans-serif',
            }}
          >
            Retrieval-Augmented Generation for Game Development
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            gap: 40,
            marginTop: 40,
            opacity: metricsAnim.opacity,
            transform: `translateY(${metricsAnim.y}px)`,
          }}
        >
          {METRICS.map((metric, i) => {
            const metricAnim = springIn(localFrame, 35 + i * 8);
            return (
              <div
                key={metric.label}
                style={{
                  textAlign: 'center',
                  opacity: metricAnim.opacity,
                  transform: `scale(${metricAnim.scale})`,
                }}
              >
                <div
                  style={{
                    fontSize: 48,
                    fontWeight: 700,
                    color: theme.primary,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                  }}
                >
                  {metric.prefix || ''}
                  {metric.isDecimal
                    ? (countUp(localFrame, 45, 30, 0, metric.value * 100) / 100).toFixed(2)
                    : countUp(localFrame, 45, 30, 0, metric.value)}
                  {metric.unit}
                </div>
                <div
                  style={{
                    fontSize: 14,
                    color: COLORS.slate600,
                    marginTop: 4,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                  }}
                >
                  {metric.label}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    );
  };

  const renderPipeline = () => {
    const localFrame = frame - SCENE_MICRO.pipeline.start;
    const titleAnim = springIn(localFrame, 0);

    // Active stage based on cursor/time
    const activeStage = Math.min(4, Math.floor(localFrame / 40));

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 60,
        }}
      >
        {/* Section Title */}
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 60 }}>
          <GradientText gradient={GRADIENTS.rag} fontSize={48} fontWeight={700}>
            RAG Pipeline Architecture
          </GradientText>
        </div>

        {/* Pipeline Flow with animated stages */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 20,
            position: 'relative',
          }}
        >
          {PIPELINE_STAGES.map((stage, i) => {
            const stageAnim = springIn(localFrame, 15 + i * 12);
            const isActive = i <= activeStage;
            const isCurrent = i === activeStage;

            // Pulse animation for current stage
            const pulseScale = isCurrent
              ? interpolate(Math.sin(localFrame * 0.15), [-1, 1], [1, 1.05])
              : 1;

            return (
              <React.Fragment key={stage.id}>
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 12,
                    opacity: stageAnim.opacity,
                    transform: `scale(${stageAnim.scale * pulseScale}) translateY(${stageAnim.y}px)`,
                  }}
                >
                  <div
                    style={{
                      width: 80,
                      height: 80,
                      borderRadius: 16,
                      backgroundColor: isActive ? `${stage.color}30` : `${COLORS.slate800}50`,
                      border: `2px solid ${isActive ? stage.color : COLORS.slate700}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 32,
                      boxShadow: isCurrent ? `0 0 30px ${stage.color}60` : isActive ? `0 0 15px ${stage.color}30` : 'none',
                    }}
                  >
                    {stage.icon}
                  </div>
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: isActive ? stage.color : COLORS.slate500,
                      fontFamily: 'system-ui, -apple-system, sans-serif',
                      textAlign: 'center',
                      width: 100,
                    }}
                  >
                    {stage.label}
                  </span>
                </div>
                {i < PIPELINE_STAGES.length - 1 && (
                  <svg width="40" height="20" style={{ opacity: stageAnim.opacity }}>
                    <defs>
                      <linearGradient id={`arrow-grad-${i}`} x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor={stage.color} />
                        <stop offset="100%" stopColor={PIPELINE_STAGES[i + 1].color} />
                      </linearGradient>
                    </defs>
                    <path
                      d="M0 10 L30 10 M25 5 L30 10 L25 15"
                      stroke={i < activeStage ? `url(#arrow-grad-${i})` : COLORS.slate600}
                      strokeWidth="2"
                      fill="none"
                      strokeDasharray="40"
                      strokeDashoffset={40 - drawPath(localFrame, 30 + i * 20, 15) * 40}
                    />
                  </svg>
                )}
              </React.Fragment>
            );
          })}

          {/* Animated data token flowing through pipeline */}
          {localFrame > 60 && localFrame < 200 && (
            <div
              style={{
                position: 'absolute',
                left: interpolate(localFrame, [60, 200], [60, 820], {
                  easing: Easing.inOut(Easing.quad),
                }),
                top: 30,
                width: 20,
                height: 20,
                borderRadius: '50%',
                background: `linear-gradient(135deg, ${COLORS.cyan500}, ${COLORS.purple500})`,
                boxShadow: `0 0 20px ${COLORS.cyan500}`,
                opacity: interpolate(localFrame, [60, 70, 190, 200], [0, 1, 1, 0]),
              }}
            />
          )}
        </div>

        {/* Description text that updates with stage */}
        <div
          style={{
            marginTop: 60,
            textAlign: 'center',
            height: 80,
          }}
        >
          {PIPELINE_STAGES.map((stage, i) => {
            const isVisible = i === activeStage;
            const textAnim = springIn(localFrame, 35 + i * 40);
            if (!isVisible) return null;
            return (
              <div
                key={stage.id}
                style={{
                  opacity: textAnim.opacity,
                  transform: `translateY(${textAnim.y}px)`,
                }}
              >
                <span
                  style={{
                    fontSize: 24,
                    color: stage.color,
                    fontFamily: 'system-ui',
                    fontWeight: 600,
                  }}
                >
                  {i === 0 && 'Natural language query enters the system'}
                  {i === 1 && 'Dense + Sparse retrieval with RRF fusion'}
                  {i === 2 && 'Neural reranking for precision'}
                  {i === 3 && 'Assembling relevant context chunks'}
                  {i === 4 && 'LLM generates grounded response'}
                </span>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    );
  };

  const renderArchitecture = () => {
    const localFrame = frame - SCENE_MICRO.architecture.start;
    const titleAnim = springIn(localFrame, 0);

    // Token animation paths
    const token1 = getTokenPosition(localFrame, 30, 40, { x1: 170, y1: 200, x2: 300, y2: 140 });
    const token2 = getTokenPosition(localFrame, 50, 40, { x1: 170, y1: 200, x2: 300, y2: 260 });
    const token3 = getTokenPosition(localFrame, 100, 30, { x1: 480, y1: 140, x2: 540, y2: 200 });
    const token4 = getTokenPosition(localFrame, 110, 30, { x1: 480, y1: 260, x2: 540, y2: 200 });
    const token5 = getTokenPosition(localFrame, 150, 30, { x1: 660, y1: 200, x2: 720, y2: 200 });

    // Active component highlighting
    const activeComponent = localFrame < 60 ? 'kb' : localFrame < 120 ? 'indices' : localFrame < 180 ? 'fusion' : 'reranker';

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 60,
        }}
      >
        {/* Section Title */}
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 30 }}>
          <GradientText gradient={GRADIENTS.rag} fontSize={42} fontWeight={700}>
            Hybrid Retrieval Architecture
          </GradientText>
        </div>

        {/* Architecture Diagram */}
        <div
          style={{
            position: 'relative',
            width: 900,
            height: 420,
          }}
        >
          <svg width="100%" height="100%" viewBox="0 0 900 420">
            {/* Knowledge Base */}
            <g
              opacity={highlightDim(activeComponent === 'kb', localFrame, 0)}
              style={{ filter: activeComponent === 'kb' ? `drop-shadow(0 0 20px ${COLORS.emerald500})` : 'none' }}
            >
              <rect
                x="50"
                y="80"
                width="160"
                height="260"
                rx="16"
                fill={`${COLORS.emerald500}15`}
                stroke={COLORS.emerald500}
                strokeWidth="2"
                opacity={springIn(localFrame, 5).opacity}
              />
              <text x="130" y="115" fill={COLORS.emerald500} fontSize="16" fontWeight="700" textAnchor="middle" fontFamily="system-ui">Knowledge Base</text>

              {/* Animated document stack */}
              {[0, 1, 2].map((row) => (
                <g key={row}>
                  {[0, 1].map((col) => {
                    const docAnim = springIn(localFrame, 15 + (row * 2 + col) * 5);
                    return (
                      <rect
                        key={`${row}-${col}`}
                        x={70 + col * 50}
                        y={135 + row * 55}
                        width="40"
                        height="40"
                        rx="6"
                        fill={`${COLORS.emerald500}30`}
                        stroke={COLORS.emerald500}
                        strokeWidth="1"
                        opacity={docAnim.opacity}
                        transform={`scale(${docAnim.scale})`}
                        style={{ transformOrigin: `${90 + col * 50}px ${155 + row * 55}px` }}
                      />
                    );
                  })}
                </g>
              ))}
              <text x="130" y="320" fill={COLORS.slate500} fontSize="12" textAnchor="middle" fontFamily="system-ui">3,885 Documents</text>
            </g>

            {/* Vector Store */}
            <g
              opacity={highlightDim(activeComponent === 'indices', localFrame, 0)}
              style={{ filter: activeComponent === 'indices' ? `drop-shadow(0 0 15px ${COLORS.cyan500})` : 'none' }}
            >
              <rect
                x="270"
                y="90"
                width="170"
                height="100"
                rx="12"
                fill={`${COLORS.cyan500}15`}
                stroke={COLORS.cyan500}
                strokeWidth="2"
                opacity={springIn(localFrame, 40).opacity}
              />
              <text x="355" y="125" fill={COLORS.cyan500} fontSize="15" fontWeight="600" textAnchor="middle" fontFamily="system-ui">pgvector</text>
              <text x="355" y="148" fill={COLORS.slate500} fontSize="11" textAnchor="middle" fontFamily="system-ui">BGE-M3 Embeddings</text>
              <text x="355" y="168" fill={COLORS.slate600} fontSize="10" textAnchor="middle" fontFamily="monospace">1024 dimensions</text>
            </g>

            {/* BM25 Index */}
            <g
              opacity={highlightDim(activeComponent === 'indices', localFrame, 0)}
              style={{ filter: activeComponent === 'indices' ? `drop-shadow(0 0 15px ${COLORS.amber500})` : 'none' }}
            >
              <rect
                x="270"
                y="220"
                width="170"
                height="100"
                rx="12"
                fill={`${COLORS.amber500}15`}
                stroke={COLORS.amber500}
                strokeWidth="2"
                opacity={springIn(localFrame, 50).opacity}
              />
              <text x="355" y="255" fill={COLORS.amber500} fontSize="15" fontWeight="600" textAnchor="middle" fontFamily="system-ui">BM25 Index</text>
              <text x="355" y="278" fill={COLORS.slate500} fontSize="11" textAnchor="middle" fontFamily="system-ui">Sparse Retrieval</text>
              <text x="355" y="298" fill={COLORS.slate600} fontSize="10" textAnchor="middle" fontFamily="monospace">TF-IDF scoring</text>
            </g>

            {/* RRF Fusion */}
            <g
              opacity={highlightDim(activeComponent === 'fusion', localFrame, 0)}
              style={{ filter: activeComponent === 'fusion' ? `drop-shadow(0 0 20px ${COLORS.purple500})` : 'none' }}
            >
              <rect
                x="500"
                y="140"
                width="130"
                height="130"
                rx="65"
                fill={`${COLORS.purple500}15`}
                stroke={COLORS.purple500}
                strokeWidth="2"
                opacity={springIn(localFrame, 70).opacity}
              />
              <text x="565" y="195" fill={COLORS.purple500} fontSize="18" fontWeight="700" textAnchor="middle" fontFamily="system-ui">RRF</text>
              <text x="565" y="218" fill={COLORS.slate500} fontSize="11" textAnchor="middle" fontFamily="system-ui">Fusion</text>
              <text x="565" y="238" fill={COLORS.slate600} fontSize="10" textAnchor="middle" fontFamily="monospace">k=60</text>
            </g>

            {/* Cross-Encoder */}
            <g
              opacity={highlightDim(activeComponent === 'reranker', localFrame, 0)}
              style={{ filter: activeComponent === 'reranker' ? `drop-shadow(0 0 20px ${COLORS.rose500})` : 'none' }}
            >
              <rect
                x="690"
                y="130"
                width="150"
                height="150"
                rx="16"
                fill={`${COLORS.rose500}15`}
                stroke={COLORS.rose500}
                strokeWidth="2"
                opacity={springIn(localFrame, 90).opacity}
              />
              <text x="765" y="175" fill={COLORS.rose500} fontSize="14" fontWeight="600" textAnchor="middle" fontFamily="system-ui">Cross-Encoder</text>
              <text x="765" y="198" fill={COLORS.slate500} fontSize="11" textAnchor="middle" fontFamily="system-ui">ms-marco-MiniLM</text>
              <text x="765" y="218" fill={COLORS.slate600} fontSize="10" textAnchor="middle" fontFamily="monospace">L-6-v2</text>
              <text x="765" y="250" fill={COLORS.emerald500} fontSize="12" textAnchor="middle" fontFamily="system-ui">Top-5 Results</text>
            </g>

            {/* Connection arrows with draw animation */}
            <g>
              {/* KB to Vector */}
              <path
                d="M210 140 L270 140"
                stroke={COLORS.emerald500}
                strokeWidth="2"
                strokeDasharray="60"
                strokeDashoffset={60 - drawPath(localFrame, 30, 20) * 60}
                markerEnd="url(#arrowhead)"
              />
              {/* KB to BM25 */}
              <path
                d="M210 270 L270 270"
                stroke={COLORS.amber500}
                strokeWidth="2"
                strokeDasharray="60"
                strokeDashoffset={60 - drawPath(localFrame, 40, 20) * 60}
                markerEnd="url(#arrowhead)"
              />
              {/* Vector to RRF */}
              <path
                d="M440 140 L500 180"
                stroke={COLORS.cyan500}
                strokeWidth="2"
                strokeDasharray="80"
                strokeDashoffset={80 - drawPath(localFrame, 80, 20) * 80}
                markerEnd="url(#arrowhead)"
              />
              {/* BM25 to RRF */}
              <path
                d="M440 270 L500 230"
                stroke={COLORS.amber500}
                strokeWidth="2"
                strokeDasharray="80"
                strokeDashoffset={80 - drawPath(localFrame, 90, 20) * 80}
                markerEnd="url(#arrowhead)"
              />
              {/* RRF to Reranker */}
              <path
                d="M630 205 L690 205"
                stroke={COLORS.purple500}
                strokeWidth="2"
                strokeDasharray="60"
                strokeDashoffset={60 - drawPath(localFrame, 120, 20) * 60}
                markerEnd="url(#arrowhead)"
              />
            </g>

            {/* Animated tokens */}
            {token1.opacity > 0 && (
              <circle cx={token1.x} cy={token1.y} r="8" fill={COLORS.cyan500} opacity={token1.opacity}>
                <animate attributeName="r" values="6;10;6" dur="0.5s" repeatCount="indefinite" />
              </circle>
            )}
            {token2.opacity > 0 && (
              <circle cx={token2.x} cy={token2.y} r="8" fill={COLORS.amber500} opacity={token2.opacity}>
                <animate attributeName="r" values="6;10;6" dur="0.5s" repeatCount="indefinite" />
              </circle>
            )}
            {token3.opacity > 0 && (
              <circle cx={token3.x} cy={token3.y} r="8" fill={COLORS.cyan500} opacity={token3.opacity} />
            )}
            {token4.opacity > 0 && (
              <circle cx={token4.x} cy={token4.y} r="8" fill={COLORS.amber500} opacity={token4.opacity} />
            )}
            {token5.opacity > 0 && (
              <circle cx={token5.x} cy={token5.y} r="10" fill={COLORS.purple500} opacity={token5.opacity}>
                <animate attributeName="r" values="8;12;8" dur="0.4s" repeatCount="indefinite" />
              </circle>
            )}

            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill={COLORS.slate500} />
              </marker>
            </defs>
          </svg>
        </div>
      </AbsoluteFill>
    );
  };

  const renderRetrieval = () => {
    const localFrame = frame - SCENE_MICRO.retrieval.start;
    const titleAnim = springIn(localFrame, 0);

    // Alternate focus between dense and sparse
    const focusIndex = Math.floor(localFrame / 100) % 2;

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 60,
        }}
      >
        {/* Section Title */}
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 30 }}>
          <GradientText gradient={GRADIENTS.rag} fontSize={42} fontWeight={700}>
            Hybrid Retrieval System
          </GradientText>
        </div>

        {/* Two-column layout */}
        <div
          style={{
            display: 'flex',
            gap: 40,
            width: '100%',
            maxWidth: 1100,
          }}
        >
          {/* Dense Retrieval */}
          <div
            style={{
              flex: 1,
              opacity: highlightDim(focusIndex === 0, localFrame, 0),
              transform: `scale(${focusIndex === 0 ? 1.02 : 1})`,
              transition: 'transform 0.3s',
            }}
          >
            <div
              style={{
                backgroundColor: `${COLORS.cyan500}10`,
                border: `2px solid ${COLORS.cyan500}`,
                borderRadius: 20,
                padding: 25,
                boxShadow: focusIndex === 0 ? `0 0 30px ${COLORS.cyan500}30` : 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <span style={{ fontSize: 28 }}>🧠</span>
                <span style={{ fontSize: 24, fontWeight: 700, color: COLORS.cyan500, fontFamily: 'system-ui' }}>
                  Dense Retrieval
                </span>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>Model</div>
                <div style={{ fontSize: 22, fontWeight: 600, color: COLORS.white, fontFamily: 'monospace' }}>BGE-M3</div>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>Embedding Dimensions</div>
                <div style={{ fontSize: 32, fontWeight: 700, color: COLORS.cyan500, fontFamily: 'monospace' }}>
                  {countUp(localFrame, 20, 25, 0, 1024)}
                </div>
              </div>

              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 13, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>Index Type</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: COLORS.white, fontFamily: 'system-ui' }}>HNSW</div>
                <div style={{ fontSize: 11, color: COLORS.slate600, fontFamily: 'monospace' }}>M=16, ef_construction=64</div>
              </div>

              <div
                style={{
                  backgroundColor: `${COLORS.cyan500}20`,
                  borderRadius: 10,
                  padding: 12,
                }}
              >
                <div style={{ fontSize: 11, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>SIMILARITY</div>
                <div style={{ fontSize: 13, color: COLORS.cyan400, fontFamily: 'monospace' }}>
                  cosine_similarity(query, doc)
                </div>
              </div>
            </div>
          </div>

          {/* Sparse Retrieval */}
          <div
            style={{
              flex: 1,
              opacity: highlightDim(focusIndex === 1, localFrame, 0),
              transform: `scale(${focusIndex === 1 ? 1.02 : 1})`,
              transition: 'transform 0.3s',
            }}
          >
            <div
              style={{
                backgroundColor: `${COLORS.amber500}10`,
                border: `2px solid ${COLORS.amber500}`,
                borderRadius: 20,
                padding: 25,
                boxShadow: focusIndex === 1 ? `0 0 30px ${COLORS.amber500}30` : 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <span style={{ fontSize: 28 }}>📝</span>
                <span style={{ fontSize: 24, fontWeight: 700, color: COLORS.amber500, fontFamily: 'system-ui' }}>
                  Sparse Retrieval
                </span>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>Algorithm</div>
                <div style={{ fontSize: 22, fontWeight: 600, color: COLORS.white, fontFamily: 'monospace' }}>BM25</div>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>Parameters</div>
                <div style={{ display: 'flex', gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 12, color: COLORS.slate600, fontFamily: 'system-ui' }}>k1</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.amber500, fontFamily: 'monospace' }}>1.2</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: COLORS.slate600, fontFamily: 'system-ui' }}>b</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.amber500, fontFamily: 'monospace' }}>0.75</div>
                  </div>
                </div>
              </div>

              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 13, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>Features</div>
                <div style={{ fontSize: 15, color: COLORS.white, fontFamily: 'system-ui' }}>TF-IDF Weighting</div>
                <div style={{ fontSize: 11, color: COLORS.slate600, fontFamily: 'system-ui' }}>Length normalization</div>
              </div>

              <div
                style={{
                  backgroundColor: `${COLORS.amber500}20`,
                  borderRadius: 10,
                  padding: 12,
                }}
              >
                <div style={{ fontSize: 11, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>SCORING</div>
                <div style={{ fontSize: 13, color: COLORS.amber400, fontFamily: 'monospace' }}>
                  BM25(query, doc, k1, b)
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RRF Fusion Formula */}
        <div
          style={{
            marginTop: 30,
            opacity: springIn(localFrame, 60).opacity,
            transform: `translateY(${springIn(localFrame, 60).y}px)`,
          }}
        >
          <div
            style={{
              backgroundColor: `${COLORS.purple500}15`,
              border: `2px solid ${COLORS.purple500}`,
              borderRadius: 14,
              padding: '16px 32px',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: 14, color: COLORS.purple400, marginBottom: 10, fontFamily: 'system-ui', fontWeight: 600 }}>
              Reciprocal Rank Fusion (RRF)
            </div>
            <div style={{ fontSize: 22, color: COLORS.white, fontFamily: 'monospace' }}>
              RRF(d) = Σ 1 / (k + rank(d))
            </div>
            <div style={{ fontSize: 12, color: COLORS.slate500, marginTop: 6, fontFamily: 'system-ui' }}>
              k = 60 (constant)
            </div>
          </div>
        </div>
      </AbsoluteFill>
    );
  };

  const renderReranking = () => {
    const localFrame = frame - SCENE_MICRO.reranking.start;
    const titleAnim = springIn(localFrame, 0);

    const rerankingSteps = [
      { step: 1, label: 'Initial Results', count: 20, color: COLORS.cyan500 },
      { step: 2, label: 'Cross-Encoder', count: 20, color: COLORS.purple500 },
      { step: 3, label: 'Top Results', count: 5, color: COLORS.emerald500 },
    ];

    // Active step
    const activeStep = Math.min(2, Math.floor(localFrame / 50));

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 50,
        }}
      >
        {/* Section Title */}
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 30 }}>
          <GradientText gradient={GRADIENTS.rag} fontSize={42} fontWeight={700}>
            Cross-Encoder Reranking
          </GradientText>
        </div>

        {/* Reranking Pipeline */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 30,
            marginBottom: 35,
          }}
        >
          {rerankingSteps.map((step, i) => {
            const stepAnim = springIn(localFrame, 10 + i * 15);
            const isActive = i <= activeStep;
            const isCurrent = i === activeStep;

            return (
              <React.Fragment key={step.step}>
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 12,
                    opacity: stepAnim.opacity,
                    transform: `scale(${stepAnim.scale * (isCurrent ? 1.05 : 1)}) translateY(${stepAnim.y}px)`,
                  }}
                >
                  <div
                    style={{
                      width: 180,
                      padding: '20px 16px',
                      backgroundColor: `${step.color}15`,
                      border: `2px solid ${step.color}`,
                      borderRadius: 14,
                      textAlign: 'center',
                      boxShadow: isCurrent ? `0 0 25px ${step.color}40` : 'none',
                    }}
                  >
                    <div style={{ fontSize: 12, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>
                      Step {step.step}
                    </div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: step.color, fontFamily: 'system-ui' }}>
                      {step.label}
                    </div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: COLORS.white, marginTop: 10, fontFamily: 'monospace' }}>
                      {isActive ? countUp(localFrame, 20 + i * 20, 20, 0, step.count) : 0}
                    </div>
                    <div style={{ fontSize: 11, color: COLORS.slate600, fontFamily: 'system-ui' }}>documents</div>
                  </div>
                </div>
                {i < rerankingSteps.length - 1 && (
                  <svg width="50" height="24" style={{ opacity: stepAnim.opacity }}>
                    <path
                      d="M0 12 L40 12 M35 7 L40 12 L35 17"
                      stroke={isActive ? step.color : COLORS.slate600}
                      strokeWidth="2"
                      fill="none"
                      strokeDasharray="50"
                      strokeDashoffset={50 - drawPath(localFrame, 30 + i * 30, 15) * 50}
                    />
                  </svg>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Model Details - Side by side */}
        <div
          style={{
            display: 'flex',
            gap: 30,
            width: '100%',
            maxWidth: 900,
          }}
        >
          {/* Cross-Encoder Card */}
          <div
            style={{
              flex: 1,
              backgroundColor: `${COLORS.purple500}10`,
              border: `2px solid ${COLORS.purple500}`,
              borderRadius: 16,
              padding: 22,
              opacity: springIn(localFrame, 80).opacity,
              transform: `translateY(${springIn(localFrame, 80).y}px)`,
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 700, color: COLORS.purple500, marginBottom: 14, fontFamily: 'system-ui' }}>
              Cross-Encoder Model
            </div>
            <div style={{ fontSize: 15, color: COLORS.white, fontFamily: 'monospace', marginBottom: 12 }}>
              ms-marco-MiniLM-L-6-v2
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Parameters</span>
                <span style={{ color: COLORS.white, fontFamily: 'monospace', fontSize: 13 }}>22.7M</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Latency</span>
                <span style={{ color: COLORS.emerald500, fontFamily: 'monospace', fontSize: 13 }}>&lt;50ms</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>NDCG@10</span>
                <span style={{ color: COLORS.cyan500, fontFamily: 'monospace', fontSize: 13 }}>0.78</span>
              </div>
            </div>
          </div>

          {/* Context Assembly */}
          <div
            style={{
              flex: 1,
              backgroundColor: `${COLORS.emerald500}10`,
              border: `2px solid ${COLORS.emerald500}`,
              borderRadius: 16,
              padding: 22,
              opacity: springIn(localFrame, 95).opacity,
              transform: `translateY(${springIn(localFrame, 95).y}px)`,
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 700, color: COLORS.emerald500, marginBottom: 14, fontFamily: 'system-ui' }}>
              Context Assembly
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Max Context</span>
                <span style={{ color: COLORS.white, fontFamily: 'monospace', fontSize: 13 }}>4,096 tokens</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Chunk Size</span>
                <span style={{ color: COLORS.white, fontFamily: 'monospace', fontSize: 13 }}>512 tokens</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Overlap</span>
                <span style={{ color: COLORS.white, fontFamily: 'monospace', fontSize: 13 }}>50 tokens</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Top-K</span>
                <span style={{ color: COLORS.emerald500, fontFamily: 'monospace', fontSize: 13 }}>5 chunks</span>
              </div>
            </div>
          </div>
        </div>

        {/* RAGAS Metrics */}
        <div
          style={{
            marginTop: 25,
            display: 'flex',
            gap: 25,
            opacity: springIn(localFrame, 120).opacity,
          }}
        >
          {[
            { label: 'Faithfulness', value: 0.85, color: COLORS.cyan500 },
            { label: 'Relevancy', value: 0.82, color: COLORS.emerald500 },
            { label: 'Context Recall', value: 0.88, color: COLORS.purple500 },
          ].map((metric, i) => {
            const metricAnim = springIn(localFrame, 130 + i * 10);
            return (
              <div
                key={metric.label}
                style={{
                  textAlign: 'center',
                  opacity: metricAnim.opacity,
                  transform: `scale(${metricAnim.scale})`,
                }}
              >
                <div style={{ fontSize: 30, fontWeight: 700, color: metric.color, fontFamily: 'monospace' }}>
                  {(countUp(localFrame, 140, 30, 0, metric.value * 100) / 100).toFixed(2)}
                </div>
                <div style={{ fontSize: 12, color: COLORS.slate500, fontFamily: 'system-ui', marginTop: 2 }}>
                  {metric.label}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    );
  };

  const renderOutro = () => {
    const localFrame = frame - SCENE_MICRO.outro.start;
    const exitProgress = interpolate(localFrame, [0, 50], [0, 1], {
      easing: EASING.snappy,
      extrapolateRight: 'clamp',
    });

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: 1 - exitProgress,
          transform: `scale(${1 + exitProgress * 0.2})`,
        }}
      >
        <GradientText gradient={GRADIENTS.rag} fontSize={60} fontWeight={800}>
          RAG System Complete
        </GradientText>
      </AbsoluteFill>
    );
  };

  // Main render with camera transform
  const renderContent = () => {
    switch (microScene) {
      case 'title':
        return renderTitle();
      case 'pipeline':
        return renderPipeline();
      case 'architecture':
        return renderArchitecture();
      case 'retrieval':
        return renderRetrieval();
      case 'reranking':
        return renderReranking();
      case 'outro':
        return renderOutro();
      default:
        return renderTitle();
    }
  };

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.slate950,
        transform: `scale(${camera.scale}) translate(${camera.x}px, ${camera.y}px)`,
        transformOrigin: 'center center',
      }}
    >
      {renderContent()}
      <Cursor frame={frame} keyframes={cursorKeyframes} color={COLORS.white} size={24} />
    </AbsoluteFill>
  );
};
