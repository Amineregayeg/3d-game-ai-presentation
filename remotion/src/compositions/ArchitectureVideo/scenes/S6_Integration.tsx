import React from 'react';
import { AbsoluteFill, interpolate, Easing } from 'remotion';
import { COLORS, GRADIENTS } from '../../../utils/colors';
import { EASING } from '../../../utils/animations';
import { GradientText } from '../components';
import { Cursor, CursorKeyframe } from '../../../components/Cursor';

interface S6_IntegrationProps {
  frame: number;
}

// Components to connect
const COMPONENTS = [
  { id: 'voxformer', label: 'VoxFormer STT', icon: '🎤', color: COLORS.cyan500, x: 150, y: 200 },
  { id: 'rag', label: 'RAG System', icon: '🔍', color: COLORS.emerald500, x: 450, y: 100 },
  { id: 'avatar', label: 'Avatar TTS', icon: '🎭', color: COLORS.rose500, x: 750, y: 200 },
  { id: 'mcp', label: 'MCP Bridge', icon: '🔌', color: COLORS.orange500, x: 450, y: 320 },
];

// Connection paths between components
const CONNECTIONS = [
  { from: 0, to: 1, label: 'Transcription' },
  { from: 1, to: 2, label: 'Response' },
  { from: 1, to: 3, label: 'Tool Calls' },
  { from: 3, to: 2, label: 'Assets' },
];

// Scene duration: 15s = 450 frames
const SCENE_MICRO = {
  title: { start: 0, duration: 90 },        // 0-3s
  buildup: { start: 90, duration: 240 },    // 3-11s
  finale: { start: 330, duration: 120 },    // 11-15s
};

// Spring animation helper
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
    scale: interpolate(progress, [0, 1], [0.85, 1]),
    y: interpolate(progress, [0, 1], [15, 0]),
  };
};

// Draw connection path animation
const drawConnection = (frame: number, start: number, duration: number): number => {
  return interpolate(frame, [start, start + duration], [0, 1], {
    easing: EASING.snappy,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

export const S6_Integration: React.FC<S6_IntegrationProps> = ({ frame }) => {
  // Determine current micro-scene
  const getMicroScene = (): string => {
    if (frame < SCENE_MICRO.buildup.start) return 'title';
    if (frame < SCENE_MICRO.finale.start) return 'buildup';
    return 'finale';
  };

  const microScene = getMicroScene();

  // Cursor keyframes - connecting components
  const cursorKeyframes: CursorKeyframe[] = [
    // Start at VoxFormer
    { frame: 100, x: 200, y: 240, state: 'moving' },
    { frame: 130, x: 200, y: 240, state: 'hover' },
    { frame: 145, x: 200, y: 240, state: 'clicking' },
    // Move to RAG
    { frame: 180, x: 500, y: 140, state: 'moving' },
    { frame: 210, x: 500, y: 140, state: 'clicking' },
    // Move to Avatar
    { frame: 250, x: 800, y: 240, state: 'moving' },
    { frame: 280, x: 800, y: 240, state: 'clicking' },
    // Move to MCP
    { frame: 310, x: 500, y: 360, state: 'moving' },
    { frame: 340, x: 500, y: 360, state: 'clicking' },
    // Center for finale
    { frame: 380, x: 500, y: 280, state: 'moving' },
    { frame: 410, x: 500, y: 280, state: 'hover' },
  ];

  const renderTitle = () => {
    const localFrame = frame;
    const titleAnim = springIn(localFrame, 0);
    const subtitleAnim = springIn(localFrame, 15);

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
          transform: `scale(${1 + exitProgress * 0.1})`,
        }}
      >
        <div
          style={{
            opacity: titleAnim.opacity,
            transform: `scale(${titleAnim.scale}) translateY(${titleAnim.y}px)`,
          }}
        >
          <GradientText gradient={GRADIENTS.rainbow} fontSize={64} fontWeight={800}>
            Complete System Integration
          </GradientText>
        </div>

        <div style={{ opacity: subtitleAnim.opacity, transform: `translateY(${subtitleAnim.y}px)` }}>
          <span
            style={{
              fontSize: 24,
              color: COLORS.slate600,
              fontFamily: 'system-ui, -apple-system, sans-serif',
            }}
          >
            All four components working together
          </span>
        </div>
      </AbsoluteFill>
    );
  };

  const renderBuildup = () => {
    const localFrame = frame - SCENE_MICRO.buildup.start;

    // Active component follows cursor
    const activeComponent = Math.min(3, Math.floor(localFrame / 50));

    // Active connection
    const activeConnection = Math.min(3, Math.floor((localFrame - 30) / 50));

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 50,
        }}
      >
        <div style={{ opacity: springIn(localFrame, 0).opacity, marginBottom: 30 }}>
          <GradientText gradient={GRADIENTS.rainbow} fontSize={40} fontWeight={700}>
            Integrated NPC Pipeline
          </GradientText>
        </div>

        {/* Integration diagram */}
        <div
          style={{
            position: 'relative',
            width: 900,
            height: 420,
          }}
        >
          <svg width="100%" height="100%" viewBox="0 0 900 420">
            {/* Connection paths - draw animated */}
            <g>
              {/* VoxFormer to RAG */}
              <path
                d={`M ${COMPONENTS[0].x + 50} ${COMPONENTS[0].y}
                   Q ${(COMPONENTS[0].x + COMPONENTS[1].x) / 2} ${COMPONENTS[0].y - 50}
                   ${COMPONENTS[1].x} ${COMPONENTS[1].y + 40}`}
                stroke={activeConnection >= 0 ? COLORS.cyan500 : COLORS.slate700}
                strokeWidth="3"
                fill="none"
                strokeDasharray="200"
                strokeDashoffset={200 - drawConnection(localFrame, 30, 30) * 200}
                opacity={0.8}
              />
              {/* RAG to Avatar */}
              <path
                d={`M ${COMPONENTS[1].x + 50} ${COMPONENTS[1].y + 40}
                   Q ${(COMPONENTS[1].x + COMPONENTS[2].x) / 2} ${COMPONENTS[1].y}
                   ${COMPONENTS[2].x - 50} ${COMPONENTS[2].y}`}
                stroke={activeConnection >= 1 ? COLORS.emerald500 : COLORS.slate700}
                strokeWidth="3"
                fill="none"
                strokeDasharray="200"
                strokeDashoffset={200 - drawConnection(localFrame, 80, 30) * 200}
                opacity={0.8}
              />
              {/* RAG to MCP */}
              <path
                d={`M ${COMPONENTS[1].x} ${COMPONENTS[1].y + 60}
                   L ${COMPONENTS[3].x} ${COMPONENTS[3].y - 40}`}
                stroke={activeConnection >= 2 ? COLORS.emerald500 : COLORS.slate700}
                strokeWidth="3"
                fill="none"
                strokeDasharray="150"
                strokeDashoffset={150 - drawConnection(localFrame, 130, 30) * 150}
                opacity={0.8}
              />
              {/* MCP to Avatar */}
              <path
                d={`M ${COMPONENTS[3].x + 50} ${COMPONENTS[3].y}
                   Q ${(COMPONENTS[3].x + COMPONENTS[2].x) / 2} ${COMPONENTS[3].y + 30}
                   ${COMPONENTS[2].x} ${COMPONENTS[2].y + 40}`}
                stroke={activeConnection >= 3 ? COLORS.orange500 : COLORS.slate700}
                strokeWidth="3"
                fill="none"
                strokeDasharray="200"
                strokeDashoffset={200 - drawConnection(localFrame, 180, 30) * 200}
                opacity={0.8}
              />
            </g>
          </svg>

          {/* Component nodes */}
          {COMPONENTS.map((comp, i) => {
            const compAnim = springIn(localFrame, 10 + i * 15);
            const isActive = i <= activeComponent;
            const isCurrent = i === activeComponent;

            const pulseScale = isCurrent
              ? interpolate(Math.sin(localFrame * 0.15), [-1, 1], [1, 1.05])
              : 1;

            return (
              <div
                key={comp.id}
                style={{
                  position: 'absolute',
                  left: comp.x - 50,
                  top: comp.y - 35,
                  width: 100,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 10,
                  opacity: compAnim.opacity,
                  transform: `scale(${compAnim.scale * pulseScale}) translateY(${compAnim.y}px)`,
                }}
              >
                <div
                  style={{
                    width: 70,
                    height: 70,
                    borderRadius: 16,
                    backgroundColor: isActive ? `${comp.color}30` : `${COLORS.slate800}50`,
                    border: `2px solid ${isActive ? comp.color : COLORS.slate700}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 32,
                    boxShadow: isCurrent ? `0 0 25px ${comp.color}50` : isActive ? `0 0 12px ${comp.color}25` : 'none',
                  }}
                >
                  {comp.icon}
                </div>
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: isActive ? comp.color : COLORS.slate500,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    textAlign: 'center',
                  }}
                >
                  {comp.label}
                </span>
              </div>
            );
          })}

          {/* Data flow labels */}
          {CONNECTIONS.map((conn, i) => {
            if (activeConnection < i) return null;
            const labelAnim = springIn(localFrame, 50 + i * 50);
            const fromComp = COMPONENTS[conn.from];
            const toComp = COMPONENTS[conn.to];
            const midX = (fromComp.x + toComp.x) / 2;
            const midY = (fromComp.y + toComp.y) / 2 - 20;

            return (
              <div
                key={`label-${i}`}
                style={{
                  position: 'absolute',
                  left: midX - 40,
                  top: midY,
                  padding: '4px 10px',
                  backgroundColor: COLORS.slate900,
                  borderRadius: 8,
                  fontSize: 10,
                  color: COLORS.slate400,
                  fontFamily: 'system-ui',
                  opacity: labelAnim.opacity,
                  transform: `scale(${labelAnim.scale})`,
                }}
              >
                {conn.label}
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    );
  };

  const renderFinale = () => {
    const localFrame = frame - SCENE_MICRO.finale.start;
    const titleAnim = springIn(localFrame, 0);

    // All components pulse together
    const allPulse = interpolate(Math.sin(localFrame * 0.12), [-1, 1], [0.95, 1.05]);

    // Exit animation
    const exitProgress = interpolate(localFrame, [90, 120], [0, 1], {
      easing: EASING.snappy,
      extrapolateRight: 'clamp',
    });

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 30,
          opacity: 1 - exitProgress,
          transform: `scale(${1 + exitProgress * 0.1})`,
        }}
      >
        <div
          style={{
            opacity: titleAnim.opacity,
            transform: `scale(${titleAnim.scale * allPulse})`,
          }}
        >
          <GradientText gradient={GRADIENTS.rainbow} fontSize={56} fontWeight={800}>
            System Ready
          </GradientText>
        </div>

        {/* All components in a row */}
        <div
          style={{
            display: 'flex',
            gap: 25,
            opacity: springIn(localFrame, 20).opacity,
            transform: `scale(${allPulse})`,
          }}
        >
          {COMPONENTS.map((comp, i) => {
            const compAnim = springIn(localFrame, 30 + i * 8);
            return (
              <div
                key={comp.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '12px 20px',
                  backgroundColor: `${comp.color}20`,
                  border: `2px solid ${comp.color}`,
                  borderRadius: 30,
                  opacity: compAnim.opacity,
                  transform: `scale(${compAnim.scale})`,
                  boxShadow: `0 0 20px ${comp.color}40`,
                }}
              >
                <span style={{ fontSize: 24 }}>{comp.icon}</span>
                <span
                  style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: comp.color,
                    fontFamily: 'system-ui',
                  }}
                >
                  {comp.label}
                </span>
              </div>
            );
          })}
        </div>

        <div style={{ opacity: springIn(localFrame, 60).opacity }}>
          <span
            style={{
              fontSize: 20,
              color: COLORS.slate500,
              fontFamily: 'system-ui',
            }}
          >
            Intelligent NPC Pipeline Complete
          </span>
        </div>
      </AbsoluteFill>
    );
  };

  // Main render
  const renderContent = () => {
    switch (microScene) {
      case 'title':
        return renderTitle();
      case 'buildup':
        return renderBuildup();
      case 'finale':
        return renderFinale();
      default:
        return renderTitle();
    }
  };

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.slate950 }}>
      {renderContent()}
      <Cursor frame={frame} keyframes={cursorKeyframes} color={COLORS.white} size={24} />
    </AbsoluteFill>
  );
};
