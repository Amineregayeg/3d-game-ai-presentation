import React from 'react';
import { AbsoluteFill, interpolate, Easing } from 'remotion';
import { COLORS, THEMES, GRADIENTS } from '../../../utils/colors';
import { EASING, highlightDim } from '../../../utils/animations';
import { GradientText } from '../components';
import { Cursor, CursorKeyframe } from '../../../components/Cursor';

interface S4_AvatarProps {
  frame: number;
}

const METRICS = [
  { label: 'TTFB', value: 75, unit: 'ms' },
  { label: 'Voice Quality', value: 4.14, unit: ' MOS', isDecimal: true },
  { label: 'Sync Accuracy', value: 95, unit: '%' },
];

// Pipeline stages
const PIPELINE_STAGES = [
  { id: 'text', label: 'LLM Response', icon: '💬', color: COLORS.cyan500 },
  { id: 'tts', label: 'ElevenLabs TTS', icon: '🔊', color: COLORS.rose500 },
  { id: 'audio', label: 'Audio Stream', icon: '🎵', color: COLORS.purple500 },
  { id: 'lipsync', label: 'Lip-Sync', icon: '👄', color: COLORS.amber500 },
  { id: 'avatar', label: 'Avatar Video', icon: '🎬', color: COLORS.emerald500 },
];

// Scene duration: 35s = 1050 frames
const SCENE_MICRO = {
  title: { start: 0, duration: 75 },         // 0-2.5s
  pipeline: { start: 75, duration: 240 },    // 2.5-10.5s
  tts: { start: 315, duration: 300 },        // 10.5-20.5s
  lipsync: { start: 615, duration: 360 },    // 20.5-32.5s
  outro: { start: 975, duration: 75 },       // 32.5-35s
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

// Count-up animation
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

// Waveform bar heights driven by frame
const getWaveformBars = (frame: number, count: number = 12): number[] => {
  return Array.from({ length: count }, (_, i) => {
    const phase = i * 0.6 + frame * 0.12;
    return Math.sin(phase) * 18 + 22;
  });
};

export const S4_Avatar: React.FC<S4_AvatarProps> = ({ frame }) => {
  const theme = THEMES.avatar;

  // Determine current micro-scene
  const getMicroScene = (): string => {
    if (frame < SCENE_MICRO.pipeline.start) return 'title';
    if (frame < SCENE_MICRO.tts.start) return 'pipeline';
    if (frame < SCENE_MICRO.lipsync.start) return 'tts';
    if (frame < SCENE_MICRO.outro.start) return 'lipsync';
    return 'outro';
  };

  const microScene = getMicroScene();

  // Cursor keyframes for the entire scene
  const cursorKeyframes: CursorKeyframe[] = [
    // Pipeline navigation
    { frame: 90, x: 180, y: 280, state: 'moving' },
    { frame: 120, x: 180, y: 280, state: 'hover' },
    { frame: 150, x: 340, y: 280, state: 'moving' },
    { frame: 180, x: 340, y: 280, state: 'clicking' },
    { frame: 210, x: 500, y: 280, state: 'moving' },
    { frame: 240, x: 660, y: 280, state: 'moving' },
    { frame: 270, x: 820, y: 280, state: 'moving' },
    { frame: 290, x: 820, y: 280, state: 'hover' },
    // TTS model selection
    { frame: 340, x: 350, y: 320, state: 'moving' },
    { frame: 380, x: 350, y: 320, state: 'hover' },
    { frame: 400, x: 350, y: 320, state: 'clicking' },
    { frame: 450, x: 750, y: 320, state: 'moving' },
    { frame: 490, x: 750, y: 320, state: 'hover' },
    { frame: 510, x: 750, y: 320, state: 'clicking' },
    // Streaming pipeline
    { frame: 550, x: 550, y: 520, state: 'moving' },
    { frame: 580, x: 550, y: 520, state: 'hover' },
    // Lip-sync model selection
    { frame: 640, x: 350, y: 350, state: 'moving' },
    { frame: 680, x: 350, y: 350, state: 'hover' },
    { frame: 700, x: 350, y: 350, state: 'clicking' },
    { frame: 780, x: 750, y: 350, state: 'moving' },
    { frame: 820, x: 750, y: 350, state: 'hover' },
    { frame: 840, x: 750, y: 350, state: 'clicking' },
    // Final metrics
    { frame: 900, x: 550, y: 550, state: 'moving' },
    { frame: 940, x: 550, y: 550, state: 'hover' },
  ];

  const renderTitle = () => {
    const localFrame = frame;
    const titleAnim = springIn(localFrame, 0);
    const subtitleAnim = springIn(localFrame, 12);

    // Exit animation
    const exitProgress = interpolate(localFrame, [55, 75], [0, 1], {
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
          transform: `scale(${1 + exitProgress * 0.1}) translateY(${-exitProgress * 40}px)`,
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
            Component 3
          </span>
        </div>

        <div
          style={{
            opacity: titleAnim.opacity,
            transform: `scale(${titleAnim.scale}) translateY(${titleAnim.y}px)`,
          }}
        >
          <GradientText gradient={GRADIENTS.avatar} fontSize={72} fontWeight={800}>
            Avatar TTS + LipSync
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
            Bringing NPCs to Life
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            gap: 40,
            marginTop: 30,
            opacity: springIn(localFrame, 25).opacity,
          }}
        >
          {METRICS.map((metric, i) => {
            const metricAnim = springIn(localFrame, 28 + i * 6);
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
                    fontSize: 42,
                    fontWeight: 700,
                    color: theme.primary,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                  }}
                >
                  {metric.isDecimal
                    ? (countUp(localFrame, 35, 20, 0, metric.value * 100) / 100).toFixed(2)
                    : countUp(localFrame, 35, 20, 0, metric.value)}
                  {metric.unit}
                </div>
                <div
                  style={{
                    fontSize: 13,
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

    // Active stage follows cursor timing
    const activeStage = Math.min(4, Math.floor(localFrame / 40));

    // Waveform bars for audio visualization
    const waveformBars = getWaveformBars(localFrame);

    // Data packet animation through pipeline
    const packetProgress = interpolate(localFrame, [60, 200], [0, 1], {
      easing: Easing.inOut(Easing.quad),
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 50,
        }}
      >
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 40 }}>
          <GradientText gradient={GRADIENTS.avatar} fontSize={44} fontWeight={700}>
            Avatar Pipeline Architecture
          </GradientText>
        </div>

        {/* Pipeline Flow */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            position: 'relative',
            marginBottom: 40,
          }}
        >
          {PIPELINE_STAGES.map((stage, i) => {
            const stageAnim = springIn(localFrame, 10 + i * 10);
            const isActive = i <= activeStage;
            const isCurrent = i === activeStage;

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
                    gap: 10,
                    opacity: stageAnim.opacity,
                    transform: `scale(${stageAnim.scale * pulseScale}) translateY(${stageAnim.y}px)`,
                  }}
                >
                  <div
                    style={{
                      width: 70,
                      height: 70,
                      borderRadius: 14,
                      backgroundColor: isActive ? `${stage.color}30` : `${COLORS.slate800}50`,
                      border: `2px solid ${isActive ? stage.color : COLORS.slate700}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 28,
                      boxShadow: isCurrent ? `0 0 25px ${stage.color}50` : isActive ? `0 0 12px ${stage.color}25` : 'none',
                    }}
                  >
                    {stage.icon}
                  </div>
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: isActive ? stage.color : COLORS.slate500,
                      fontFamily: 'system-ui, -apple-system, sans-serif',
                      textAlign: 'center',
                      width: 90,
                    }}
                  >
                    {stage.label}
                  </span>
                </div>
                {i < PIPELINE_STAGES.length - 1 && (
                  <svg width="35" height="20" style={{ opacity: stageAnim.opacity }}>
                    <defs>
                      <linearGradient id={`avatar-arrow-${i}`} x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor={stage.color} />
                        <stop offset="100%" stopColor={PIPELINE_STAGES[i + 1].color} />
                      </linearGradient>
                    </defs>
                    <path
                      d="M0 10 L25 10 M20 5 L25 10 L20 15"
                      stroke={i < activeStage ? `url(#avatar-arrow-${i})` : COLORS.slate600}
                      strokeWidth="2"
                      fill="none"
                      strokeDasharray="35"
                      strokeDashoffset={35 - drawPath(localFrame, 25 + i * 15, 12) * 35}
                    />
                  </svg>
                )}
              </React.Fragment>
            );
          })}

          {/* Animated data packet */}
          {packetProgress > 0 && packetProgress < 1 && (
            <div
              style={{
                position: 'absolute',
                left: interpolate(packetProgress, [0, 1], [50, 780]),
                top: 25,
                width: 16,
                height: 16,
                borderRadius: '50%',
                background: `linear-gradient(135deg, ${COLORS.cyan500}, ${COLORS.emerald500})`,
                boxShadow: `0 0 15px ${COLORS.cyan500}`,
                opacity: interpolate(packetProgress, [0, 0.05, 0.95, 1], [0, 1, 1, 0]),
              }}
            />
          )}
        </div>

        {/* Architecture Diagram with waveform */}
        <div
          style={{
            position: 'relative',
            width: 950,
            height: 320,
            opacity: springIn(localFrame, 60).opacity,
          }}
        >
          <svg width="100%" height="100%" viewBox="0 0 950 320">
            {/* LLM Response */}
            <g opacity={highlightDim(activeStage === 0, localFrame, 0)}>
              <rect x="30" y="80" width="140" height="120" rx="14" fill={`${COLORS.cyan500}15`} stroke={COLORS.cyan500} strokeWidth="2" />
              <text x="100" y="115" fill={COLORS.cyan500} fontSize="14" fontWeight="600" textAnchor="middle" fontFamily="system-ui">LLM</text>
              <rect x="45" y="130" width="110" height="35" rx="6" fill={`${COLORS.cyan500}20`} />
              <text x="100" y="152" fill={COLORS.slate400} fontSize="10" textAnchor="middle" fontFamily="monospace">"Hello world..."</text>
            </g>

            {/* ElevenLabs TTS */}
            <g opacity={highlightDim(activeStage === 1, localFrame, 0)}>
              <rect x="210" y="50" width="180" height="180" rx="14" fill={`${COLORS.rose500}15`} stroke={COLORS.rose500} strokeWidth="2" />
              <text x="300" y="85" fill={COLORS.rose500} fontSize="16" fontWeight="700" textAnchor="middle" fontFamily="system-ui">ElevenLabs</text>
              <rect x="230" y="105" width="140" height="50" rx="8" fill={`${COLORS.rose500}20`} />
              <text x="300" y="128" fill={COLORS.white} fontSize="12" fontWeight="600" textAnchor="middle" fontFamily="system-ui">Turbo v2.5</text>
              <text x="300" y="145" fill={COLORS.slate500} fontSize="10" textAnchor="middle" fontFamily="monospace">~75ms TTFB</text>
              <rect x="230" y="165" width="140" height="40" rx="8" fill={`${COLORS.emerald500}20`} stroke={COLORS.emerald500} strokeWidth="1" />
              <text x="300" y="190" fill={COLORS.emerald500} fontSize="11" fontWeight="600" textAnchor="middle" fontFamily="system-ui">WebSocket Stream</text>
            </g>

            {/* Audio Processing with animated waveform */}
            <g opacity={highlightDim(activeStage === 2, localFrame, 0)}>
              <rect x="430" y="70" width="150" height="150" rx="14" fill={`${COLORS.purple500}15`} stroke={COLORS.purple500} strokeWidth="2" />
              <text x="505" y="100" fill={COLORS.purple500} fontSize="14" fontWeight="600" textAnchor="middle" fontFamily="system-ui">Audio</text>

              {/* Animated waveform */}
              <g transform="translate(450, 120)">
                {waveformBars.map((height, i) => (
                  <rect
                    key={i}
                    x={i * 9}
                    y={40 - height / 2}
                    width="5"
                    height={height}
                    rx="2"
                    fill={COLORS.purple500}
                    opacity={0.5 + Math.sin(i * 0.5 + localFrame * 0.1) * 0.3}
                  />
                ))}
              </g>

              <text x="505" y="195" fill={COLORS.slate500} fontSize="10" textAnchor="middle" fontFamily="monospace">44.1kHz PCM</text>
            </g>

            {/* Lip-Sync */}
            <g opacity={highlightDim(activeStage === 3, localFrame, 0)}>
              <rect x="620" y="50" width="160" height="180" rx="14" fill={`${COLORS.amber500}15`} stroke={COLORS.amber500} strokeWidth="2" />
              <text x="700" y="85" fill={COLORS.amber500} fontSize="14" fontWeight="600" textAnchor="middle" fontFamily="system-ui">Lip-Sync</text>
              <rect x="640" y="105" width="120" height="40" rx="8" fill={`${COLORS.cyan500}20`} stroke={COLORS.cyan500} strokeWidth="1" />
              <text x="700" y="130" fill={COLORS.cyan500} fontSize="11" fontWeight="600" textAnchor="middle" fontFamily="system-ui">Wav2Lip</text>
              <rect x="640" y="155" width="120" height="40" rx="8" fill={`${COLORS.emerald500}20`} stroke={COLORS.emerald500} strokeWidth="1" />
              <text x="700" y="180" fill={COLORS.emerald500} fontSize="11" fontWeight="600" textAnchor="middle" fontFamily="system-ui">SadTalker</text>
            </g>

            {/* Avatar Output */}
            <g opacity={highlightDim(activeStage === 4, localFrame, 0)}>
              <rect x="820" y="80" width="110" height="120" rx="14" fill={`${COLORS.emerald500}15`} stroke={COLORS.emerald500} strokeWidth="2" />
              <text x="875" y="115" fill={COLORS.emerald500} fontSize="14" fontWeight="600" textAnchor="middle" fontFamily="system-ui">Avatar</text>
              <circle cx="875" cy="160" r="22" fill={`${COLORS.emerald500}30`} stroke={COLORS.emerald500} strokeWidth="2" />
              <text x="875" y="167" fill={COLORS.white} fontSize="20" textAnchor="middle">🎭</text>
            </g>

            {/* Arrows */}
            <g opacity={springIn(localFrame, 80).opacity}>
              <path d="M170 140 L210 140" stroke={COLORS.cyan500} strokeWidth="2" markerEnd="url(#avatar-arrowhead)" />
              <path d="M390 140 L430 140" stroke={COLORS.rose500} strokeWidth="2" markerEnd="url(#avatar-arrowhead)" />
              <path d="M580 140 L620 140" stroke={COLORS.purple500} strokeWidth="2" markerEnd="url(#avatar-arrowhead)" />
              <path d="M780 140 L820 140" stroke={COLORS.amber500} strokeWidth="2" markerEnd="url(#avatar-arrowhead)" />
            </g>

            <defs>
              <marker id="avatar-arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill={COLORS.slate500} />
              </marker>
            </defs>
          </svg>
        </div>
      </AbsoluteFill>
    );
  };

  const renderTTS = () => {
    const localFrame = frame - SCENE_MICRO.tts.start;
    const titleAnim = springIn(localFrame, 0);

    // Which model is selected (cursor-driven timing)
    const selectedModel = localFrame < 150 ? 0 : 1;

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 50,
        }}
      >
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 30 }}>
          <GradientText gradient={GRADIENTS.avatar} fontSize={44} fontWeight={700}>
            ElevenLabs Text-to-Speech
          </GradientText>
        </div>

        {/* Voice Models */}
        <div
          style={{
            display: 'flex',
            gap: 30,
            width: '100%',
            maxWidth: 950,
            marginBottom: 30,
          }}
        >
          {/* Turbo v2.5 */}
          <div
            style={{
              flex: 1,
              opacity: highlightDim(selectedModel === 0, localFrame, 0),
              transform: `scale(${selectedModel === 0 ? 1.02 : 1})`,
              transition: 'transform 0.2s',
            }}
          >
            <div
              style={{
                backgroundColor: `${COLORS.rose500}10`,
                border: `2px solid ${COLORS.rose500}`,
                borderRadius: 18,
                padding: 24,
                boxShadow: selectedModel === 0 ? `0 0 25px ${COLORS.rose500}30` : 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <span style={{ fontSize: 22, fontWeight: 700, color: COLORS.rose500, fontFamily: 'system-ui' }}>
                  Turbo v2.5
                </span>
                <span
                  style={{
                    padding: '5px 10px',
                    backgroundColor: COLORS.emerald500,
                    borderRadius: 16,
                    fontSize: 11,
                    fontWeight: 600,
                    color: COLORS.white,
                    fontFamily: 'system-ui',
                  }}
                >
                  RECOMMENDED
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Latency (TTFB)</span>
                  <span style={{ color: COLORS.emerald500, fontFamily: 'monospace', fontWeight: 600, fontSize: 13 }}>~75ms</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Quality</span>
                  <span style={{ color: COLORS.white, fontFamily: 'system-ui', fontSize: 13 }}>High</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Streaming</span>
                  <span style={{ color: COLORS.emerald500, fontFamily: 'system-ui', fontSize: 13 }}>Yes</span>
                </div>
              </div>

              <div
                style={{
                  marginTop: 16,
                  padding: 14,
                  backgroundColor: `${COLORS.rose500}20`,
                  borderRadius: 10,
                }}
              >
                <div style={{ fontSize: 11, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>USE CASE</div>
                <div style={{ fontSize: 13, color: COLORS.white, fontFamily: 'system-ui' }}>
                  Real-time conversations, game NPCs
                </div>
              </div>
            </div>
          </div>

          {/* Multilingual v2 */}
          <div
            style={{
              flex: 1,
              opacity: highlightDim(selectedModel === 1, localFrame, 0),
              transform: `scale(${selectedModel === 1 ? 1.02 : 1})`,
              transition: 'transform 0.2s',
            }}
          >
            <div
              style={{
                backgroundColor: `${COLORS.purple500}10`,
                border: `2px solid ${COLORS.purple500}`,
                borderRadius: 18,
                padding: 24,
                boxShadow: selectedModel === 1 ? `0 0 25px ${COLORS.purple500}30` : 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <span style={{ fontSize: 22, fontWeight: 700, color: COLORS.purple500, fontFamily: 'system-ui' }}>
                  Multilingual v2
                </span>
                <span
                  style={{
                    padding: '5px 10px',
                    backgroundColor: `${COLORS.purple500}30`,
                    borderRadius: 16,
                    fontSize: 11,
                    fontWeight: 600,
                    color: COLORS.purple400,
                    fontFamily: 'system-ui',
                  }}
                >
                  PREMIUM
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Latency (TTFB)</span>
                  <span style={{ color: COLORS.amber500, fontFamily: 'monospace', fontWeight: 600, fontSize: 13 }}>~200ms</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Quality</span>
                  <span style={{ color: COLORS.purple400, fontFamily: 'system-ui', fontSize: 13 }}>Premium</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Streaming</span>
                  <span style={{ color: COLORS.emerald500, fontFamily: 'system-ui', fontSize: 13 }}>Yes</span>
                </div>
              </div>

              <div
                style={{
                  marginTop: 16,
                  padding: 14,
                  backgroundColor: `${COLORS.purple500}20`,
                  borderRadius: 10,
                }}
              >
                <div style={{ fontSize: 11, color: COLORS.slate500, marginBottom: 6, fontFamily: 'system-ui' }}>USE CASE</div>
                <div style={{ fontSize: 13, color: COLORS.white, fontFamily: 'system-ui' }}>
                  Cinematic cutscenes, narration
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Streaming Pipeline */}
        <div
          style={{
            width: '100%',
            maxWidth: 950,
            opacity: springIn(localFrame, 80).opacity,
            transform: `translateY(${springIn(localFrame, 80).y}px)`,
          }}
        >
          <div
            style={{
              backgroundColor: `${COLORS.cyan500}10`,
              border: `2px solid ${COLORS.cyan500}`,
              borderRadius: 16,
              padding: 24,
            }}
          >
            <div style={{ fontSize: 18, fontWeight: 700, color: COLORS.cyan500, marginBottom: 20, fontFamily: 'system-ui' }}>
              WebSocket Streaming Pipeline
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
              {[
                { label: 'Text Chunk', value: '~50 chars', color: COLORS.cyan500 },
                { label: 'Audio Buffer', value: '128 bytes', color: COLORS.purple500 },
                { label: 'Playback', value: 'Real-time', color: COLORS.emerald500 },
              ].map((item, i) => {
                const itemAnim = springIn(localFrame, 100 + i * 15);
                return (
                  <React.Fragment key={item.label}>
                    <div
                      style={{
                        textAlign: 'center',
                        opacity: itemAnim.opacity,
                        transform: `scale(${itemAnim.scale})`,
                      }}
                    >
                      <div style={{ fontSize: 24, fontWeight: 700, color: item.color, fontFamily: 'monospace' }}>
                        {item.value}
                      </div>
                      <div style={{ fontSize: 12, color: COLORS.slate500, marginTop: 6, fontFamily: 'system-ui' }}>
                        {item.label}
                      </div>
                    </div>
                    {i < 2 && (
                      <svg width="50" height="24" style={{ opacity: itemAnim.opacity }}>
                        <path
                          d="M0 12 L40 12 M35 7 L40 12 L35 17"
                          stroke={COLORS.slate600}
                          strokeWidth="2"
                          fill="none"
                          strokeDasharray="50"
                          strokeDashoffset={50 - drawPath(localFrame, 110 + i * 20, 15) * 50}
                        />
                      </svg>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </div>
      </AbsoluteFill>
    );
  };

  const renderLipSync = () => {
    const localFrame = frame - SCENE_MICRO.lipsync.start;
    const titleAnim = springIn(localFrame, 0);

    // Selected lip-sync method (cursor-driven)
    const selectedMethod = localFrame < 180 ? 0 : 1;

    const lipsyncMethods = [
      {
        name: 'Wav2Lip',
        color: COLORS.cyan500,
        pros: ['Real-time capable', 'Low VRAM (4GB)', 'Good lip accuracy'],
        cons: ['Lower video quality', 'Edge artifacts'],
        latency: '~50ms/frame',
        quality: 'Good',
        useCase: 'Real-time avatar',
      },
      {
        name: 'SadTalker',
        color: COLORS.emerald500,
        pros: ['High video quality', 'Natural head motion', 'Better expressions'],
        cons: ['Higher latency', 'More VRAM (8GB)'],
        latency: '~200ms/frame',
        quality: 'Excellent',
        useCase: 'Pre-rendered content',
      },
    ];

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 50,
        }}
      >
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 25 }}>
          <GradientText gradient={GRADIENTS.avatar} fontSize={44} fontWeight={700}>
            Lip-Sync Technology
          </GradientText>
        </div>

        {/* Method Cards */}
        <div
          style={{
            display: 'flex',
            gap: 30,
            width: '100%',
            maxWidth: 1000,
            marginBottom: 25,
          }}
        >
          {lipsyncMethods.map((method, idx) => (
            <div
              key={method.name}
              style={{
                flex: 1,
                opacity: highlightDim(selectedMethod === idx, localFrame, 0),
                transform: `scale(${selectedMethod === idx ? 1.02 : 1})`,
                transition: 'transform 0.2s',
              }}
            >
              <div
                style={{
                  backgroundColor: `${method.color}10`,
                  border: `2px solid ${method.color}`,
                  borderRadius: 18,
                  padding: 22,
                  height: '100%',
                  boxShadow: selectedMethod === idx ? `0 0 25px ${method.color}30` : 'none',
                }}
              >
                <div style={{ fontSize: 24, fontWeight: 700, color: method.color, marginBottom: 18, fontFamily: 'system-ui' }}>
                  {method.name}
                </div>

                {/* Metrics */}
                <div style={{ display: 'flex', gap: 25, marginBottom: 18 }}>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.slate500, marginBottom: 4, fontFamily: 'system-ui' }}>LATENCY</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: COLORS.white, fontFamily: 'monospace' }}>{method.latency}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.slate500, marginBottom: 4, fontFamily: 'system-ui' }}>QUALITY</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: method.color, fontFamily: 'system-ui' }}>{method.quality}</div>
                  </div>
                </div>

                {/* Pros */}
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.emerald500, marginBottom: 8, fontFamily: 'system-ui' }}>
                    Advantages
                  </div>
                  {method.pros.map((pro) => (
                    <div
                      key={pro}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        marginBottom: 5,
                        fontSize: 12,
                        color: COLORS.slate400,
                        fontFamily: 'system-ui',
                      }}
                    >
                      <span style={{ color: COLORS.emerald500 }}>✓</span>
                      {pro}
                    </div>
                  ))}
                </div>

                {/* Cons */}
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.amber500, marginBottom: 8, fontFamily: 'system-ui' }}>
                    Trade-offs
                  </div>
                  {method.cons.map((con) => (
                    <div
                      key={con}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        marginBottom: 5,
                        fontSize: 12,
                        color: COLORS.slate400,
                        fontFamily: 'system-ui',
                      }}
                    >
                      <span style={{ color: COLORS.amber500 }}>•</span>
                      {con}
                    </div>
                  ))}
                </div>

                {/* Use Case */}
                <div
                  style={{
                    padding: 12,
                    backgroundColor: `${method.color}20`,
                    borderRadius: 10,
                  }}
                >
                  <div style={{ fontSize: 10, color: COLORS.slate500, marginBottom: 5, fontFamily: 'system-ui' }}>BEST FOR</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: method.color, fontFamily: 'system-ui' }}>{method.useCase}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Final Metrics */}
        <div
          style={{
            display: 'flex',
            gap: 40,
            opacity: springIn(localFrame, 200).opacity,
          }}
        >
          {[
            { label: 'Sync Accuracy', value: 95, unit: '%', color: COLORS.emerald500 },
            { label: 'End-to-End Latency', value: 150, unit: 'ms', color: COLORS.cyan500 },
            { label: 'MOS Score', value: 4.14, unit: '', color: COLORS.purple500, isDecimal: true },
          ].map((metric, i) => {
            const metricAnim = springIn(localFrame, 220 + i * 12);
            return (
              <div
                key={metric.label}
                style={{
                  textAlign: 'center',
                  opacity: metricAnim.opacity,
                  transform: `scale(${metricAnim.scale})`,
                }}
              >
                <div style={{ fontSize: 36, fontWeight: 700, color: metric.color, fontFamily: 'monospace' }}>
                  {metric.isDecimal
                    ? (countUp(localFrame, 230, 25, 0, metric.value * 100) / 100).toFixed(2)
                    : countUp(localFrame, 230, 25, 0, metric.value)}
                  {metric.unit}
                </div>
                <div style={{ fontSize: 12, color: COLORS.slate500, fontFamily: 'system-ui', marginTop: 6 }}>
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
    const exitProgress = interpolate(localFrame, [0, 60], [0, 1], {
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
          transform: `scale(${1 + exitProgress * 0.15})`,
        }}
      >
        <GradientText gradient={GRADIENTS.avatar} fontSize={56} fontWeight={800}>
          Avatar System Complete
        </GradientText>
      </AbsoluteFill>
    );
  };

  // Main render
  const renderContent = () => {
    switch (microScene) {
      case 'title':
        return renderTitle();
      case 'pipeline':
        return renderPipeline();
      case 'tts':
        return renderTTS();
      case 'lipsync':
        return renderLipSync();
      case 'outro':
        return renderOutro();
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
