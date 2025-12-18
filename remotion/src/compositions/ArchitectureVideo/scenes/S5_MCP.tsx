import React from 'react';
import { AbsoluteFill, interpolate, Easing } from 'remotion';
import { COLORS, THEMES, GRADIENTS } from '../../../utils/colors';
import { EASING, highlightDim } from '../../../utils/animations';
import { GradientText } from '../components';
import { Cursor, CursorKeyframe } from '../../../components/Cursor';

interface S5_MCPProps {
  frame: number;
}

const METRICS = [
  { label: 'MCP Tools', value: 24, unit: '' },
  { label: 'Asset Sources', value: 5, unit: '' },
  { label: 'Socket Latency', value: 100, unit: 'ms', prefix: '<' },
];

// MCP Protocol layers
const PROTOCOL_LAYERS = [
  { id: 'client', label: 'Claude AI Client', icon: '🤖', color: COLORS.cyan500 },
  { id: 'mcp', label: 'MCP Protocol', icon: '🔌', color: COLORS.purple500 },
  { id: 'server', label: 'Blender Server', icon: '🎨', color: COLORS.orange500 },
  { id: 'blender', label: 'Blender Python', icon: '🐍', color: COLORS.emerald500 },
];

// Tool categories
const TOOL_CATEGORIES = [
  {
    name: 'Object Creation',
    color: COLORS.cyan500,
    tools: ['create_mesh', 'create_primitive', 'create_text', 'import_model'],
  },
  {
    name: 'Transformations',
    color: COLORS.emerald500,
    tools: ['transform_object', 'rotate_object', 'scale_object', 'duplicate'],
  },
  {
    name: 'Materials',
    color: COLORS.purple500,
    tools: ['create_material', 'assign_material', 'set_color', 'set_texture'],
  },
  {
    name: 'Animation',
    color: COLORS.rose500,
    tools: ['add_keyframe', 'create_animation', 'set_timeline', 'render_frame'],
  },
  {
    name: 'Scene',
    color: COLORS.amber500,
    tools: ['get_scene_info', 'list_objects', 'delete_object', 'clear_scene'],
  },
  {
    name: 'Export',
    color: COLORS.orange500,
    tools: ['export_glb', 'export_fbx', 'export_obj', 'render_image'],
  },
];

// Scene duration: 35s = 1050 frames
const SCENE_MICRO = {
  title: { start: 0, duration: 75 },          // 0-2.5s
  architecture: { start: 75, duration: 300 }, // 2.5-12.5s
  protocol: { start: 375, duration: 300 },    // 12.5-22.5s
  tools: { start: 675, duration: 300 },       // 22.5-32.5s
  outro: { start: 975, duration: 75 },        // 32.5-35s
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

export const S5_MCP: React.FC<S5_MCPProps> = ({ frame }) => {
  const theme = THEMES.mcp;

  // Determine current micro-scene
  const getMicroScene = (): string => {
    if (frame < SCENE_MICRO.architecture.start) return 'title';
    if (frame < SCENE_MICRO.protocol.start) return 'architecture';
    if (frame < SCENE_MICRO.tools.start) return 'protocol';
    if (frame < SCENE_MICRO.outro.start) return 'tools';
    return 'outro';
  };

  const microScene = getMicroScene();

  // Cursor keyframes for the entire scene
  const cursorKeyframes: CursorKeyframe[] = [
    // Architecture navigation
    { frame: 90, x: 180, y: 270, state: 'moving' },
    { frame: 120, x: 180, y: 270, state: 'hover' },
    { frame: 145, x: 180, y: 270, state: 'clicking' },
    { frame: 175, x: 380, y: 270, state: 'moving' },
    { frame: 205, x: 380, y: 270, state: 'hover' },
    { frame: 235, x: 580, y: 270, state: 'moving' },
    { frame: 265, x: 580, y: 270, state: 'clicking' },
    { frame: 295, x: 780, y: 270, state: 'moving' },
    { frame: 330, x: 780, y: 270, state: 'hover' },
    // Protocol section
    { frame: 400, x: 350, y: 350, state: 'moving' },
    { frame: 440, x: 350, y: 350, state: 'hover' },
    { frame: 460, x: 350, y: 350, state: 'clicking' },
    { frame: 510, x: 750, y: 350, state: 'moving' },
    { frame: 550, x: 750, y: 350, state: 'hover' },
    { frame: 600, x: 550, y: 520, state: 'moving' },
    { frame: 640, x: 550, y: 520, state: 'hover' },
    // Tools section - click through categories
    { frame: 700, x: 280, y: 250, state: 'moving' },
    { frame: 730, x: 280, y: 250, state: 'hover' },
    { frame: 750, x: 280, y: 250, state: 'clicking' },
    { frame: 790, x: 550, y: 250, state: 'moving' },
    { frame: 820, x: 550, y: 250, state: 'clicking' },
    { frame: 860, x: 820, y: 250, state: 'moving' },
    { frame: 890, x: 820, y: 250, state: 'clicking' },
    { frame: 920, x: 550, y: 420, state: 'moving' },
    { frame: 950, x: 550, y: 420, state: 'hover' },
  ];

  // Camera horizontal pan for tools section
  const getCameraPan = (): { x: number } => {
    if (microScene !== 'tools') return { x: 0 };
    const localFrame = frame - SCENE_MICRO.tools.start;
    const panProgress = interpolate(localFrame, [0, 150, 250], [0, -50, 0], {
      easing: Easing.inOut(Easing.quad),
      extrapolateRight: 'clamp',
    });
    return { x: panProgress };
  };

  const cameraPan = getCameraPan();

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
            Component 4
          </span>
        </div>

        <div
          style={{
            opacity: titleAnim.opacity,
            transform: `scale(${titleAnim.scale}) translateY(${titleAnim.y}px)`,
          }}
        >
          <GradientText gradient={GRADIENTS.mcp} fontSize={72} fontWeight={800}>
            Blender MCP Bridge
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
            AI-Powered 3D Asset Generation
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
                  {metric.prefix || ''}
                  {countUp(localFrame, 35, 20, 0, metric.value)}
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

  const renderArchitecture = () => {
    const localFrame = frame - SCENE_MICRO.architecture.start;
    const titleAnim = springIn(localFrame, 0);

    // Active layer follows cursor timing
    const activeLayer = Math.min(3, Math.floor(localFrame / 60));

    // Data packet flowing through layers
    const packetProgress = interpolate(localFrame, [80, 220], [0, 1], {
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
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 35 }}>
          <GradientText gradient={GRADIENTS.mcp} fontSize={44} fontWeight={700}>
            MCP System Architecture
          </GradientText>
        </div>

        {/* Protocol Layers Flow */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 25,
            marginBottom: 35,
            position: 'relative',
          }}
        >
          {PROTOCOL_LAYERS.map((layer, i) => {
            const layerAnim = springIn(localFrame, 10 + i * 12);
            const isActive = i <= activeLayer;
            const isCurrent = i === activeLayer;

            const pulseScale = isCurrent
              ? interpolate(Math.sin(localFrame * 0.12), [-1, 1], [1, 1.04])
              : 1;

            return (
              <React.Fragment key={layer.id}>
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 10,
                    opacity: layerAnim.opacity,
                    transform: `scale(${layerAnim.scale * pulseScale}) translateY(${layerAnim.y}px)`,
                  }}
                >
                  <div
                    style={{
                      width: 90,
                      height: 90,
                      borderRadius: 18,
                      backgroundColor: isActive ? `${layer.color}30` : `${COLORS.slate800}50`,
                      border: `2px solid ${isActive ? layer.color : COLORS.slate700}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 36,
                      boxShadow: isCurrent ? `0 0 25px ${layer.color}45` : isActive ? `0 0 12px ${layer.color}25` : 'none',
                    }}
                  >
                    {layer.icon}
                  </div>
                  <span
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: isActive ? layer.color : COLORS.slate500,
                      fontFamily: 'system-ui, -apple-system, sans-serif',
                      textAlign: 'center',
                      width: 110,
                    }}
                  >
                    {layer.label}
                  </span>
                </div>
                {i < PROTOCOL_LAYERS.length - 1 && (
                  <svg width="50" height="24" style={{ opacity: layerAnim.opacity }}>
                    <defs>
                      <linearGradient id={`mcp-arrow-${i}`} x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor={layer.color} />
                        <stop offset="100%" stopColor={PROTOCOL_LAYERS[i + 1].color} />
                      </linearGradient>
                    </defs>
                    <path
                      d="M0 12 L40 12 M35 7 L40 12 L35 17"
                      stroke={i < activeLayer ? `url(#mcp-arrow-${i})` : COLORS.slate600}
                      strokeWidth="2"
                      fill="none"
                      strokeDasharray="50"
                      strokeDashoffset={50 - drawPath(localFrame, 30 + i * 25, 15) * 50}
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
                left: interpolate(packetProgress, [0, 1], [60, 700]),
                top: 35,
                width: 16,
                height: 16,
                borderRadius: '50%',
                background: `linear-gradient(135deg, ${COLORS.purple500}, ${COLORS.orange500})`,
                boxShadow: `0 0 15px ${COLORS.purple500}`,
                opacity: interpolate(packetProgress, [0, 0.05, 0.95, 1], [0, 1, 1, 0]),
              }}
            />
          )}
        </div>

        {/* Architecture Diagram */}
        <div
          style={{
            position: 'relative',
            width: 900,
            height: 320,
            opacity: springIn(localFrame, 60).opacity,
          }}
        >
          <svg width="100%" height="100%" viewBox="0 0 900 320">
            {/* Claude AI Client */}
            <g opacity={highlightDim(activeLayer === 0, localFrame, 0)}>
              <rect x="20" y="40" width="180" height="240" rx="14" fill={`${COLORS.cyan500}15`} stroke={COLORS.cyan500} strokeWidth="2" />
              <text x="110" y="75" fill={COLORS.cyan500} fontSize="16" fontWeight="700" textAnchor="middle" fontFamily="system-ui">Claude AI</text>
              <rect x="40" y="95" width="140" height="35" rx="6" fill={`${COLORS.cyan500}20`} />
              <text x="110" y="117" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Natural Language</text>
              <rect x="40" y="140" width="140" height="35" rx="6" fill={`${COLORS.cyan500}20`} />
              <text x="110" y="162" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Intent Recognition</text>
              <rect x="40" y="185" width="140" height="35" rx="6" fill={`${COLORS.cyan500}20`} />
              <text x="110" y="207" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Tool Selection</text>
              <rect x="40" y="230" width="140" height="35" rx="6" fill={`${COLORS.emerald500}20`} stroke={COLORS.emerald500} strokeWidth="1" />
              <text x="110" y="252" fill={COLORS.emerald500} fontSize="11" textAnchor="middle" fontFamily="system-ui">MCP Client SDK</text>
            </g>

            {/* MCP Protocol */}
            <g opacity={highlightDim(activeLayer === 1, localFrame, 0)}>
              <rect x="250" y="80" width="160" height="160" rx="14" fill={`${COLORS.purple500}15`} stroke={COLORS.purple500} strokeWidth="2" />
              <text x="330" y="115" fill={COLORS.purple500} fontSize="15" fontWeight="700" textAnchor="middle" fontFamily="system-ui">MCP Protocol</text>
              <rect x="270" y="135" width="120" height="30" rx="5" fill={`${COLORS.purple500}20`} />
              <text x="330" y="155" fill={COLORS.white} fontSize="10" textAnchor="middle" fontFamily="monospace">JSON-RPC 2.0</text>
              <rect x="270" y="175" width="120" height="30" rx="5" fill={`${COLORS.purple500}20`} />
              <text x="330" y="195" fill={COLORS.white} fontSize="10" textAnchor="middle" fontFamily="monospace">stdio / WebSocket</text>
              <text x="330" y="225" fill={COLORS.slate500} fontSize="10" textAnchor="middle" fontFamily="system-ui">&lt;100ms latency</text>
            </g>

            {/* Blender MCP Server */}
            <g opacity={highlightDim(activeLayer === 2, localFrame, 0)}>
              <rect x="460" y="40" width="180" height="240" rx="14" fill={`${COLORS.orange500}15`} stroke={COLORS.orange500} strokeWidth="2" />
              <text x="550" y="75" fill={COLORS.orange500} fontSize="15" fontWeight="700" textAnchor="middle" fontFamily="system-ui">MCP Server</text>
              <rect x="480" y="95" width="140" height="35" rx="6" fill={`${COLORS.orange500}20`} />
              <text x="550" y="117" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Tool Registry</text>
              <rect x="480" y="140" width="140" height="35" rx="6" fill={`${COLORS.orange500}20`} />
              <text x="550" y="162" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Request Handler</text>
              <rect x="480" y="185" width="140" height="35" rx="6" fill={`${COLORS.orange500}20`} />
              <text x="550" y="207" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Response Builder</text>
              <rect x="480" y="230" width="140" height="35" rx="6" fill={`${COLORS.amber500}20`} stroke={COLORS.amber500} strokeWidth="1" />
              <text x="550" y="252" fill={COLORS.amber500} fontSize="11" textAnchor="middle" fontFamily="system-ui">24 Tools</text>
            </g>

            {/* Blender */}
            <g opacity={highlightDim(activeLayer === 3, localFrame, 0)}>
              <rect x="690" y="80" width="180" height="160" rx="14" fill={`${COLORS.emerald500}15`} stroke={COLORS.emerald500} strokeWidth="2" />
              <text x="780" y="115" fill={COLORS.emerald500} fontSize="15" fontWeight="700" textAnchor="middle" fontFamily="system-ui">Blender 4.x</text>
              <rect x="710" y="135" width="140" height="30" rx="5" fill={`${COLORS.emerald500}20`} />
              <text x="780" y="155" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Python API</text>
              <rect x="710" y="175" width="140" height="30" rx="5" fill={`${COLORS.emerald500}20`} />
              <text x="780" y="195" fill={COLORS.white} fontSize="11" textAnchor="middle" fontFamily="system-ui">Scene Graph</text>
              <text x="780" y="225" fill={COLORS.slate500} fontSize="10" textAnchor="middle" fontFamily="monospace">bpy module</text>
            </g>

            {/* Arrows */}
            <g opacity={springIn(localFrame, 100).opacity}>
              <path d="M200 160 L250 160" stroke={COLORS.cyan500} strokeWidth="2" markerEnd="url(#mcp-arrowhead)" />
              <path d="M410 160 L460 160" stroke={COLORS.purple500} strokeWidth="2" markerEnd="url(#mcp-arrowhead)" />
              <path d="M640 160 L690 160" stroke={COLORS.orange500} strokeWidth="2" markerEnd="url(#mcp-arrowhead)" />
            </g>

            <defs>
              <marker id="mcp-arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill={COLORS.slate500} />
              </marker>
            </defs>
          </svg>
        </div>
      </AbsoluteFill>
    );
  };

  const renderProtocol = () => {
    const localFrame = frame - SCENE_MICRO.protocol.start;
    const titleAnim = springIn(localFrame, 0);

    const messageTypes = [
      { type: 'Request', direction: '→', color: COLORS.cyan500, example: 'tools/call' },
      { type: 'Response', direction: '←', color: COLORS.emerald500, example: 'result/success' },
      { type: 'Notification', direction: '↔', color: COLORS.amber500, example: 'progress/update' },
    ];

    // Active message type
    const activeMsg = Math.min(2, Math.floor(localFrame / 80));

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
          <GradientText gradient={GRADIENTS.mcp} fontSize={44} fontWeight={700}>
            MCP Protocol Details
          </GradientText>
        </div>

        {/* Protocol Info Side by Side */}
        <div
          style={{
            display: 'flex',
            gap: 30,
            width: '100%',
            maxWidth: 1000,
            marginBottom: 30,
          }}
        >
          {/* JSON-RPC */}
          <div
            style={{
              flex: 1,
              opacity: springIn(localFrame, 20).opacity,
              transform: `translateY(${springIn(localFrame, 20).y}px)`,
            }}
          >
            <div
              style={{
                backgroundColor: `${COLORS.purple500}10`,
                border: `2px solid ${COLORS.purple500}`,
                borderRadius: 18,
                padding: 24,
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.purple500, marginBottom: 16, fontFamily: 'system-ui' }}>
                JSON-RPC 2.0
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 18 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Transport</span>
                  <span style={{ color: COLORS.white, fontFamily: 'monospace', fontSize: 13 }}>stdio / WebSocket</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Format</span>
                  <span style={{ color: COLORS.white, fontFamily: 'monospace', fontSize: 13 }}>JSON</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: COLORS.slate500, fontFamily: 'system-ui', fontSize: 13 }}>Streaming</span>
                  <span style={{ color: COLORS.emerald500, fontFamily: 'system-ui', fontSize: 13 }}>Supported</span>
                </div>
              </div>

              {/* Code example with typewriter effect */}
              <div
                style={{
                  backgroundColor: COLORS.slate900,
                  borderRadius: 10,
                  padding: 14,
                  fontFamily: 'monospace',
                  fontSize: 11,
                  color: COLORS.slate300,
                }}
              >
                <div style={{ color: COLORS.slate500 }}>// Tool call request</div>
                <div>{'{'}</div>
                <div style={{ paddingLeft: 14 }}>
                  <span style={{ color: COLORS.cyan400 }}>"method"</span>: <span style={{ color: COLORS.amber400 }}>"tools/call"</span>,
                </div>
                <div style={{ paddingLeft: 14 }}>
                  <span style={{ color: COLORS.cyan400 }}>"params"</span>: {'{'}
                </div>
                <div style={{ paddingLeft: 28 }}>
                  <span style={{ color: COLORS.cyan400 }}>"name"</span>: <span style={{ color: COLORS.amber400 }}>"create_cube"</span>
                </div>
                <div style={{ paddingLeft: 14 }}>{'}'}</div>
                <div>{'}'}</div>
              </div>
            </div>
          </div>

          {/* Message Types */}
          <div
            style={{
              flex: 1,
              opacity: springIn(localFrame, 40).opacity,
              transform: `translateY(${springIn(localFrame, 40).y}px)`,
            }}
          >
            <div
              style={{
                backgroundColor: `${COLORS.orange500}10`,
                border: `2px solid ${COLORS.orange500}`,
                borderRadius: 18,
                padding: 24,
              }}
            >
              <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.orange500, marginBottom: 16, fontFamily: 'system-ui' }}>
                Message Types
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {messageTypes.map((msg, i) => {
                  const msgAnim = springIn(localFrame, 60 + i * 20);
                  const isActive = i === activeMsg;

                  return (
                    <div
                      key={msg.type}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 14,
                        padding: 14,
                        backgroundColor: `${msg.color}15`,
                        border: `1px solid ${msg.color}`,
                        borderRadius: 10,
                        opacity: msgAnim.opacity,
                        transform: `scale(${isActive ? 1.02 : msgAnim.scale})`,
                        boxShadow: isActive ? `0 0 20px ${msg.color}30` : 'none',
                      }}
                    >
                      <span style={{ fontSize: 22, color: msg.color }}>{msg.direction}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 15, fontWeight: 600, color: msg.color, fontFamily: 'system-ui' }}>
                          {msg.type}
                        </div>
                        <div style={{ fontSize: 11, color: COLORS.slate500, fontFamily: 'monospace' }}>
                          {msg.example}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Capabilities */}
        <div
          style={{
            display: 'flex',
            gap: 25,
            opacity: springIn(localFrame, 150).opacity,
          }}
        >
          {[
            { label: 'Tools', value: 24, color: COLORS.cyan500 },
            { label: 'Resources', value: 5, color: COLORS.emerald500 },
            { label: 'Prompts', value: 8, color: COLORS.purple500 },
          ].map((cap, i) => {
            const capAnim = springIn(localFrame, 170 + i * 15);
            return (
              <div
                key={cap.label}
                style={{
                  textAlign: 'center',
                  padding: '16px 32px',
                  backgroundColor: `${cap.color}15`,
                  border: `2px solid ${cap.color}`,
                  borderRadius: 14,
                  opacity: capAnim.opacity,
                  transform: `scale(${capAnim.scale})`,
                }}
              >
                <div style={{ fontSize: 32, fontWeight: 700, color: cap.color, fontFamily: 'monospace' }}>
                  {countUp(localFrame, 180, 25, 0, cap.value)}
                </div>
                <div style={{ fontSize: 12, color: COLORS.slate500, fontFamily: 'system-ui', marginTop: 6 }}>
                  {cap.label}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    );
  };

  const renderTools = () => {
    const localFrame = frame - SCENE_MICRO.tools.start;
    const titleAnim = springIn(localFrame, 0);

    // Active category follows cursor clicks (batch reveal)
    const activeCategory = Math.min(5, Math.floor(localFrame / 40));

    return (
      <AbsoluteFill
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 50,
          transform: `translateX(${cameraPan.x}px)`,
        }}
      >
        <div style={{ opacity: titleAnim.opacity, transform: `translateY(${titleAnim.y}px)`, marginBottom: 25 }}>
          <GradientText gradient={GRADIENTS.mcp} fontSize={44} fontWeight={700}>
            24 MCP Tools
          </GradientText>
        </div>

        {/* Tool Categories Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 20,
            width: '100%',
            maxWidth: 1050,
          }}
        >
          {TOOL_CATEGORIES.map((category, catIdx) => {
            const catAnim = springIn(localFrame, 15 + catIdx * 10);
            const isActive = catIdx <= activeCategory;
            const isCurrent = catIdx === activeCategory;

            return (
              <div
                key={category.name}
                style={{
                  opacity: catAnim.opacity,
                  transform: `scale(${isCurrent ? 1.02 : catAnim.scale})`,
                }}
              >
                <div
                  style={{
                    backgroundColor: `${category.color}10`,
                    border: `2px solid ${category.color}`,
                    borderRadius: 14,
                    padding: 20,
                    boxShadow: isCurrent ? `0 0 20px ${category.color}30` : 'none',
                    opacity: highlightDim(isActive, localFrame, 0),
                  }}
                >
                  <div style={{ fontSize: 16, fontWeight: 700, color: category.color, marginBottom: 14, fontFamily: 'system-ui' }}>
                    {category.name}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {category.tools.map((tool, toolIdx) => {
                      // Tools reveal with stagger after category becomes active
                      const toolVisible = isActive && localFrame > 20 + catIdx * 40 + toolIdx * 6;
                      const toolAnim = toolVisible ? springIn(localFrame, 20 + catIdx * 40 + toolIdx * 6) : { opacity: 0, scale: 0.8, y: 10 };

                      return (
                        <div
                          key={tool}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 6,
                            opacity: toolAnim.opacity,
                            transform: `translateY(${toolAnim.y}px)`,
                          }}
                        >
                          <div
                            style={{
                              width: 5,
                              height: 5,
                              borderRadius: 2.5,
                              backgroundColor: category.color,
                            }}
                          />
                          <span
                            style={{
                              fontSize: 12,
                              color: COLORS.slate400,
                              fontFamily: 'monospace',
                            }}
                          >
                            {tool}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Asset Sources */}
        <div
          style={{
            marginTop: 25,
            display: 'flex',
            gap: 16,
            opacity: springIn(localFrame, 200).opacity,
          }}
        >
          {[
            { name: 'Sketchfab', icon: '🎨' },
            { name: 'Poly Haven', icon: '🏔️' },
            { name: 'Mixamo', icon: '🏃' },
            { name: 'Quixel', icon: '🌲' },
            { name: 'Local Files', icon: '📁' },
          ].map((source, i) => {
            const sourceAnim = springIn(localFrame, 220 + i * 10);
            return (
              <div
                key={source.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '10px 16px',
                  backgroundColor: `${COLORS.slate800}80`,
                  border: `1px solid ${COLORS.slate700}`,
                  borderRadius: 24,
                  opacity: sourceAnim.opacity,
                  transform: `scale(${sourceAnim.scale})`,
                }}
              >
                <span style={{ fontSize: 18 }}>{source.icon}</span>
                <span style={{ fontSize: 12, color: COLORS.slate400, fontFamily: 'system-ui' }}>{source.name}</span>
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
        <GradientText gradient={GRADIENTS.mcp} fontSize={56} fontWeight={800}>
          MCP Bridge Complete
        </GradientText>
      </AbsoluteFill>
    );
  };

  // Main render
  const renderContent = () => {
    switch (microScene) {
      case 'title':
        return renderTitle();
      case 'architecture':
        return renderArchitecture();
      case 'protocol':
        return renderProtocol();
      case 'tools':
        return renderTools();
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
