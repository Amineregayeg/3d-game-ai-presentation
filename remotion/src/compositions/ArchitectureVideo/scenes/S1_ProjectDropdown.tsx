import React from 'react';
import { AbsoluteFill, interpolate } from 'remotion';
import { COLORS, THEMES } from '../../../utils/colors';
import { fadeIn, scale, entranceAnimation, springAnimation, EASING } from '../../../utils/animations';
import { secondsToFrames, STAGGER, SCENES, TRANSITION_FRAMES } from '../../../utils/timings';
import { GradientText } from '../components';
import { Cursor, createCursorPath } from '../../../components/Cursor';

interface S1_ProjectDropdownProps {
  frame: number;
}

const COMPONENTS = [
  { type: 'voxformer' as const, title: 'VoxFormer STT', subtitle: 'Speech-to-Text Transformer' },
  { type: 'rag' as const, title: 'Advanced RAG System', subtitle: 'Retrieval-Augmented Generation' },
  { type: 'avatar' as const, title: 'Avatar TTS + LipSync', subtitle: 'Text-to-Speech & Animation' },
  { type: 'mcp' as const, title: 'Blender MCP Bridge', subtitle: 'AI-Powered 3D Asset Generation' },
];

export const S1_ProjectDropdown: React.FC<S1_ProjectDropdownProps> = ({ frame }) => {
  // Compressed timings (10 seconds total)
  const cardAppearStart = secondsToFrames(0.3);
  const dropdownStart = secondsToFrames(0.8);
  const cursorEnterStart = secondsToFrames(1.5);

  // Cursor clicks each item in sequence
  const clickTimes = [
    secondsToFrames(2.2),   // Click VoxFormer
    secondsToFrames(3.5),   // Click RAG
    secondsToFrames(4.8),   // Click Avatar
    secondsToFrames(6.1),   // Click MCP
    secondsToFrames(7.4),   // Final click on VoxFormer
  ];

  // Main card animation with spring
  const cardAnim = entranceAnimation(frame, cardAppearStart, secondsToFrames(0.35), {
    fadeIn: true,
    scaleFrom: 0.92,
  });

  // Dropdown expansion with spring instead of linear
  const dropdownProgress = springAnimation(frame, dropdownStart, secondsToFrames(0.5), 1.5);
  const dropdownHeight = dropdownProgress * 340;

  // Arrow rotation with spring
  const arrowRotation = interpolate(
    frame,
    [dropdownStart, dropdownStart + secondsToFrames(0.4)],
    [0, 180],
    { easing: EASING.spring, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  // Determine which item is highlighted based on cursor clicks
  const getHighlightedIndex = () => {
    if (frame >= clickTimes[4]) return 0; // Final: VoxFormer
    if (frame >= clickTimes[3]) return 3; // MCP
    if (frame >= clickTimes[2]) return 2; // Avatar
    if (frame >= clickTimes[1]) return 1; // RAG
    if (frame >= clickTimes[0]) return 0; // VoxFormer
    return -1;
  };

  const highlightedIndex = getHighlightedIndex();

  // Cursor keyframes - visits each item
  const cursorKeyframes = createCursorPath(cursorEnterStart, [
    { x: 960, y: 200, delay: 0 },
    { x: 400, y: 370, delay: 18, action: 'click' },  // VoxFormer
    { x: 400, y: 445, delay: 30, action: 'click' },  // RAG
    { x: 400, y: 520, delay: 30, action: 'click' },  // Avatar
    { x: 400, y: 595, delay: 30, action: 'click' },  // MCP
    { x: 400, y: 370, delay: 30, action: 'click' },  // Back to VoxFormer
  ]);

  // Scene exit transition
  const exitStart = SCENES.dropdown.duration - TRANSITION_FRAMES;
  const exitProgress = interpolate(
    frame,
    [exitStart, SCENES.dropdown.duration],
    [0, 1],
    { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  const exitScale = interpolate(exitProgress, [0, 1], [1, 1.3]);
  const exitOpacity = interpolate(exitProgress, [0, 0.6, 1], [1, 1, 0]);

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transform: `scale(${exitScale})`,
        opacity: exitOpacity,
      }}
    >
      {/* Main container */}
      <div
        style={{
          width: 480,
          opacity: cardAnim.opacity,
          transform: cardAnim.transform,
        }}
      >
        {/* Header card */}
        <div
          style={{
            padding: '20px 28px',
            backgroundColor: `${COLORS.slate800}90`,
            borderRadius: 14,
            border: `1px solid ${COLORS.slate700}`,
            backdropFilter: 'blur(10px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <GradientText fontSize={26} fontWeight={700}>
                3D Game AI Assistant
              </GradientText>
              <div
                style={{
                  fontSize: 13,
                  color: COLORS.slate600,
                  marginTop: 3,
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                }}
              >
                4 Core Components
              </div>
            </div>

            {/* Dropdown arrow with spring rotation */}
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 7,
                backgroundColor: `${COLORS.cyan500}20`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transform: `rotate(${arrowRotation}deg)`,
              }}
            >
              <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={COLORS.cyan500} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </div>
          </div>
        </div>

        {/* Dropdown content */}
        <div style={{ height: dropdownHeight, overflow: 'hidden', marginTop: 6 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '6px 0' }}>
            {COMPONENTS.map((comp, index) => {
              const itemStart = dropdownStart + index * STAGGER.slow;
              const isHighlighted = highlightedIndex === index;
              const itemOpacity = fadeIn(frame, itemStart, secondsToFrames(0.2));
              const itemScale = scale(frame, itemStart, secondsToFrames(0.25), 0.9, 1, EASING.spring);

              // Click feedback animation
              const clickFrame = clickTimes[index === 0 && frame >= clickTimes[4] ? 4 : index];
              const clickActive = frame >= clickFrame && frame < clickFrame + 20;
              const clickScale = clickActive
                ? interpolate(frame, [clickFrame, clickFrame + 6, clickFrame + 15], [1, 0.96, 1], { extrapolateRight: 'clamp', extrapolateLeft: 'clamp' })
                : 1;

              // Highlight glow intensity
              const glowIntensity = isHighlighted
                ? interpolate(frame, [clickFrame, clickFrame + 10], [0, 1], { extrapolateRight: 'clamp', extrapolateLeft: 'clamp' })
                : 0;

              return (
                <div
                  key={comp.type}
                  style={{
                    opacity: itemOpacity,
                    transform: `scale(${itemScale * clickScale}) translateX(${isHighlighted ? 8 : 0}px)`,
                  }}
                >
                  <div
                    style={{
                      padding: '14px 18px',
                      backgroundColor: isHighlighted ? `${THEMES[comp.type].primary}25` : `${COLORS.slate800}60`,
                      borderRadius: 10,
                      border: `2px solid ${isHighlighted ? THEMES[comp.type].primary : COLORS.slate700}`,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 14,
                      boxShadow: isHighlighted ? `0 0 ${18 * glowIntensity}px ${THEMES[comp.type].glow}` : 'none',
                    }}
                  >
                    {/* Icon */}
                    <div
                      style={{
                        fontSize: 24,
                        width: 40,
                        height: 40,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: `${THEMES[comp.type].primary}30`,
                        borderRadius: 7,
                      }}
                    >
                      {comp.type === 'voxformer' && '🎙️'}
                      {comp.type === 'rag' && '🔍'}
                      {comp.type === 'avatar' && '🗣️'}
                      {comp.type === 'mcp' && '🎨'}
                    </div>

                    {/* Content */}
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 15, fontWeight: 600, color: COLORS.white, fontFamily: 'system-ui' }}>
                        {comp.title}
                      </div>
                      <div style={{ fontSize: 11, color: COLORS.slate600, marginTop: 2, fontFamily: 'system-ui' }}>
                        {comp.subtitle}
                      </div>
                    </div>

                    {/* Selection indicator */}
                    <div
                      style={{
                        width: 18,
                        height: 18,
                        borderRadius: '50%',
                        border: `2px solid ${isHighlighted ? THEMES[comp.type].primary : COLORS.slate600}`,
                        backgroundColor: isHighlighted ? THEMES[comp.type].primary : 'transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transform: `scale(${isHighlighted ? scale(frame, clickFrame, secondsToFrames(0.2), 0.5, 1, EASING.spring) : 1})`,
                      }}
                    >
                      {isHighlighted && (
                        <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke={COLORS.white} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Instruction text */}
      <div
        style={{
          position: 'absolute',
          bottom: 90,
          opacity: fadeIn(frame, secondsToFrames(2), secondsToFrames(0.3)),
        }}
      >
        <span style={{ fontSize: 15, color: COLORS.slate600, fontFamily: 'system-ui', letterSpacing: '0.05em' }}>
          Let's explore each component in detail
        </span>
      </div>

      {/* Cursor */}
      <Cursor frame={frame} keyframes={cursorKeyframes} color={COLORS.white} size={22} />
    </AbsoluteFill>
  );
};
