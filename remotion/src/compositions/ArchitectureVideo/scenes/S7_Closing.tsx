import React from 'react';
import { AbsoluteFill, interpolate } from 'remotion';
import { COLORS, GRADIENTS } from '../../../utils/colors';
import { EASING } from '../../../utils/animations';
import { GradientText } from '../components';

interface S7_ClosingProps {
  frame: number;
}

// Scene duration: 8s = 240 frames
const SCENE_DURATION = 240;

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

export const S7_Closing: React.FC<S7_ClosingProps> = ({ frame }) => {
  // Logo animation - quick entrance (not long pulse hold)
  const logoAnim = springIn(frame, 0);

  // Glow pulse - only 2 seconds of subtle pulse (60 frames)
  const glowPulse = interpolate(
    Math.sin(frame * 0.1),
    [-1, 1],
    [0.6, 1],
  );

  // Tagline with staggered entrance
  const taglineAnim = springIn(frame, 30);

  // Subtitle
  const subtitleAnim = springIn(frame, 50);

  // CTA elements - staggered reveal
  const ctaItems = [
    { label: 'AI-Powered', icon: '🤖' },
    { label: 'Real-Time', icon: '⚡' },
    { label: 'Scalable', icon: '📈' },
  ];

  // Eased fade out starting at 180 frames (6s)
  const fadeOutProgress = interpolate(frame, [180, 230], [0, 1], {
    easing: EASING.snappy,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.slate950,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 30,
        opacity: 1 - fadeOutProgress,
        transform: `scale(${1 + fadeOutProgress * 0.1})`,
      }}
    >
      {/* Logo */}
      <div
        style={{
          position: 'relative',
          opacity: logoAnim.opacity,
          transform: `scale(${logoAnim.scale})`,
        }}
      >
        {/* Glow effect - subtle, short duration */}
        <div
          style={{
            position: 'absolute',
            inset: -50,
            background: `radial-gradient(circle, ${COLORS.cyanAlpha(glowPulse * 0.35)} 0%, transparent 70%)`,
            filter: 'blur(35px)',
          }}
        />

        <div
          style={{
            width: 160,
            height: 160,
            borderRadius: '50%',
            background: GRADIENTS.voxformer,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 0 ${40 * glowPulse}px ${COLORS.cyan500}50`,
          }}
        >
          <span
            style={{
              fontSize: 64,
              fontWeight: 800,
              color: COLORS.white,
              fontFamily: 'system-ui, -apple-system, sans-serif',
              letterSpacing: '-0.05em',
            }}
          >
            3D
          </span>
        </div>
      </div>

      {/* Tagline */}
      <div
        style={{
          opacity: taglineAnim.opacity,
          transform: `scale(${taglineAnim.scale}) translateY(${taglineAnim.y}px)`,
        }}
      >
        <GradientText gradient={GRADIENTS.rainbow} fontSize={44} fontWeight={700}>
          The Future of Intelligent NPCs
        </GradientText>
      </div>

      {/* Subtitle */}
      <div
        style={{
          opacity: subtitleAnim.opacity,
          transform: `translateY(${subtitleAnim.y}px)`,
        }}
      >
        <span
          style={{
            fontSize: 20,
            color: COLORS.slate500,
            fontFamily: 'system-ui, -apple-system, sans-serif',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
          }}
        >
          AI-Powered Game Development
        </span>
      </div>

      {/* CTA badges - staggered reveal */}
      <div
        style={{
          display: 'flex',
          gap: 20,
          marginTop: 20,
        }}
      >
        {ctaItems.map((item, i) => {
          const itemAnim = springIn(frame, 80 + i * 15);
          return (
            <div
              key={item.label}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 18px',
                backgroundColor: `${COLORS.slate800}80`,
                border: `1px solid ${COLORS.slate700}`,
                borderRadius: 30,
                opacity: itemAnim.opacity,
                transform: `scale(${itemAnim.scale}) translateY(${itemAnim.y}px)`,
              }}
            >
              <span style={{ fontSize: 18 }}>{item.icon}</span>
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: COLORS.slate400,
                  fontFamily: 'system-ui',
                }}
              >
                {item.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Final CTA */}
      <div
        style={{
          marginTop: 20,
          opacity: springIn(frame, 130).opacity,
          transform: `scale(${springIn(frame, 130).scale})`,
        }}
      >
        <div
          style={{
            padding: '14px 32px',
            background: GRADIENTS.voxformer,
            borderRadius: 40,
            boxShadow: `0 0 25px ${COLORS.cyan500}40`,
          }}
        >
          <span
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: COLORS.white,
              fontFamily: 'system-ui',
              letterSpacing: '0.05em',
            }}
          >
            3D Game AI Assistant
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
