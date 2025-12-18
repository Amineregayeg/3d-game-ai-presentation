import React from 'react';
import { AbsoluteFill, interpolate, Easing } from 'remotion';
import { COLORS, GRADIENTS } from '../../../utils/colors';
import { fadeIn, scale, pulse, entranceAnimation, parallax, EASING } from '../../../utils/animations';
import { secondsToFrames, SCENES, TRANSITION_FRAMES } from '../../../utils/timings';
import { GradientText } from '../components';

interface S0_LogoIntroProps {
  frame: number;
}

export const S0_LogoIntro: React.FC<S0_LogoIntroProps> = ({ frame }) => {
  // Compressed timings (5 seconds total)
  const logoStart = 0;
  const titleStart = secondsToFrames(0.8);
  const taglineStart = secondsToFrames(1.3);
  const badgesStart = secondsToFrames(1.8);
  const transitionStart = secondsToFrames(3.5);

  // Parallax background drift
  const bgParallax = parallax(frame, SCENES.logo.duration, 15, 0.5);

  // Logo animations with spring easing
  const logoOpacity = fadeIn(frame, logoStart, secondsToFrames(0.4));
  const logoScale = scale(frame, logoStart, secondsToFrames(0.5), 0.6, 1, EASING.spring);
  const glowPulse = pulse(frame, secondsToFrames(1.5), 0.5, 1);

  // Title animations - faster
  const titleAnim = entranceAnimation(frame, titleStart, secondsToFrames(0.35), {
    fadeIn: true,
    slideFromY: 20,
  });

  // Tagline animations
  const taglineAnim = entranceAnimation(frame, taglineStart, secondsToFrames(0.3), {
    fadeIn: true,
    slideFromY: 15,
  });

  // Transition out with spring easing instead of linear
  const transitionProgress = interpolate(
    frame,
    [transitionStart, transitionStart + secondsToFrames(1.2)],
    [0, 1],
    { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  const finalScale = interpolate(transitionProgress, [0, 1], [1, 0.3]);
  const finalX = interpolate(transitionProgress, [0, 1], [0, -600]);
  const finalY = interpolate(transitionProgress, [0, 1], [0, -300]);
  const contentFade = interpolate(transitionProgress, [0, 0.7, 1], [1, 1, 0]);

  // Scene exit transition (zoom through)
  const exitStart = SCENES.logo.duration - TRANSITION_FRAMES;
  const exitProgress = interpolate(
    frame,
    [exitStart, SCENES.logo.duration],
    [0, 1],
    { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  const exitScale = interpolate(exitProgress, [0, 1], [1, 1.5]);
  const exitOpacity = interpolate(exitProgress, [0, 0.5, 1], [1, 1, 0]);

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 30,
        transform: `scale(${exitScale})`,
        opacity: exitOpacity,
      }}
    >
      {/* Animated background particles with parallax */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          overflow: 'hidden',
          opacity: 0.4,
          transform: `translate(${bgParallax.x}px, ${bgParallax.y}px)`,
        }}
      >
        {[...Array(15)].map((_, i) => {
          const x = (i * 137.5) % 100;
          const y = (i * 73.3) % 100;
          const size = 4 + (i % 4) * 2;
          const delay = i * 0.15;
          const particleOpacity = pulse(frame + secondsToFrames(delay), secondsToFrames(2), 0.3, 0.9);
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: `${x}%`,
                top: `${y}%`,
                width: size,
                height: size,
                borderRadius: '50%',
                backgroundColor: i % 2 === 0 ? COLORS.cyan500 : COLORS.purple500,
                opacity: particleOpacity,
                filter: 'blur(1px)',
              }}
            />
          );
        })}
      </div>

      {/* Main content container */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 30,
          transform: `scale(${finalScale}) translate(${finalX}px, ${finalY}px)`,
          opacity: contentFade,
        }}
      >
        {/* Logo */}
        <div
          style={{
            position: 'relative',
            opacity: logoOpacity,
            transform: `scale(${logoScale})`,
          }}
        >
          {/* Glow effect */}
          <div
            style={{
              position: 'absolute',
              inset: -35,
              background: `radial-gradient(circle, ${COLORS.cyanAlpha(glowPulse * 0.5)} 0%, transparent 70%)`,
              filter: 'blur(25px)',
            }}
          />

          {/* Logo circle */}
          <div
            style={{
              width: 160,
              height: 160,
              borderRadius: '50%',
              background: GRADIENTS.voxformer,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 0 ${35 * glowPulse}px ${COLORS.cyan500}60`,
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

        {/* Title */}
        <div
          style={{
            textAlign: 'center',
            opacity: titleAnim.opacity,
            transform: titleAnim.transform,
          }}
        >
          <GradientText
            gradient={GRADIENTS.rainbow}
            fontSize={64}
            fontWeight={800}
          >
            3D Game AI Assistant
          </GradientText>
        </div>

        {/* Tagline */}
        <div
          style={{
            opacity: taglineAnim.opacity,
            transform: taglineAnim.transform,
          }}
        >
          <span
            style={{
              fontSize: 24,
              color: COLORS.slate600,
              fontFamily: 'system-ui, -apple-system, sans-serif',
              fontWeight: 400,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}
          >
            Intelligent NPC Pipeline
          </span>
        </div>

        {/* Metrics badges - rapid stagger */}
        <div
          style={{
            display: 'flex',
            gap: 16,
            marginTop: 15,
            opacity: fadeIn(frame, badgesStart, secondsToFrames(0.3)),
          }}
        >
          {[
            { label: 'STT', color: COLORS.cyan500 },
            { label: 'RAG', color: COLORS.emerald500 },
            { label: 'TTS', color: COLORS.rose500 },
            { label: 'MCP', color: COLORS.orange500 },
          ].map((badge, i) => (
            <div
              key={badge.label}
              style={{
                padding: '6px 14px',
                backgroundColor: `${badge.color}20`,
                border: `1px solid ${badge.color}50`,
                borderRadius: 6,
                color: badge.color,
                fontSize: 13,
                fontWeight: 600,
                fontFamily: 'system-ui, -apple-system, sans-serif',
                opacity: fadeIn(
                  frame,
                  badgesStart + i * secondsToFrames(0.06),
                  secondsToFrames(0.2)
                ),
                transform: `scale(${scale(frame, badgesStart + i * secondsToFrames(0.06), secondsToFrames(0.25), 0.8, 1, EASING.spring)})`,
              }}
            >
              {badge.label}
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};
