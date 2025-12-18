import React from 'react';
import { staticFile, Img } from 'remotion';
import { COLORS } from '../../../utils/colors';
import { fadeIn, scale, pulse } from '../../../utils/animations';
import { secondsToFrames } from '../../../utils/timings';

interface AnimatedLogoProps {
  frame: number;
  size?: number;
  showGlow?: boolean;
  glowColor?: string;
}

export const AnimatedLogo: React.FC<AnimatedLogoProps> = ({
  frame,
  size = 200,
  showGlow = true,
  glowColor = COLORS.cyan500,
}) => {
  // Animation values
  const opacity = fadeIn(frame, 0, secondsToFrames(0.8));
  const scaleValue = scale(frame, 0, secondsToFrames(0.8), 0.8, 1);
  const glowIntensity = showGlow ? pulse(frame, secondsToFrames(2), 0.3, 0.8) : 0;

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        opacity,
        transform: `scale(${scaleValue})`,
      }}
    >
      {/* Glow effect */}
      {showGlow && (
        <div
          style={{
            position: 'absolute',
            inset: -20,
            background: `radial-gradient(circle, ${glowColor}${Math.round(glowIntensity * 255).toString(16).padStart(2, '0')} 0%, transparent 70%)`,
            filter: 'blur(20px)',
            borderRadius: '50%',
          }}
        />
      )}

      {/* Logo placeholder - using a stylized text logo */}
      <div
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          background: `linear-gradient(135deg, ${COLORS.cyan500} 0%, ${COLORS.purple500} 100%)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: `0 0 ${30 * glowIntensity}px ${glowColor}`,
        }}
      >
        <span
          style={{
            fontSize: size * 0.4,
            fontWeight: 'bold',
            color: COLORS.white,
            fontFamily: 'system-ui, -apple-system, sans-serif',
            letterSpacing: '-0.05em',
          }}
        >
          3D
        </span>
      </div>
    </div>
  );
};
