import React from 'react';
import { COLORS, THEMES } from '../../../utils/colors';
import { entranceAnimation } from '../../../utils/animations';
import { secondsToFrames } from '../../../utils/timings';

type ComponentType = 'voxformer' | 'rag' | 'avatar' | 'mcp';

interface ComponentCardProps {
  frame: number;
  startFrame?: number;
  type: ComponentType;
  title: string;
  subtitle?: string;
  icon?: string;
  isActive?: boolean;
  isCompleted?: boolean;
  width?: number;
}

const ICONS: Record<ComponentType, string> = {
  voxformer: '🎙️',
  rag: '🔍',
  avatar: '🗣️',
  mcp: '🎨',
};

export const ComponentCard: React.FC<ComponentCardProps> = ({
  frame,
  startFrame = 0,
  type,
  title,
  subtitle,
  icon,
  isActive = false,
  isCompleted = false,
  width = 400,
}) => {
  const theme = THEMES[type];
  const relativeFrame = Math.max(0, frame - startFrame);
  const { opacity, transform } = entranceAnimation(
    relativeFrame,
    0,
    secondsToFrames(0.5),
    { fadeIn: true, slideFromY: 20 }
  );

  return (
    <div
      style={{
        width,
        padding: '16px 24px',
        backgroundColor: isActive
          ? `${theme.primary}20`
          : `${COLORS.slate800}80`,
        borderRadius: 12,
        border: `2px solid ${isActive ? theme.primary : COLORS.slate700}`,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        opacity,
        transform,
        boxShadow: isActive ? `0 0 20px ${theme.glow}` : 'none',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Icon */}
      <div
        style={{
          fontSize: 32,
          width: 48,
          height: 48,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: `${theme.primary}30`,
          borderRadius: 8,
        }}
      >
        {icon || ICONS[type]}
      </div>

      {/* Content */}
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 18,
            fontWeight: 600,
            color: COLORS.white,
            fontFamily: 'system-ui, -apple-system, sans-serif',
          }}
        >
          {title}
        </div>
        {subtitle && (
          <div
            style={{
              fontSize: 14,
              color: COLORS.slate600,
              marginTop: 4,
              fontFamily: 'system-ui, -apple-system, sans-serif',
            }}
          >
            {subtitle}
          </div>
        )}
      </div>

      {/* Status indicator */}
      <div
        style={{
          width: 24,
          height: 24,
          borderRadius: '50%',
          border: `2px solid ${isCompleted ? COLORS.green500 : COLORS.slate600}`,
          backgroundColor: isCompleted ? COLORS.green500 : 'transparent',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {isCompleted && (
          <svg
            width={14}
            height={14}
            viewBox="0 0 24 24"
            fill="none"
            stroke={COLORS.white}
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
        )}
      </div>
    </div>
  );
};
