import React from 'react';
import { COLORS } from '../../../utils/colors';
import { fadeIn, scale } from '../../../utils/animations';
import { secondsToFrames } from '../../../utils/timings';

interface CheckmarkProps {
  frame: number;
  startFrame?: number;
  size?: number;
  color?: string;
}

export const Checkmark: React.FC<CheckmarkProps> = ({
  frame,
  startFrame = 0,
  size = 24,
  color = COLORS.green500,
}) => {
  const relativeFrame = Math.max(0, frame - startFrame);
  const opacity = fadeIn(relativeFrame, 0, secondsToFrames(0.3));
  const scaleValue = scale(relativeFrame, 0, secondsToFrames(0.4), 0, 1);

  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        backgroundColor: color,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        opacity,
        transform: `scale(${scaleValue})`,
        boxShadow: `0 0 10px ${color}80`,
      }}
    >
      <svg
        width={size * 0.6}
        height={size * 0.6}
        viewBox="0 0 24 24"
        fill="none"
        stroke={COLORS.white}
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="20 6 9 17 4 12" />
      </svg>
    </div>
  );
};
