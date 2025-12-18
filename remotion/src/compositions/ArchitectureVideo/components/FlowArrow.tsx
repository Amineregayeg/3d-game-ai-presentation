import React from 'react';
import { COLORS } from '../../../utils/colors';
import { drawPath } from '../../../utils/animations';
import { secondsToFrames } from '../../../utils/timings';

interface FlowArrowProps {
  frame: number;
  startFrame?: number;
  duration?: number;
  width?: number;
  direction?: 'right' | 'down' | 'left' | 'up';
  color?: string;
  strokeWidth?: number;
}

export const FlowArrow: React.FC<FlowArrowProps> = ({
  frame,
  startFrame = 0,
  duration = secondsToFrames(0.6),
  width = 60,
  direction = 'right',
  color = COLORS.cyan500,
  strokeWidth = 3,
}) => {
  const relativeFrame = Math.max(0, frame - startFrame);
  const pathLength = width + 20; // Account for arrow head
  const dashOffset = drawPath(relativeFrame, 0, duration, pathLength);

  // Calculate rotation based on direction
  const rotation = {
    right: 0,
    down: 90,
    left: 180,
    up: -90,
  }[direction];

  const arrowHeight = 20;

  return (
    <svg
      width={width + 15}
      height={arrowHeight}
      style={{
        transform: `rotate(${rotation}deg)`,
        overflow: 'visible',
      }}
    >
      {/* Arrow line */}
      <line
        x1={0}
        y1={arrowHeight / 2}
        x2={width}
        y2={arrowHeight / 2}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={pathLength}
        strokeDashoffset={dashOffset}
        strokeLinecap="round"
      />
      {/* Arrow head */}
      <polygon
        points={`${width},${arrowHeight / 2} ${width - 10},${arrowHeight / 2 - 6} ${width - 10},${arrowHeight / 2 + 6}`}
        fill={color}
        opacity={dashOffset < pathLength * 0.3 ? 1 : 0}
      />
    </svg>
  );
};
