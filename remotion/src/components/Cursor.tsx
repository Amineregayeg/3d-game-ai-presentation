import React from 'react';
import { interpolate, Easing } from 'remotion';
import { COLORS } from '../utils/colors';

// Cursor state types
export type CursorState = 'idle' | 'moving' | 'hover' | 'clicking' | 'clicked';

export interface CursorKeyframe {
  frame: number;
  x: number;
  y: number;
  state?: CursorState;
  scale?: number;
}

interface CursorProps {
  frame: number;
  keyframes: CursorKeyframe[];
  color?: string;
  size?: number;
  showTrail?: boolean;
}

// Spring easing for natural cursor movement
const springEasing = Easing.out(Easing.back(1.2));
const clickEasing = Easing.out(Easing.cubic);

// Interpolate cursor position with lag effect
const interpolateCursorPosition = (
  frame: number,
  keyframes: CursorKeyframe[],
  property: 'x' | 'y'
): number => {
  if (keyframes.length === 0) return 0;
  if (keyframes.length === 1) return keyframes[0][property];

  // Find surrounding keyframes
  let startIdx = 0;
  for (let i = 0; i < keyframes.length - 1; i++) {
    if (frame >= keyframes[i].frame && frame < keyframes[i + 1].frame) {
      startIdx = i;
      break;
    }
    if (frame >= keyframes[keyframes.length - 1].frame) {
      return keyframes[keyframes.length - 1][property];
    }
  }

  const start = keyframes[startIdx];
  const end = keyframes[startIdx + 1];
  const duration = end.frame - start.frame;

  // Add slight lag (3-frame delay simulation via easing)
  return interpolate(
    frame,
    [start.frame, start.frame + duration],
    [start[property], end[property]],
    {
      easing: springEasing,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
};

// Get cursor state at frame
const getCursorState = (frame: number, keyframes: CursorKeyframe[]): CursorState => {
  if (keyframes.length === 0) return 'idle';

  let currentState: CursorState = 'idle';
  for (const kf of keyframes) {
    if (frame >= kf.frame && kf.state) {
      currentState = kf.state;
    }
  }
  return currentState;
};

// Calculate cursor scale based on state
const getCursorScale = (
  frame: number,
  keyframes: CursorKeyframe[],
  baseScale: number = 1
): number => {
  const state = getCursorState(frame, keyframes);

  // Find the keyframe that set this state
  let stateFrame = 0;
  for (const kf of keyframes) {
    if (kf.state === state) {
      stateFrame = kf.frame;
    }
  }

  const timeSinceState = frame - stateFrame;

  switch (state) {
    case 'hover':
      // Slight scale up on hover (3-6 frames)
      return interpolate(
        timeSinceState,
        [0, 5],
        [baseScale, baseScale * 1.1],
        { easing: springEasing, extrapolateRight: 'clamp', extrapolateLeft: 'clamp' }
      );
    case 'clicking':
      // Scale down quickly (6-10 frames)
      return interpolate(
        timeSinceState,
        [0, 4, 8],
        [baseScale * 1.1, baseScale * 0.85, baseScale],
        { easing: clickEasing, extrapolateRight: 'clamp', extrapolateLeft: 'clamp' }
      );
    case 'clicked':
      return baseScale;
    default:
      return baseScale;
  }
};

// Get click ring opacity
const getClickRingOpacity = (frame: number, keyframes: CursorKeyframe[]): number => {
  // Find clicking keyframe
  let clickFrame = -1;
  for (const kf of keyframes) {
    if (kf.state === 'clicking' && frame >= kf.frame) {
      clickFrame = kf.frame;
    }
  }

  if (clickFrame < 0) return 0;

  const timeSinceClick = frame - clickFrame;
  if (timeSinceClick < 0 || timeSinceClick > 15) return 0;

  return interpolate(
    timeSinceClick,
    [0, 5, 15],
    [0.8, 0.5, 0],
    { extrapolateRight: 'clamp', extrapolateLeft: 'clamp' }
  );
};

// Get click ring scale
const getClickRingScale = (frame: number, keyframes: CursorKeyframe[]): number => {
  let clickFrame = -1;
  for (const kf of keyframes) {
    if (kf.state === 'clicking' && frame >= kf.frame) {
      clickFrame = kf.frame;
    }
  }

  if (clickFrame < 0) return 1;

  const timeSinceClick = frame - clickFrame;
  if (timeSinceClick < 0 || timeSinceClick > 15) return 1;

  return interpolate(
    timeSinceClick,
    [0, 15],
    [1, 2.5],
    { easing: Easing.out(Easing.quad), extrapolateRight: 'clamp', extrapolateLeft: 'clamp' }
  );
};

export const Cursor: React.FC<CursorProps> = ({
  frame,
  keyframes,
  color = COLORS.white,
  size = 24,
  showTrail = false,
}) => {
  if (keyframes.length === 0) return null;

  // Check if cursor should be visible
  const firstFrame = keyframes[0].frame;
  const lastFrame = keyframes[keyframes.length - 1].frame + 30; // Linger 30 frames after last keyframe

  if (frame < firstFrame || frame > lastFrame) return null;

  const x = interpolateCursorPosition(frame, keyframes, 'x');
  const y = interpolateCursorPosition(frame, keyframes, 'y');
  const scale = getCursorScale(frame, keyframes);
  const clickRingOpacity = getClickRingOpacity(frame, keyframes);
  const clickRingScale = getClickRingScale(frame, keyframes);
  const state = getCursorState(frame, keyframes);

  // Fade in/out at edges
  const opacity = interpolate(
    frame,
    [firstFrame, firstFrame + 8, lastFrame - 8, lastFrame],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        transform: `translate(-2px, -2px) scale(${scale})`,
        opacity,
        pointerEvents: 'none',
        zIndex: 9999,
      }}
    >
      {/* Click ring effect */}
      {clickRingOpacity > 0 && (
        <div
          style={{
            position: 'absolute',
            left: size / 2 - 15,
            top: size / 2 - 15,
            width: 30,
            height: 30,
            borderRadius: '50%',
            border: `2px solid ${color}`,
            opacity: clickRingOpacity,
            transform: `scale(${clickRingScale})`,
          }}
        />
      )}

      {/* Cursor SVG */}
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        style={{
          filter: state === 'hover' ? `drop-shadow(0 0 8px ${color})` : `drop-shadow(0 2px 4px rgba(0,0,0,0.3))`,
        }}
      >
        <path
          d="M4 4L10.5 20.5L13 13L20.5 10.5L4 4Z"
          fill={color}
          stroke={COLORS.slate900}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
};

// Helper to create cursor keyframes easily
export const createCursorPath = (
  baseFrame: number,
  points: Array<{ x: number; y: number; delay?: number; action?: 'hover' | 'click' }>
): CursorKeyframe[] => {
  const keyframes: CursorKeyframe[] = [];
  let currentFrame = baseFrame;

  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    const delay = point.delay ?? 12; // Default 12 frames between points

    if (i > 0) {
      currentFrame += delay;
    }

    keyframes.push({
      frame: currentFrame,
      x: point.x,
      y: point.y,
      state: 'moving',
    });

    if (point.action === 'hover') {
      // Hover state after arriving (3-6 frames)
      keyframes.push({
        frame: currentFrame + 4,
        x: point.x,
        y: point.y,
        state: 'hover',
      });
    } else if (point.action === 'click') {
      // Hover briefly then click
      keyframes.push({
        frame: currentFrame + 4,
        x: point.x,
        y: point.y,
        state: 'hover',
      });
      keyframes.push({
        frame: currentFrame + 10,
        x: point.x,
        y: point.y,
        state: 'clicking',
      });
      keyframes.push({
        frame: currentFrame + 18,
        x: point.x,
        y: point.y,
        state: 'clicked',
      });
    }
  }

  return keyframes;
};

export default Cursor;
