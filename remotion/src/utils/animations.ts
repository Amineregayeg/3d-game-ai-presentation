import { interpolate, Easing } from 'remotion';

// Premium easing functions - NO LINEAR
export const EASING = {
  // Standard transitions - smooth and premium
  entrance: Easing.out(Easing.cubic),
  exit: Easing.in(Easing.cubic),
  smooth: Easing.inOut(Easing.quad),

  // Spring-based for snappy interactions
  spring: Easing.out(Easing.back(1.7)),
  springStrong: Easing.out(Easing.back(2.2)),
  springSubtle: Easing.out(Easing.back(1.2)),

  // Emphasis
  bounce: Easing.out(Easing.back(1.7)),
  elastic: Easing.out(Easing.elastic(1)),

  // UI elements - snappy but controlled
  snappy: Easing.out(Easing.quad),
  gentle: Easing.inOut(Easing.sine),

  // For counters and progress - use quad instead of linear
  counter: Easing.inOut(Easing.quad),

  // Fade out - use cubic instead of linear
  fadeOutSmooth: Easing.inOut(Easing.cubic),
};

// Fade in animation
export const fadeIn = (
  frame: number,
  startFrame: number,
  duration: number,
  easing = EASING.entrance
): number => {
  return interpolate(
    frame,
    [startFrame, startFrame + duration],
    [0, 1],
    {
      easing,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
};

// Fade out animation
export const fadeOut = (
  frame: number,
  startFrame: number,
  duration: number,
  easing = EASING.fadeOutSmooth
): number => {
  return interpolate(
    frame,
    [startFrame, startFrame + duration],
    [1, 0],
    {
      easing,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
};

// Scale animation (from -> to)
export const scale = (
  frame: number,
  startFrame: number,
  duration: number,
  from: number,
  to: number,
  easing = EASING.spring
): number => {
  return interpolate(
    frame,
    [startFrame, startFrame + duration],
    [from, to],
    {
      easing,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
};

// Slide animation (x or y position)
export const slide = (
  frame: number,
  startFrame: number,
  duration: number,
  from: number,
  to: number,
  easing = EASING.spring
): number => {
  return interpolate(
    frame,
    [startFrame, startFrame + duration],
    [from, to],
    {
      easing,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
};

// Combined transform for common entrance animations
export const entranceAnimation = (
  frame: number,
  startFrame: number,
  duration: number,
  options: {
    fadeIn?: boolean;
    scaleFrom?: number;
    slideFromX?: number;
    slideFromY?: number;
  } = {}
): {
  opacity: number;
  transform: string;
} => {
  const { fadeIn: doFade = true, scaleFrom, slideFromX, slideFromY } = options;

  const opacity = doFade
    ? interpolate(
        frame,
        [startFrame, startFrame + duration],
        [0, 1],
        { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
      )
    : 1;

  const transforms: string[] = [];

  if (scaleFrom !== undefined) {
    const scaleValue = interpolate(
      frame,
      [startFrame, startFrame + duration],
      [scaleFrom, 1],
      { easing: EASING.spring, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
    transforms.push(`scale(${scaleValue})`);
  }

  if (slideFromX !== undefined) {
    const translateX = interpolate(
      frame,
      [startFrame, startFrame + duration],
      [slideFromX, 0],
      { easing: EASING.spring, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
    transforms.push(`translateX(${translateX}px)`);
  }

  if (slideFromY !== undefined) {
    const translateY = interpolate(
      frame,
      [startFrame, startFrame + duration],
      [slideFromY, 0],
      { easing: EASING.spring, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
    transforms.push(`translateY(${translateY}px)`);
  }

  return {
    opacity,
    transform: transforms.length > 0 ? transforms.join(' ') : 'none',
  };
};

// Pulse animation (for glowing effects)
export const pulse = (
  frame: number,
  cycleDuration: number,
  minValue: number = 0.5,
  maxValue: number = 1
): number => {
  const cycleProgress = (frame % cycleDuration) / cycleDuration;
  const sineValue = Math.sin(cycleProgress * Math.PI * 2);
  return interpolate(sineValue, [-1, 1], [minValue, maxValue]);
};

// Draw animation for SVG paths (returns 0-1 progress value)
export const drawPath = (
  frame: number,
  startFrame: number,
  duration: number,
  easing = EASING.smooth
): number => {
  const progress = interpolate(
    frame,
    [startFrame, startFrame + duration],
    [0, 1],
    {
      easing,
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
  return progress;
};

// Counter animation (for number counting) - uses quad easing, not linear
export const countUp = (
  frame: number,
  startFrame: number,
  duration: number,
  from: number,
  to: number,
  easing = EASING.counter
): number => {
  return Math.round(
    interpolate(
      frame,
      [startFrame, startFrame + duration],
      [from, to],
      {
        easing,
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }
    )
  );
};

// Staggered animation helper
export const getStaggeredStart = (
  baseStart: number,
  index: number,
  staggerDelay: number
): number => {
  return baseStart + index * staggerDelay;
};

// Spring physics simulation
export const springAnimation = (
  frame: number,
  startFrame: number,
  duration: number,
  overshoot: number = 1.7
): number => {
  const progress = interpolate(
    frame,
    [startFrame, startFrame + duration],
    [0, 1],
    {
      easing: Easing.out(Easing.back(overshoot)),
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
  return progress;
};

// Highlight animation - dims to highlight
export const highlightDim = (
  frame: number,
  startFrame: number,
  duration: number,
  isActive: boolean
): number => {
  if (isActive) {
    return interpolate(
      frame,
      [startFrame, startFrame + duration],
      [0.3, 1],
      { easing: EASING.snappy, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
  }
  return interpolate(
    frame,
    [startFrame, startFrame + duration],
    [1, 0.3],
    { easing: EASING.snappy, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
};

// Camera zoom effect
export const cameraZoom = (
  frame: number,
  startFrame: number,
  duration: number,
  zoomLevel: number = 1.15,
  centerX: number = 50,
  centerY: number = 50
): { scale: number; transformOrigin: string } => {
  const scaleValue = interpolate(
    frame,
    [startFrame, startFrame + duration / 2, startFrame + duration],
    [1, zoomLevel, 1],
    { easing: EASING.smooth, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return {
    scale: scaleValue,
    transformOrigin: `${centerX}% ${centerY}%`,
  };
};

// Parallax effect for background layers
export const parallax = (
  frame: number,
  totalFrames: number,
  amplitude: number = 20,
  speed: number = 1
): { x: number; y: number } => {
  const progress = (frame / totalFrames) * speed;
  return {
    x: Math.sin(progress * Math.PI * 2) * amplitude,
    y: Math.cos(progress * Math.PI * 2) * amplitude * 0.5,
  };
};

// Scene transition: zoom through
export const zoomThroughTransition = (
  frame: number,
  startFrame: number,
  duration: number
): { scale: number; opacity: number } => {
  const scaleValue = interpolate(
    frame,
    [startFrame, startFrame + duration],
    [1, 3],
    { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  const opacityValue = interpolate(
    frame,
    [startFrame, startFrame + duration * 0.6, startFrame + duration],
    [1, 1, 0],
    { easing: EASING.smooth, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  return { scale: scaleValue, opacity: opacityValue };
};

// Scene transition: push
export const pushTransition = (
  frame: number,
  startFrame: number,
  duration: number,
  direction: 'left' | 'right' | 'up' | 'down' = 'left'
): { outgoing: number; incoming: number } => {
  const progress = interpolate(
    frame,
    [startFrame, startFrame + duration],
    [0, 1],
    { easing: EASING.spring, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  const multiplier = direction === 'left' || direction === 'up' ? -1 : 1;
  const isHorizontal = direction === 'left' || direction === 'right';
  const dimension = isHorizontal ? 1920 : 1080;

  return {
    outgoing: progress * dimension * multiplier,
    incoming: (1 - progress) * dimension * -multiplier,
  };
};

// Match cut scale transition
export const matchCutScale = (
  frame: number,
  startFrame: number,
  duration: number,
  targetScale: number = 0.15
): { scale: number; opacity: number } => {
  const midFrame = startFrame + duration / 2;

  if (frame < midFrame) {
    // Scale down first half
    const scaleValue = interpolate(
      frame,
      [startFrame, midFrame],
      [1, targetScale],
      { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
    return { scale: scaleValue, opacity: 1 };
  } else {
    // Scale up second half
    const scaleValue = interpolate(
      frame,
      [midFrame, startFrame + duration],
      [targetScale, 1],
      { easing: EASING.spring, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
    const opacityValue = interpolate(
      frame,
      [midFrame, midFrame + 3],
      [0, 1],
      { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
    );
    return { scale: scaleValue, opacity: opacityValue };
  }
};

// Micro-scene helper - determines which micro-scene is active
export const getMicroScene = (
  frame: number,
  scenes: Array<{ start: number; duration: number }>
): number => {
  for (let i = scenes.length - 1; i >= 0; i--) {
    if (frame >= scenes[i].start) {
      return i;
    }
  }
  return 0;
};

// Get progress within a micro-scene
export const getMicroSceneProgress = (
  frame: number,
  scenes: Array<{ start: number; duration: number }>,
  sceneIndex: number
): number => {
  const scene = scenes[sceneIndex];
  if (!scene) return 0;
  return Math.min(1, Math.max(0, (frame - scene.start) / scene.duration));
};
