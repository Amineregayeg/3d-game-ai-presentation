// Timing utilities for the Architecture Video
// COMPRESSED TIMELINE - target ~3.5 minutes with dense motion
// All times in frames at 30 FPS

export const FPS = 30;

// Convert seconds to frames
export const secondsToFrames = (seconds: number): number => Math.round(seconds * FPS);

// Convert frames to seconds
export const framesToSeconds = (frames: number): number => frames / FPS;

// Transition duration (15 frames = 0.5s)
export const TRANSITION_FRAMES = 15;

// Scene timing configuration - COMPRESSED
export const SCENES = {
  logo: {
    start: 0,
    duration: secondsToFrames(5),  // 150 frames (was 8s)
    transition: TRANSITION_FRAMES,
  },
  dropdown: {
    start: secondsToFrames(5),
    duration: secondsToFrames(10), // 300 frames (was 17s)
    transition: TRANSITION_FRAMES,
  },
  voxformer: {
    start: secondsToFrames(15),
    duration: secondsToFrames(45), // 1350 frames (was 155s)
    transition: TRANSITION_FRAMES,
  },
  rag: {
    start: secondsToFrames(60),
    duration: secondsToFrames(40), // 1200 frames (was 135s)
    transition: TRANSITION_FRAMES,
  },
  avatar: {
    start: secondsToFrames(100),
    duration: secondsToFrames(35), // 1050 frames (was 135s)
    transition: TRANSITION_FRAMES,
  },
  mcp: {
    start: secondsToFrames(135),
    duration: secondsToFrames(35), // 1050 frames (was 135s)
    transition: TRANSITION_FRAMES,
  },
  integration: {
    start: secondsToFrames(170),
    duration: secondsToFrames(15), // 450 frames (was 60s)
    transition: TRANSITION_FRAMES,
  },
  closing: {
    start: secondsToFrames(185),
    duration: secondsToFrames(8), // 240 frames (was 15s)
    transition: 0, // Final scene, no transition
  },
};

// Total video duration: 193 seconds (~3.2 minutes)
export const TOTAL_DURATION = secondsToFrames(193);

// Standard animation durations (in frames) - SNAPPIER
export const DURATIONS = {
  instant: 1,
  veryFast: secondsToFrames(0.1),   // 3 frames
  fast: secondsToFrames(0.2),        // 6 frames
  medium: secondsToFrames(0.35),     // 10-11 frames
  slow: secondsToFrames(0.5),        // 15 frames
  verySlow: secondsToFrames(0.8),    // 24 frames
  extraSlow: secondsToFrames(1.2),   // 36 frames
};

// Stagger delays (in frames) - TIGHTER
export const STAGGER = {
  fast: secondsToFrames(0.05),       // 1-2 frames
  medium: secondsToFrames(0.08),     // 2-3 frames
  slow: secondsToFrames(0.12),       // 3-4 frames
  verySlow: secondsToFrames(0.2),    // 6 frames
};

// Micro-scene durations for each main scene
export const MICRO_SCENES = {
  voxformer: {
    title: { start: 0, duration: secondsToFrames(4) },
    arch1: { start: secondsToFrames(4), duration: secondsToFrames(6) },
    arch2: { start: secondsToFrames(10), duration: secondsToFrames(6) },
    dsp1: { start: secondsToFrames(16), duration: secondsToFrames(5) },
    dsp2: { start: secondsToFrames(21), duration: secondsToFrames(5) },
    attn1: { start: secondsToFrames(26), duration: secondsToFrames(4) },
    attn2: { start: secondsToFrames(30), duration: secondsToFrames(4) },
    attn3: { start: secondsToFrames(34), duration: secondsToFrames(4) },
    attn4: { start: secondsToFrames(38), duration: secondsToFrames(4) },
    outro: { start: secondsToFrames(42), duration: secondsToFrames(3) },
  },
  rag: {
    title: { start: 0, duration: secondsToFrames(3) },
    pipeline: { start: secondsToFrames(3), duration: secondsToFrames(6) },
    kb: { start: secondsToFrames(9), duration: secondsToFrames(5) },
    vector: { start: secondsToFrames(14), duration: secondsToFrames(5) },
    fusion: { start: secondsToFrames(19), duration: secondsToFrames(5) },
    retrieval: { start: secondsToFrames(24), duration: secondsToFrames(6) },
    rerank: { start: secondsToFrames(30), duration: secondsToFrames(6) },
    outro: { start: secondsToFrames(36), duration: secondsToFrames(4) },
  },
  avatar: {
    title: { start: 0, duration: secondsToFrames(3) },
    pipeline: { start: secondsToFrames(3), duration: secondsToFrames(5) },
    diagram: { start: secondsToFrames(8), duration: secondsToFrames(6) },
    tts1: { start: secondsToFrames(14), duration: secondsToFrames(5) },
    tts2: { start: secondsToFrames(19), duration: secondsToFrames(4) },
    lipsync1: { start: secondsToFrames(23), duration: secondsToFrames(5) },
    lipsync2: { start: secondsToFrames(28), duration: secondsToFrames(4) },
    outro: { start: secondsToFrames(32), duration: secondsToFrames(3) },
  },
  mcp: {
    title: { start: 0, duration: secondsToFrames(3) },
    layers: { start: secondsToFrames(3), duration: secondsToFrames(5) },
    diagram: { start: secondsToFrames(8), duration: secondsToFrames(6) },
    protocol1: { start: secondsToFrames(14), duration: secondsToFrames(5) },
    protocol2: { start: secondsToFrames(19), duration: secondsToFrames(4) },
    tools1: { start: secondsToFrames(23), duration: secondsToFrames(4) },
    tools2: { start: secondsToFrames(27), duration: secondsToFrames(4) },
    outro: { start: secondsToFrames(31), duration: secondsToFrames(4) },
  },
};

// Helper to calculate relative frame within a scene
export const getRelativeFrame = (
  absoluteFrame: number,
  sceneStart: number
): number => {
  return Math.max(0, absoluteFrame - sceneStart);
};

// Helper to check if current frame is within a scene
export const isInScene = (
  frame: number,
  sceneStart: number,
  sceneDuration: number
): boolean => {
  return frame >= sceneStart && frame < sceneStart + sceneDuration;
};

// Calculate progress within a time range (0 to 1)
export const getProgress = (
  frame: number,
  startFrame: number,
  duration: number
): number => {
  if (frame < startFrame) return 0;
  if (frame >= startFrame + duration) return 1;
  return (frame - startFrame) / duration;
};

// Check if in transition zone between scenes
export const isInTransition = (
  frame: number,
  sceneEnd: number,
  transitionDuration: number = TRANSITION_FRAMES
): boolean => {
  return frame >= sceneEnd - transitionDuration && frame < sceneEnd;
};

// Get transition progress (0 to 1)
export const getTransitionProgress = (
  frame: number,
  sceneEnd: number,
  transitionDuration: number = TRANSITION_FRAMES
): number => {
  if (!isInTransition(frame, sceneEnd, transitionDuration)) return 0;
  return (frame - (sceneEnd - transitionDuration)) / transitionDuration;
};
