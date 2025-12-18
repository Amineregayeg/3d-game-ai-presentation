import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from 'remotion';
import { COLORS, GRADIENTS } from '../../utils/colors';
import { SCENES, getRelativeFrame, TRANSITION_FRAMES } from '../../utils/timings';

// Scene imports
import { S0_LogoIntro } from './scenes/S0_LogoIntro';
import { S1_ProjectDropdown } from './scenes/S1_ProjectDropdown';
import { S2_VoxFormer } from './scenes/S2_VoxFormer';
import { S3_RAGSystem } from './scenes/S3_RAGSystem';
import { S4_Avatar } from './scenes/S4_Avatar';
import { S5_MCP } from './scenes/S5_MCP';
import { S6_Integration } from './scenes/S6_Integration';
import { S7_Closing } from './scenes/S7_Closing';

interface ArchitectureVideoProps {
  previewScene?: string;
}

// Scene render helper - determines if scene should be visible with transition overlap
const shouldRenderScene = (
  frame: number,
  sceneStart: number,
  sceneDuration: number,
  transitionFrames: number = TRANSITION_FRAMES
): boolean => {
  // Scene renders from start until end + transition buffer for overlap
  return frame >= sceneStart && frame < sceneStart + sceneDuration + transitionFrames;
};

// Get scene entrance opacity (fade in during first few frames)
const getEntranceOpacity = (
  relativeFrame: number,
  transitionFrames: number = TRANSITION_FRAMES
): number => {
  // Handle zero transition case
  if (transitionFrames <= 0) return 1;

  return interpolate(relativeFrame, [0, transitionFrames], [0, 1], {
    easing: Easing.out(Easing.cubic),
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

// Scene definitions with z-index ordering
const SCENE_CONFIG = [
  { key: 'logo', Component: S0_LogoIntro, zIndex: 1 },
  { key: 'dropdown', Component: S1_ProjectDropdown, zIndex: 2 },
  { key: 'voxformer', Component: S2_VoxFormer, zIndex: 3 },
  { key: 'rag', Component: S3_RAGSystem, zIndex: 4 },
  { key: 'avatar', Component: S4_Avatar, zIndex: 5 },
  { key: 'mcp', Component: S5_MCP, zIndex: 6 },
  { key: 'integration', Component: S6_Integration, zIndex: 7 },
  { key: 'closing', Component: S7_Closing, zIndex: 8 },
] as const;

export const ArchitectureVideo: React.FC<ArchitectureVideoProps> = ({ previewScene }) => {
  const frame = useCurrentFrame();

  // Background component
  const Background = () => (
    <AbsoluteFill
      style={{
        background: GRADIENTS.background,
      }}
    >
      {/* Subtle grid pattern */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `
            linear-gradient(${COLORS.slate800}20 1px, transparent 1px),
            linear-gradient(90deg, ${COLORS.slate800}20 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          opacity: 0.5,
        }}
      />
      {/* Gradient orbs */}
      <div
        style={{
          position: 'absolute',
          top: '-20%',
          left: '-10%',
          width: '50%',
          height: '50%',
          background: `radial-gradient(circle, ${COLORS.cyanAlpha(0.15)} 0%, transparent 70%)`,
          filter: 'blur(60px)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: '-20%',
          right: '-10%',
          width: '50%',
          height: '50%',
          background: `radial-gradient(circle, ${COLORS.purpleAlpha ? COLORS.purpleAlpha(0.15) : 'rgba(168, 85, 247, 0.15)'} 0%, transparent 70%)`,
          filter: 'blur(60px)',
        }}
      />
    </AbsoluteFill>
  );

  // If previewing a specific scene, only render that scene
  if (previewScene) {
    return (
      <AbsoluteFill>
        <Background />
        {previewScene === 'logo' && <S0_LogoIntro frame={frame} />}
        {previewScene === 'dropdown' && <S1_ProjectDropdown frame={frame} />}
        {previewScene === 'voxformer' && <S2_VoxFormer frame={frame} />}
        {previewScene === 'rag' && <S3_RAGSystem frame={frame} />}
        {previewScene === 'avatar' && <S4_Avatar frame={frame} />}
        {previewScene === 'mcp' && <S5_MCP frame={frame} />}
        {previewScene === 'integration' && <S6_Integration frame={frame} />}
        {previewScene === 'closing' && <S7_Closing frame={frame} />}
      </AbsoluteFill>
    );
  }

  return (
    <AbsoluteFill>
      <Background />

      {/* Render scenes with transition overlap and z-indexing */}
      {SCENE_CONFIG.map(({ key, Component, zIndex }) => {
        const scene = SCENES[key as keyof typeof SCENES];
        const relativeFrame = getRelativeFrame(frame, scene.start);
        const transitionFrames = scene.transition || 0;

        if (!shouldRenderScene(frame, scene.start, scene.duration, transitionFrames)) {
          return null;
        }

        // Calculate entrance opacity for smooth transition-in
        // (Each scene handles its own exit animation internally)
        const entranceOpacity = key === 'logo'
          ? 1  // First scene doesn't fade in
          : getEntranceOpacity(relativeFrame, transitionFrames);

        return (
          <AbsoluteFill
            key={key}
            style={{
              zIndex,
              opacity: entranceOpacity,
            }}
          >
            <Component frame={relativeFrame} />
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};
