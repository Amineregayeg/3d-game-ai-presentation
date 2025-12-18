import React from 'react';
import { AbsoluteFill, interpolate } from 'remotion';
import { COLORS, THEMES, GRADIENTS } from '../../../utils/colors';
import { fadeIn, entranceAnimation, countUp, drawPath, cameraZoom, EASING, highlightDim } from '../../../utils/animations';
import { secondsToFrames, STAGGER, SCENES, TRANSITION_FRAMES, MICRO_SCENES } from '../../../utils/timings';
import { GradientText } from '../components';
import { Cursor, createCursorPath } from '../../../components/Cursor';

interface S2_VoxFormerProps {
  frame: number;
}

const METRICS = [
  { label: 'Total Params', value: 142, unit: 'M' },
  { label: 'Trainable', value: 47, unit: 'M' },
  { label: 'WER Target', value: 8, unit: '%', prefix: '<' },
];

const DSP_STAGES = [
  { num: '01', title: 'Signal Conditioning', color: COLORS.cyan500, items: ['DC Offset', 'Pre-Emphasis', 'Sample Rate'] },
  { num: '02', title: 'Voice Activity', color: COLORS.emerald500, items: ['Energy VAD', 'Spectral Entropy', 'Neural VAD'] },
  { num: '03', title: 'Noise Estimation', color: COLORS.purple500, items: ['MCRA Algorithm', 'Adaptive Track', 'SNR Est.'] },
  { num: '04', title: 'Noise Reduction', color: COLORS.rose500, items: ['Spectral Sub', 'Wiener Filter', 'MMSE-STSA'] },
  { num: '05', title: 'Echo Cancel', color: COLORS.amber500, items: ['Adaptive Filter', 'NLMS/RLS', 'Double-Talk'] },
  { num: '06', title: 'Voice Isolation', color: COLORS.orange500, items: ['Deep Attractor', 'Source Sep', 'Mask Est.'] },
];

export const S2_VoxFormer: React.FC<S2_VoxFormerProps> = ({ frame }) => {
  const theme = THEMES.voxformer;
  const MS = MICRO_SCENES.voxformer;

  // Determine current micro-scene
  const getMicroScene = () => {
    if (frame < MS.arch1.start) return 'title';
    if (frame < MS.arch2.start) return 'arch1';
    if (frame < MS.dsp1.start) return 'arch2';
    if (frame < MS.dsp2.start) return 'dsp1';
    if (frame < MS.attn1.start) return 'attn1';
    if (frame < MS.attn2.start) return 'attn2';
    if (frame < MS.attn3.start) return 'attn3';
    if (frame < MS.attn4.start) return 'attn4';
    return 'outro';
  };

  const currentMicro = getMicroScene();

  // Scene exit transition
  const exitStart = SCENES.voxformer.duration - TRANSITION_FRAMES;
  const exitProgress = interpolate(
    frame,
    [exitStart, SCENES.voxformer.duration],
    [0, 1],
    { easing: EASING.entrance, extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  const exitScale = interpolate(exitProgress, [0, 1], [1, 1.2]);
  const exitOpacity = interpolate(exitProgress, [0, 0.6, 1], [1, 1, 0]);

  // Title Section (0-4s)
  const renderTitle = () => {
    const badgeAnim = entranceAnimation(frame, 0, secondsToFrames(0.3), { fadeIn: true, scaleFrom: 0.8 });
    const titleAnim = entranceAnimation(frame, secondsToFrames(0.25), secondsToFrames(0.3), { fadeIn: true, slideFromY: 15 });
    const subtitleAnim = entranceAnimation(frame, secondsToFrames(0.5), secondsToFrames(0.25), { fadeIn: true });

    return (
      <AbsoluteFill style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20 }}>
        <div style={{ padding: '10px 20px', backgroundColor: `${theme.primary}20`, border: `2px solid ${theme.primary}`, borderRadius: 50, opacity: badgeAnim.opacity, transform: badgeAnim.transform }}>
          <span style={{ fontSize: 16, fontWeight: 600, color: theme.primary, fontFamily: 'system-ui', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Component 1</span>
        </div>
        <div style={{ opacity: titleAnim.opacity, transform: titleAnim.transform }}>
          <GradientText gradient={GRADIENTS.voxformer} fontSize={72} fontWeight={800}>VoxFormer STT</GradientText>
        </div>
        <div style={{ opacity: subtitleAnim.opacity }}>
          <span style={{ fontSize: 24, color: COLORS.slate600, fontFamily: 'system-ui' }}>Custom Speech-to-Text Transformer Architecture</span>
        </div>
        <div style={{ display: 'flex', gap: 35, marginTop: 30, opacity: fadeIn(frame, secondsToFrames(1), secondsToFrames(0.3)) }}>
          {METRICS.map((metric, i) => (
            <div key={metric.label} style={{ textAlign: 'center', opacity: fadeIn(frame, secondsToFrames(1) + i * STAGGER.medium, secondsToFrames(0.2)) }}>
              <div style={{ fontSize: 42, fontWeight: 700, color: theme.primary, fontFamily: 'system-ui' }}>
                {metric.prefix || ''}{countUp(frame, secondsToFrames(1.2), secondsToFrames(0.6), 0, metric.value)}{metric.unit}
              </div>
              <div style={{ fontSize: 12, color: COLORS.slate600, marginTop: 3, fontFamily: 'system-ui', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{metric.label}</div>
            </div>
          ))}
        </div>
      </AbsoluteFill>
    );
  };

  // Architecture Section with guided tour (4-16s split into arch1 and arch2)
  const renderArchitecture = () => {
    const localFrame = frame - MS.arch1.start;
    const isArch2 = frame >= MS.arch2.start;
    const arch2Local = isArch2 ? frame - MS.arch2.start : 0;

    // Active block tracking for cursor tour
    const activeBlock = isArch2
      ? Math.min(4, Math.floor(arch2Local / secondsToFrames(1.2)))
      : Math.min(2, Math.floor(localFrame / secondsToFrames(2)));

    // Camera zoom to active area
    const zoomTargets = [
      { x: 15, y: 50 }, // Audio input
      { x: 30, y: 50 }, // WavLM
      { x: 50, y: 50 }, // Adapter + Zipformer
      { x: 75, y: 50 }, // Decoder
      { x: 90, y: 50 }, // Output
    ];
    const currentZoom = cameraZoom(localFrame % secondsToFrames(2.5), 0, secondsToFrames(2.5), 1.08, zoomTargets[activeBlock]?.x || 50, zoomTargets[activeBlock]?.y || 50);

    // Cursor path through architecture
    const cursorKeyframes = createCursorPath(secondsToFrames(0.5), [
      { x: 100, y: 300, delay: 0, action: 'hover' },
      { x: 220, y: 280, delay: 40, action: 'click' },
      { x: 380, y: 280, delay: 35, action: 'click' },
      { x: 560, y: 260, delay: 35, action: 'click' },
      { x: 760, y: 280, delay: 35, action: 'click' },
      { x: 920, y: 280, delay: 35, action: 'click' },
    ]);

    const headerAnim = entranceAnimation(localFrame, 0, secondsToFrames(0.3), { fadeIn: true, slideFromY: -15 });

    return (
      <AbsoluteFill style={{ padding: 50, transform: `scale(${currentZoom.scale})`, transformOrigin: currentZoom.transformOrigin }}>
        <div style={{ opacity: headerAnim.opacity, transform: headerAnim.transform, marginBottom: 20 }}>
          <GradientText gradient={GRADIENTS.voxformer} fontSize={42} fontWeight={700}>System Architecture</GradientText>
          <p style={{ fontSize: 18, color: COLORS.slate600, marginTop: 6, fontFamily: 'system-ui' }}>WavLM backbone + Zipformer encoder + Transformer decoder</p>
        </div>

        <svg viewBox="0 0 950 350" style={{ width: '100%', height: 300 }}>
          <defs>
            <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor={COLORS.cyan500} stopOpacity="0.8"/><stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/></linearGradient>
            <linearGradient id="gradPurple" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor={COLORS.purple500} stopOpacity="0.8"/><stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/></linearGradient>
            <linearGradient id="gradEmerald" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor={COLORS.emerald500} stopOpacity="0.8"/><stop offset="100%" stopColor="#059669" stopOpacity="0.8"/></linearGradient>
            <linearGradient id="gradRose" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor={COLORS.rose500} stopOpacity="0.8"/><stop offset="100%" stopColor="#e11d48" stopOpacity="0.8"/></linearGradient>
            <filter id="glow"><feGaussianBlur stdDeviation="3" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill={COLORS.slate600}/></marker>
          </defs>

          {/* Audio Input - highlight when active */}
          <g transform="translate(20, 110)" style={{ opacity: highlightDim(localFrame, 0, secondsToFrames(0.3), activeBlock === 0) }}>
            <rect width="70" height="90" rx="8" fill={COLORS.slate800} stroke={activeBlock === 0 ? COLORS.cyan500 : COLORS.slate700} strokeWidth={activeBlock === 0 ? 3 : 2}/>
            <text x="35" y="35" textAnchor="middle" fill={COLORS.slate500} fontSize="10">Raw</text>
            <text x="35" y="50" textAnchor="middle" fill={COLORS.slate500} fontSize="10">Audio</text>
            <path d="M15 65 Q25 55 35 65 Q45 75 55 65" stroke={COLORS.cyan500} fill="none" strokeWidth="2"/>
          </g>

          <path d="M95 155 L125 155" stroke={COLORS.slate600} strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="20" strokeDashoffset={20 - drawPath(localFrame, secondsToFrames(0.3), secondsToFrames(0.2)) * 20}/>

          {/* WavLM */}
          <g transform="translate(130, 70)" style={{ opacity: highlightDim(localFrame, secondsToFrames(0.5), secondsToFrames(0.3), activeBlock === 1) }}>
            <rect width="130" height="160" rx="8" fill="url(#gradPurple)" filter={activeBlock === 1 ? "url(#glow)" : "none"}/>
            <text x="65" y="22" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">WavLM-Base</text>
            <text x="65" y="38" textAnchor="middle" fill="#e9d5ff" fontSize="8">95M params (frozen)</text>
            <rect x="8" y="48" width="114" height="22" rx="4" fill="#0f172a" opacity="0.5"/><text x="65" y="63" textAnchor="middle" fill="#e9d5ff" fontSize="8">12 Transformer Layers</text>
            <rect x="8" y="75" width="114" height="22" rx="4" fill="#0f172a" opacity="0.5"/><text x="65" y="90" textAnchor="middle" fill="#e9d5ff" fontSize="8">Weighted Layer Sum</text>
            <rect x="8" y="102" width="114" height="22" rx="4" fill="#0f172a" opacity="0.5"/><text x="65" y="117" textAnchor="middle" fill="#e9d5ff" fontSize="8">768-dim @ 50fps</text>
            <text x="65" y="145" textAnchor="middle" fill="#c4b5fd" fontSize="7">Pretrained on 94K hours</text>
          </g>

          <path d="M265 155 L295 155" stroke={COLORS.slate600} strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="20" strokeDashoffset={20 - drawPath(localFrame, secondsToFrames(1), secondsToFrames(0.2)) * 20}/>

          {/* Adapter + Zipformer */}
          <g transform="translate(300, 50)" style={{ opacity: highlightDim(localFrame, secondsToFrames(1.2), secondsToFrames(0.3), activeBlock === 2) }}>
            <rect width="200" height="200" rx="10" fill={COLORS.slate800} stroke={activeBlock === 2 ? COLORS.cyan500 : COLORS.slate700} strokeWidth={activeBlock === 2 ? 3 : 2}/>
            <text x="100" y="22" textAnchor="middle" fill={COLORS.cyan500} fontSize="11" fontWeight="bold">ZIPFORMER ENCODER</text>
            <text x="100" y="38" textAnchor="middle" fill={COLORS.slate600} fontSize="8">6 Blocks | 25M params</text>
            <g transform="translate(15, 48)">
              <rect width="170" height="140" rx="6" fill="url(#gradCyan)" filter={activeBlock === 2 ? "url(#glow)" : "none"}/>
              <rect x="8" y="10" width="154" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="85" y="25" textAnchor="middle" fill="#a5f3fc" fontSize="8">Multi-Head Attn (8)</text>
              <rect x="8" y="36" width="154" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="85" y="51" textAnchor="middle" fill="#a5f3fc" fontSize="8">Depthwise Conv (k=31)</text>
              <rect x="8" y="62" width="154" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="85" y="77" textAnchor="middle" fill="#a5f3fc" fontSize="8">FFN + LayerNorm</text>
              <rect x="8" y="88" width="154" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="85" y="103" textAnchor="middle" fill="#a5f3fc" fontSize="8">SwiGLU Activation</text>
              <text x="85" y="128" textAnchor="middle" fill="#67e8f9" fontSize="7">U-Net: 50→25→12.5 fps</text>
            </g>
          </g>

          <path d="M505 155 L535 155" stroke={COLORS.slate600} strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="20" strokeDashoffset={20 - drawPath(localFrame, secondsToFrames(2), secondsToFrames(0.2)) * 20}/>

          {/* Decoder */}
          <g transform="translate(540, 50)" style={{ opacity: highlightDim(localFrame, secondsToFrames(2.2), secondsToFrames(0.3), activeBlock === 3) }}>
            <rect width="160" height="200" rx="10" fill={COLORS.slate800} stroke={activeBlock === 3 ? COLORS.rose500 : COLORS.slate700} strokeWidth={activeBlock === 3 ? 3 : 2}/>
            <text x="80" y="22" textAnchor="middle" fill={COLORS.rose500} fontSize="11" fontWeight="bold">TRANSFORMER DECODER</text>
            <text x="80" y="38" textAnchor="middle" fill={COLORS.slate600} fontSize="8">4 Layers | 20M params</text>
            <g transform="translate(12, 48)">
              <rect width="136" height="130" rx="6" fill="url(#gradRose)" filter={activeBlock === 3 ? "url(#glow)" : "none"}/>
              <rect x="8" y="10" width="120" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="68" y="25" textAnchor="middle" fill="#fce7f3" fontSize="7">Masked Self-Attention</text>
              <rect x="8" y="36" width="120" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="68" y="51" textAnchor="middle" fill="#fce7f3" fontSize="7">Cross-Attention</text>
              <rect x="8" y="62" width="120" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="68" y="77" textAnchor="middle" fill="#fce7f3" fontSize="7">Feed-Forward (2048)</text>
              <rect x="8" y="88" width="120" height="22" rx="3" fill="#0f172a" opacity="0.5"/><text x="68" y="103" textAnchor="middle" fill="#fce7f3" fontSize="7">KV-Cache</text>
            </g>
            <text x="80" y="190" textAnchor="middle" fill="#f9a8d4" fontSize="7">BPE vocab: 2000 tokens</text>
          </g>

          <path d="M705 155 L735 155" stroke={COLORS.slate600} strokeWidth="2" markerEnd="url(#arrowhead)" strokeDasharray="20" strokeDashoffset={20 - drawPath(localFrame, secondsToFrames(3), secondsToFrames(0.2)) * 20}/>

          {/* Output */}
          <g transform="translate(740, 95)" style={{ opacity: highlightDim(localFrame, secondsToFrames(3.2), secondsToFrames(0.3), activeBlock === 4) }}>
            <rect width="80" height="110" rx="8" fill={COLORS.slate800} stroke={activeBlock === 4 ? COLORS.emerald500 : COLORS.slate700} strokeWidth={activeBlock === 4 ? 3 : 2}/>
            <text x="40" y="25" textAnchor="middle" fill={COLORS.emerald500} fontSize="10" fontWeight="bold">Output</text>
            <text x="40" y="45" textAnchor="middle" fill={COLORS.slate500} fontSize="8">CE Loss</text>
            <text x="40" y="58" textAnchor="middle" fill={COLORS.slate500} fontSize="7">0.7 weight</text>
            <text x="40" y="82" textAnchor="middle" fill={COLORS.emerald500} fontSize="22">T</text>
            <text x="40" y="100" textAnchor="middle" fill="#6ee7b7" fontSize="7">Text Output</text>
          </g>
        </svg>

        {/* Model Specs - animate in during arch2 */}
        {isArch2 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: 14, marginTop: 15, opacity: fadeIn(arch2Local, secondsToFrames(2), secondsToFrames(0.3)) }}>
            {[
              { name: 'WavLM', params: '95M', desc: 'Frozen', color: COLORS.purple500 },
              { name: 'Adapter', params: '2M', desc: '768→512', color: COLORS.emerald500 },
              { name: 'Zipformer', params: '25M', desc: '6 blocks', color: COLORS.cyan500 },
              { name: 'Decoder', params: '20M', desc: '4 layers', color: COLORS.rose500 },
              { name: 'Total', params: '142M', desc: '47M train', color: COLORS.amber500 },
            ].map((comp, i) => (
              <div key={comp.name} style={{ backgroundColor: `${COLORS.slate800}80`, border: `1px solid ${COLORS.slate700}80`, borderRadius: 6, padding: '10px 14px', textAlign: 'center', opacity: fadeIn(arch2Local, secondsToFrames(2) + i * STAGGER.fast, secondsToFrames(0.2)) }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: comp.color }}>{comp.name}</div>
                <div style={{ fontSize: 11, color: COLORS.white }}>{comp.params}</div>
                <div style={{ fontSize: 9, color: COLORS.slate500 }}>{comp.desc}</div>
              </div>
            ))}
          </div>
        )}

        <Cursor frame={localFrame} keyframes={cursorKeyframes} color={COLORS.white} size={20} />
      </AbsoluteFill>
    );
  };

  // DSP Pipeline with staggered reveal (16-26s)
  const renderDSP = () => {
    const localFrame = frame - MS.dsp1.start;
    const isDSP2 = frame >= MS.dsp2.start;
    const dsp2Local = isDSP2 ? frame - MS.dsp2.start : 0;
    const activeStage = isDSP2 ? Math.min(5, 3 + Math.floor(dsp2Local / secondsToFrames(1.2))) : Math.min(2, Math.floor(localFrame / secondsToFrames(1.5)));

    const cursorKeyframes = createCursorPath(secondsToFrames(0.3), [
      { x: 200, y: 300, delay: 0 },
      { x: 180, y: 350, delay: 30, action: 'click' },
      { x: 340, y: 350, delay: 35, action: 'click' },
      { x: 500, y: 350, delay: 35, action: 'click' },
      { x: 660, y: 350, delay: 35, action: 'click' },
      { x: 820, y: 350, delay: 35, action: 'click' },
      { x: 980, y: 350, delay: 35, action: 'click' },
    ]);

    const headerAnim = entranceAnimation(localFrame, 0, secondsToFrames(0.3), { fadeIn: true });

    return (
      <AbsoluteFill style={{ padding: 50 }}>
        <div style={{ opacity: headerAnim.opacity, marginBottom: 25 }}>
          <GradientText gradient={GRADIENTS.voxformer} fontSize={42} fontWeight={700}>Voice Isolation Pipeline</GradientText>
          <p style={{ fontSize: 18, color: COLORS.slate600, marginTop: 6, fontFamily: 'system-ui' }}>6-stage DSP pipeline for robust voice extraction</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 14, marginTop: 30 }}>
          {DSP_STAGES.map((stage, idx) => {
            const isActive = idx === activeStage;
            const stageOpacity = highlightDim(localFrame, secondsToFrames(0.3) + idx * STAGGER.slow, secondsToFrames(0.3), isActive);
            return (
              <div key={stage.num} style={{
                padding: 14,
                borderRadius: 10,
                backgroundColor: isActive ? `${stage.color}30` : `${stage.color}15`,
                border: `2px solid ${isActive ? stage.color : `${stage.color}50`}`,
                opacity: stageOpacity,
                transform: `scale(${isActive ? 1.03 : 1})`,
                boxShadow: isActive ? `0 0 20px ${stage.color}40` : 'none',
              }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: stage.color, opacity: 0.4, marginBottom: 3 }}>{stage.num}</div>
                <h3 style={{ fontSize: 13, fontWeight: 600, color: COLORS.white, marginBottom: 10 }}>{stage.title}</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {stage.items.map((item, itemIdx) => (
                    <div key={item} style={{
                      fontSize: 10,
                      color: isActive ? COLORS.white : COLORS.slate400,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 5,
                      opacity: fadeIn(localFrame, secondsToFrames(0.5) + idx * STAGGER.slow + itemIdx * STAGGER.fast, secondsToFrames(0.15)),
                    }}>
                      <div style={{ width: 4, height: 4, borderRadius: '50%', backgroundColor: stage.color }} />
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Bottom Metrics */}
        <div style={{ marginTop: 30, padding: 18, backgroundColor: `${COLORS.slate800}30`, borderRadius: 10, border: `1px solid ${COLORS.slate700}50`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', opacity: fadeIn(localFrame, secondsToFrames(2), secondsToFrames(0.3)) }}>
          <div>
            <span style={{ fontSize: 11, color: COLORS.slate500, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Core DSP</span>
            <div style={{ fontSize: 13, color: COLORS.slate300 }}>Custom FFT (Cooley-Tukey) | FIR/IIR Filters | Adaptive NLMS</div>
          </div>
          <div style={{ display: 'flex', gap: 35 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: COLORS.cyan500 }}>&lt;10ms</div>
              <div style={{ fontSize: 10, color: COLORS.slate500 }}>Latency</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: COLORS.purple500 }}>&gt;-20dB</div>
              <div style={{ fontSize: 10, color: COLORS.slate500 }}>Noise Reduction</div>
            </div>
          </div>
        </div>

        <Cursor frame={localFrame} keyframes={cursorKeyframes} color={COLORS.white} size={20} />
      </AbsoluteFill>
    );
  };

  // Attention Section split into micro-scenes (26-45s)
  const renderAttention = () => {
    const localFrame = frame - MS.attn1.start;
    const attnPhase = frame < MS.attn2.start ? 0 : frame < MS.attn3.start ? 1 : frame < MS.attn4.start ? 2 : 3;

    const cursorKeyframes = createCursorPath(secondsToFrames(0.3), [
      { x: 400, y: 300, delay: 0 },
      { x: 350, y: 380, delay: 25, action: 'click' },
      { x: 700, y: 350, delay: 60, action: 'click' },
      { x: 700, y: 480, delay: 60, action: 'click' },
    ]);

    const headerAnim = entranceAnimation(localFrame, 0, secondsToFrames(0.3), { fadeIn: true });

    return (
      <AbsoluteFill style={{ padding: 50 }}>
        <div style={{ opacity: headerAnim.opacity, marginBottom: 25 }}>
          <GradientText gradient={GRADIENTS.voxformer} fontSize={42} fontWeight={700}>Multi-Head Attention + RoPE</GradientText>
          <p style={{ fontSize: 18, color: COLORS.slate600, marginTop: 6, fontFamily: 'system-ui' }}>Custom attention with Rotary Position Embeddings</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 25, marginTop: 15 }}>
          {/* Attention Diagram */}
          <div style={{ backgroundColor: `${COLORS.slate800}30`, borderRadius: 10, padding: 20, border: `1px solid ${attnPhase === 0 ? COLORS.purple500 : COLORS.slate700}50`, opacity: highlightDim(localFrame, 0, secondsToFrames(0.3), attnPhase === 0) }}>
            <h3 style={{ fontSize: 16, fontWeight: 600, color: COLORS.white, marginBottom: 14 }}>Scaled Dot-Product Attention</h3>
            <svg viewBox="0 0 400 240" style={{ width: '100%', height: 200 }}>
              <defs>
                <linearGradient id="qGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#f472b6"/><stop offset="100%" stopColor="#ec4899"/></linearGradient>
                <linearGradient id="kGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#a78bfa"/><stop offset="100%" stopColor="#8b5cf6"/></linearGradient>
                <linearGradient id="vGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stopColor="#34d399"/><stop offset="100%" stopColor="#10b981"/></linearGradient>
              </defs>
              <rect x="170" y="5" width="60" height="28" rx="4" fill={COLORS.slate700}/><text x="200" y="24" textAnchor="middle" fill={COLORS.slate400} fontSize="11">X</text>
              <rect x="50" y="55" width="55" height="28" rx="4" fill="url(#qGrad)"/><text x="78" y="74" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Q</text>
              <rect x="170" y="55" width="55" height="28" rx="4" fill="url(#kGrad)"/><text x="198" y="74" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">K</text>
              <rect x="290" y="55" width="55" height="28" rx="4" fill="url(#vGrad)"/><text x="318" y="74" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">V</text>
              <path d="M200 33 L78 55" stroke={COLORS.slate600} strokeWidth="1.5"/>
              <path d="M200 33 L198 55" stroke={COLORS.slate600} strokeWidth="1.5"/>
              <path d="M200 33 L318 55" stroke={COLORS.slate600} strokeWidth="1.5"/>
              <circle cx="140" cy="115" r="18" fill={COLORS.slate800} stroke="#8b5cf6" strokeWidth="2"/><text x="140" y="119" textAnchor="middle" fill="#c4b5fd" fontSize="9">QK^T</text>
              <rect x="110" y="145" width="60" height="22" rx="4" fill={COLORS.slate700}/><text x="140" y="160" textAnchor="middle" fill={COLORS.slate400} fontSize="8">/sqrt(d_k)</text>
              <rect x="110" y="175" width="60" height="22" rx="4" fill="#7c3aed"/><text x="140" y="190" textAnchor="middle" fill="white" fontSize="8">Softmax</text>
              <circle cx="220" cy="188" r="18" fill={COLORS.slate800} stroke="#10b981" strokeWidth="2"/><text x="220" y="192" textAnchor="middle" fill="#6ee7b7" fontSize="9">@ V</text>
              <rect x="260" y="210" width="70" height="22" rx="4" fill="#0f766e"/><text x="295" y="225" textAnchor="middle" fill="white" fontSize="10">Output</text>
            </svg>
          </div>

          {/* Multi-Head Config */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div style={{ backgroundColor: `${COLORS.slate800}30`, borderRadius: 10, padding: 18, border: `1px solid ${attnPhase === 1 ? COLORS.cyan500 : COLORS.slate700}50`, opacity: highlightDim(localFrame, secondsToFrames(4), secondsToFrames(0.3), attnPhase === 1) }}>
              <h3 style={{ fontSize: 13, color: COLORS.slate400, marginBottom: 10 }}>Mathematical Formulation</h3>
              <div style={{ backgroundColor: `${COLORS.slate900}80`, padding: 14, borderRadius: 6, fontFamily: 'monospace', fontSize: 13 }}>
                <div style={{ color: COLORS.purple500 }}>Attention(Q, K, V) =</div>
                <div style={{ color: COLORS.cyan500, marginLeft: 14 }}>softmax(QK^T / sqrt(d_k)) V</div>
              </div>
            </div>
            <div style={{ backgroundColor: `${COLORS.slate800}30`, borderRadius: 10, padding: 18, border: `1px solid ${attnPhase >= 2 ? COLORS.emerald500 : COLORS.slate700}50`, opacity: highlightDim(localFrame, secondsToFrames(8), secondsToFrames(0.3), attnPhase >= 2) }}>
              <h3 style={{ fontSize: 13, color: COLORS.slate400, marginBottom: 10 }}>Multi-Head Configuration</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
                {[
                  { value: '8', label: 'Attention Heads', color: COLORS.purple500 },
                  { value: '64', label: 'Head Dimension', color: COLORS.cyan500 },
                  { value: '512', label: 'Model Dimension', color: COLORS.emerald500 },
                  { value: '0', label: 'Bias Terms', color: COLORS.rose500 },
                ].map((item, i) => (
                  <div key={item.label} style={{ textAlign: 'center', padding: 10, backgroundColor: `${COLORS.slate900}50`, borderRadius: 6, opacity: fadeIn(localFrame, secondsToFrames(8) + i * STAGGER.medium, secondsToFrames(0.2)) }}>
                    <div style={{ fontSize: 24, fontWeight: 700, color: item.color }}>{item.value}</div>
                    <div style={{ fontSize: 10, color: COLORS.slate500 }}>{item.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <Cursor frame={localFrame} keyframes={cursorKeyframes} color={COLORS.white} size={20} />
      </AbsoluteFill>
    );
  };

  // Render appropriate micro-scene
  const getContent = () => {
    if (currentMicro === 'title') return renderTitle();
    if (currentMicro === 'arch1' || currentMicro === 'arch2') return renderArchitecture();
    if (currentMicro === 'dsp1' || currentMicro === 'dsp2') return renderDSP();
    return renderAttention();
  };

  return (
    <AbsoluteFill style={{ transform: `scale(${exitScale})`, opacity: exitOpacity }}>
      {getContent()}
    </AbsoluteFill>
  );
};
