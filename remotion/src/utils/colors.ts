// Color palette for the Architecture Video
// Matches the existing slide presentation themes

export const COLORS = {
  // Background colors
  slate950: '#020617',
  slate900: '#0f172a',
  slate800: '#1e293b',
  slate700: '#334155',
  slate600: '#475569',

  // VoxFormer theme (Cyan/Purple)
  cyan400: '#22d3ee',
  cyan500: '#06b6d4',
  cyan600: '#0891b2',
  purple400: '#c084fc',
  purple500: '#a855f7',
  purple600: '#9333ea',

  // RAG theme (Emerald/Cyan)
  emerald400: '#34d399',
  emerald500: '#10b981',
  emerald600: '#059669',

  // Avatar theme (Rose/Pink)
  rose400: '#fb7185',
  rose500: '#f43f5e',
  rose600: '#e11d48',
  pink400: '#f472b6',
  pink500: '#ec4899',
  fuchsia500: '#d946ef',

  // MCP theme (Orange/Amber)
  orange400: '#fb923c',
  orange500: '#f97316',
  orange600: '#ea580c',
  amber400: '#fbbf24',
  amber500: '#f59e0b',
  yellow500: '#eab308',

  // Accent colors
  white: '#ffffff',
  black: '#000000',
  green500: '#22c55e',  // Success/Checkmarks
  red500: '#ef4444',    // Errors
  blue500: '#3b82f6',   // Links/Info

  // Transparent variants
  whiteAlpha: (alpha: number) => `rgba(255, 255, 255, ${alpha})`,
  blackAlpha: (alpha: number) => `rgba(0, 0, 0, ${alpha})`,
  cyanAlpha: (alpha: number) => `rgba(6, 182, 212, ${alpha})`,
  emeraldAlpha: (alpha: number) => `rgba(16, 185, 129, ${alpha})`,
  roseAlpha: (alpha: number) => `rgba(244, 63, 94, ${alpha})`,
  orangeAlpha: (alpha: number) => `rgba(249, 115, 22, ${alpha})`,
};

// Component-specific color schemes
export const THEMES = {
  voxformer: {
    primary: COLORS.cyan500,
    secondary: COLORS.purple500,
    accent: COLORS.cyan400,
    glow: COLORS.cyanAlpha(0.3),
  },
  rag: {
    primary: COLORS.emerald500,
    secondary: COLORS.cyan500,
    accent: COLORS.emerald400,
    glow: COLORS.emeraldAlpha(0.3),
  },
  avatar: {
    primary: COLORS.rose500,
    secondary: COLORS.pink500,
    accent: COLORS.rose400,
    glow: COLORS.roseAlpha(0.3),
  },
  mcp: {
    primary: COLORS.orange500,
    secondary: COLORS.amber500,
    accent: COLORS.orange400,
    glow: COLORS.orangeAlpha(0.3),
  },
};

// Gradient definitions
export const GRADIENTS = {
  background: `linear-gradient(135deg, ${COLORS.slate950} 0%, ${COLORS.slate900} 50%, ${COLORS.slate950} 100%)`,
  voxformer: `linear-gradient(135deg, ${COLORS.cyan500} 0%, ${COLORS.purple500} 100%)`,
  rag: `linear-gradient(135deg, ${COLORS.emerald500} 0%, ${COLORS.cyan500} 100%)`,
  avatar: `linear-gradient(135deg, ${COLORS.rose500} 0%, ${COLORS.pink500} 100%)`,
  mcp: `linear-gradient(135deg, ${COLORS.orange500} 0%, ${COLORS.amber500} 100%)`,
  rainbow: `linear-gradient(135deg, ${COLORS.cyan500} 0%, ${COLORS.purple500} 25%, ${COLORS.rose500} 50%, ${COLORS.orange500} 75%, ${COLORS.emerald500} 100%)`,
};
