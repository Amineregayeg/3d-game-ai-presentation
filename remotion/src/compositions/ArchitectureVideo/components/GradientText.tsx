import React from 'react';
import { GRADIENTS } from '../../../utils/colors';

interface GradientTextProps {
  children: React.ReactNode;
  gradient?: string;
  fontSize?: number;
  fontWeight?: number;
  style?: React.CSSProperties;
}

export const GradientText: React.FC<GradientTextProps> = ({
  children,
  gradient = GRADIENTS.voxformer,
  fontSize = 48,
  fontWeight = 700,
  style = {},
}) => {
  return (
    <span
      style={{
        fontSize,
        fontWeight,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        background: gradient,
        backgroundClip: 'text',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        ...style,
      }}
    >
      {children}
    </span>
  );
};
