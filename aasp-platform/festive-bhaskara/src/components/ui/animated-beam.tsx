'use client';

import { cn } from '@/lib/utils';
import { motion } from 'motion/react';
import * as React from 'react';

interface AnimatedBeamProps {
  className?: string;
  containerRef: React.RefObject<HTMLElement | null>;
  fromRef: React.RefObject<HTMLElement | null>;
  toRef: React.RefObject<HTMLElement | null>;
  curvature?: number;
  reverse?: boolean;
  duration?: number;
  delay?: number;
  pathColor?: string;
  pathWidth?: number;
  pathOpacity?: number;
  gradientStartColor?: string;
  gradientStopColor?: string;
}

export function AnimatedBeam({
  className,
  containerRef,
  fromRef,
  toRef,
  curvature = 0,
  reverse = false,
  duration = 2,
  delay = 0,
  pathColor = 'gray',
  pathWidth = 2,
  pathOpacity = 0.2,
  gradientStartColor = '#18CCFC',
  gradientStopColor = '#6344F5',
}: AnimatedBeamProps) {
  const id = React.useId();
  const [pathD, setPathD] = React.useState('');
  const [svgDimensions, setSvgDimensions] = React.useState({ width: 0, height: 0 });

  React.useEffect(() => {
    const updatePath = () => {
      if (containerRef.current && fromRef.current && toRef.current) {
        const containerRect = containerRef.current.getBoundingClientRect();
        const fromRect = fromRef.current.getBoundingClientRect();
        const toRect = toRef.current.getBoundingClientRect();

        const startX = fromRect.left - containerRect.left + fromRect.width / 2;
        const startY = fromRect.top - containerRect.top + fromRect.height / 2;
        const endX = toRect.left - containerRect.left + toRect.width / 2;
        const endY = toRect.top - containerRect.top + toRect.height / 2;

        const controlX = (startX + endX) / 2;
        const controlY = (startY + endY) / 2 + curvature;

        setPathD(`M ${startX} ${startY} Q ${controlX} ${controlY} ${endX} ${endY}`);
        setSvgDimensions({
          width: containerRect.width,
          height: containerRect.height,
        });
      }
    };

    updatePath();
    const resizeObserver = new ResizeObserver(updatePath);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, [containerRef, fromRef, toRef, curvature]);

  return (
    <svg
      fill="none"
      width={svgDimensions.width}
      height={svgDimensions.height}
      xmlns="http://www.w3.org/2000/svg"
      className={cn('pointer-events-none absolute left-0 top-0', className)}
    >
      <path
        d={pathD}
        stroke={pathColor}
        strokeWidth={pathWidth}
        strokeOpacity={pathOpacity}
        strokeLinecap="round"
      />
      <path
        d={pathD}
        stroke={`url(#${id})`}
        strokeWidth={pathWidth}
        strokeOpacity="1"
        strokeLinecap="round"
      />
      <defs>
        <motion.linearGradient
          id={id}
          gradientUnits="userSpaceOnUse"
          initial={{
            x1: reverse ? '0%' : '100%',
            y1: reverse ? '0%' : '100%',
            x2: reverse ? '0%' : '100%',
            y2: reverse ? '0%' : '100%',
          }}
          animate={{
            x1: reverse ? '100%' : '0%',
            y1: reverse ? '100%' : '0%',
            x2: reverse ? '200%' : '-100%',
            y2: reverse ? '200%' : '-100%',
          }}
          transition={{
            duration,
            delay,
            repeat: Infinity,
            repeatType: 'loop',
            ease: 'linear',
          }}
        >
          <stop stopColor={gradientStartColor} stopOpacity="0" />
          <stop offset="0.5" stopColor={gradientStartColor} />
          <stop offset="1" stopColor={gradientStopColor} stopOpacity="0" />
        </motion.linearGradient>
      </defs>
    </svg>
  );
}

interface BeamNodeProps {
  children: React.ReactNode;
  className?: string;
}

export const BeamNode = React.forwardRef<HTMLDivElement, BeamNodeProps>(
  ({ children, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'relative z-10 flex items-center justify-center rounded-xl border border-border/50 bg-card p-4 shadow-lg',
          className
        )}
      >
        {children}
      </div>
    );
  }
);
BeamNode.displayName = 'BeamNode';
