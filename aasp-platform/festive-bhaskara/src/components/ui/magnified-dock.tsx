'use client';

import { cn } from '@/lib/utils';
import {
  motion,
  useMotionValue,
  useSpring,
  useTransform,
  type MotionValue,
} from 'motion/react';
import * as React from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';

const TooltipProvider = TooltipPrimitive.Provider;
const Tooltip = TooltipPrimitive.Root;
const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        'z-50 overflow-hidden rounded-lg border bg-popover/95 backdrop-blur-xl px-4 py-2 text-sm font-medium text-popover-foreground shadow-xl animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
        className
      )}
      {...props}
    />
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

interface DockProps {
  children: React.ReactNode;
  className?: string;
  magnification?: number;
  distance?: number;
}

interface DockItemProps {
  children: React.ReactNode;
  className?: string;
  label?: string;
  onClick?: () => void;
  isActive?: boolean;
}

const DockContext = React.createContext<{
  mouseX: MotionValue<number>;
  magnification: number;
  distance: number;
} | null>(null);

function useDock() {
  const context = React.useContext(DockContext);
  if (!context) {
    throw new Error('useDock must be used within a Dock');
  }
  return context;
}

export function Dock({
  children,
  className,
  magnification = 60,
  distance = 140,
}: DockProps) {
  const mouseX = useMotionValue(Infinity);

  return (
    <DockContext.Provider value={{ mouseX, magnification, distance }}>
      <TooltipProvider delayDuration={0}>
        <motion.div
          onMouseMove={(e) => mouseX.set(e.pageX)}
          onMouseLeave={() => mouseX.set(Infinity)}
          className={cn(
            'mx-auto flex h-16 items-end gap-3 rounded-2xl border border-border/50 bg-card/80 px-4 pb-3 backdrop-blur-2xl shadow-2xl',
            className
          )}
        >
          {children}
        </motion.div>
      </TooltipProvider>
    </DockContext.Provider>
  );
}

export function DockItem({
  children,
  className,
  label,
  onClick,
  isActive,
}: DockItemProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const { mouseX, magnification, distance } = useDock();

  const distanceCalc = useTransform(mouseX, (val: number) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });

  const widthSync = useTransform(
    distanceCalc,
    [-distance, 0, distance],
    [40, magnification, 40]
  );

  const width = useSpring(widthSync, {
    mass: 0.1,
    stiffness: 150,
    damping: 12,
  });

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div
          ref={ref}
          style={{ width }}
          onClick={onClick}
          className={cn(
            'relative aspect-square cursor-pointer rounded-xl flex items-center justify-center transition-colors',
            isActive
              ? 'bg-primary/20 text-primary shadow-lg shadow-primary/20'
              : 'bg-secondary/50 text-muted-foreground hover:bg-secondary hover:text-foreground',
            className
          )}
        >
          {children}
          {isActive && (
            <motion.div
              layoutId="dock-indicator"
              className="absolute -bottom-2 left-1/2 h-1 w-1 -translate-x-1/2 rounded-full bg-primary"
            />
          )}
        </motion.div>
      </TooltipTrigger>
      {label && <TooltipContent side="top">{label}</TooltipContent>}
    </Tooltip>
  );
}

export function DockSeparator() {
  return <div className="mx-1 h-8 w-px bg-border/50" />;
}
