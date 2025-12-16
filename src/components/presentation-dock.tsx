//@ts-nocheck
'use client';

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/magnified-doc';
import {
  type MotionValue,
  motion,
  useMotionValue,
  useSpring,
  useTransform,
  AnimatePresence,
} from 'motion/react';
import Image from 'next/image';
import Link from 'next/link';
import React, { useRef, useState, useEffect } from 'react';

interface PresentationApp {
  id: string;
  name: string;
  href: string;
  icon?: React.ReactNode;
  iconUrl?: string;
  color: string;
  bgGradient: string;
}

interface PresentationDockProps {
  items: PresentationApp[];
}

export function PresentationDock({ items }: PresentationDockProps) {
  const mouseX = useMotionValue(Infinity);
  const [isVisible, setIsVisible] = useState(true);
  const [isHovering, setIsHovering] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-hide after 3 seconds on initial load
  useEffect(() => {
    timeoutRef.current = setTimeout(() => {
      if (!isHovering) {
        setIsVisible(false);
      }
    }, 3000);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  // Handle hover on trigger bar
  const handleTriggerEnter = () => {
    setIsHovering(true);
    setIsVisible(true);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };

  // Handle mouse leave from dock area
  const handleDockLeave = () => {
    setIsHovering(false);
    mouseX.set(Infinity);
    // Hide after a short delay when mouse leaves
    timeoutRef.current = setTimeout(() => {
      setIsVisible(false);
    }, 500);
  };

  // Handle mouse enter on dock (keep it visible)
  const handleDockEnter = () => {
    setIsHovering(true);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };

  return (
    <div className='fixed bottom-0 left-0 right-0 z-50 flex flex-col items-center'>
      {/* Dock with slide animation */}
      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            transition={{
              type: 'spring',
              stiffness: 300,
              damping: 30,
              mass: 0.8
            }}
            onMouseEnter={handleDockEnter}
            onMouseLeave={handleDockLeave}
            className='mb-0'
          >
            <TooltipProvider delayDuration={0}>
              <motion.div
                onMouseMove={(e) => mouseX.set(e.pageX)}
                onMouseLeave={() => mouseX.set(Infinity)}
                className='mx-auto flex items-end gap-2 rounded-t-2xl px-3 py-3 dark:bg-slate-900/90 bg-slate-900/90 backdrop-blur-xl border border-b-0 border-slate-700/50'
                style={{
                  boxShadow: '0 -20px 40px rgba(0, 0, 0, 0.3)',
                }}
              >
                {items.map((app) => {
                  return (
                    <Tooltip key={app.id}>
                      <TooltipTrigger asChild>
                        <Link href={app.href}>
                          <span className='sr-only'>{app.name}</span>
                          <AppIcon
                            mouseX={mouseX}
                            icon={app.icon}
                            iconUrl={app.iconUrl}
                            bgGradient={app.bgGradient}
                          />
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent className='py-1.5 px-3 rounded-md' sideOffset={12}>
                        <p className='text-xs font-medium'>{app.name}</p>
                      </TooltipContent>
                    </Tooltip>
                  );
                })}
              </motion.div>
            </TooltipProvider>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Subtle trigger bar - always visible when dock is hidden */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: isVisible ? 0 : 1 }}
        transition={{ duration: 0.3, delay: isVisible ? 0 : 0.2 }}
        onMouseEnter={handleTriggerEnter}
        className='w-32 h-1 bg-white/30 rounded-full mb-2 cursor-pointer hover:bg-white/50 transition-colors'
        style={{ pointerEvents: isVisible ? 'none' : 'auto' }}
      />
    </div>
  );
}

function AppIcon({
  mouseX,
  icon,
  iconUrl,
  bgGradient,
}: {
  mouseX: MotionValue;
  icon?: React.ReactNode;
  iconUrl?: string;
  bgGradient: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const distance = useTransform(mouseX, (val) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });

  const widthSync = useTransform(distance, [-100, 0, 100], [40, 65, 40]);
  const width = useSpring(widthSync, {
    mass: 0.1,
    stiffness: 150,
    damping: 12,
  });

  return (
    <motion.div
      ref={ref}
      style={{ width }}
      className={`relative h-12 rounded-lg overflow-hidden flex items-center justify-center ${bgGradient} hover:shadow-lg transition-shadow`}
    >
      <div className='flex items-center justify-center w-full h-full p-2'>
        {iconUrl ? (
          <img
            src={iconUrl}
            alt=""
            className="w-6 h-6 object-contain"
            style={{ filter: 'brightness(0) invert(1)' }}
          />
        ) : (
          icon
        )}
      </div>
    </motion.div>
  );
}
