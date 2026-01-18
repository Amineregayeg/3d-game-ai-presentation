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
import Link from 'next/link';
import React, { useRef, useState, useEffect } from 'react';
import { LucideIcon } from 'lucide-react';

interface NavItem {
  id: string;
  label: string;
  href: string;
  icon: LucideIcon;
}

interface DashboardDockProps {
  items: NavItem[];
  currentPath: string;
}

export function DashboardDock({ items, currentPath }: DashboardDockProps) {
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

  const handleTriggerEnter = () => {
    setIsHovering(true);
    setIsVisible(true);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };

  const handleDockLeave = () => {
    setIsHovering(false);
    mouseX.set(Infinity);
    timeoutRef.current = setTimeout(() => {
      setIsVisible(false);
    }, 500);
  };

  const handleDockEnter = () => {
    setIsHovering(true);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };

  // Check if current path matches item
  const isActive = (href: string) => {
    if (href === '/dashboard') {
      return currentPath === '/dashboard';
    }
    return currentPath.startsWith(href);
  };

  return (
    <div className='fixed bottom-0 left-0 right-0 z-50 flex flex-col items-center'>
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
                className='mx-auto flex items-end gap-2 rounded-t-2xl px-4 py-3 dark:bg-slate-900/90 bg-slate-900/90 backdrop-blur-xl border border-b-0 border-slate-700/50'
                style={{
                  boxShadow: '0 -20px 40px rgba(0, 0, 0, 0.3)',
                }}
              >
                {items.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.href);

                  return (
                    <Tooltip key={item.id}>
                      <TooltipTrigger asChild>
                        <Link href={item.href}>
                          <span className='sr-only'>{item.label}</span>
                          <DockIcon
                            mouseX={mouseX}
                            isActive={active}
                          >
                            <Icon className="w-5 h-5" />
                          </DockIcon>
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent className='py-1.5 px-3 rounded-md' sideOffset={12}>
                        <p className='text-xs font-medium'>{item.label}</p>
                      </TooltipContent>
                    </Tooltip>
                  );
                })}
              </motion.div>
            </TooltipProvider>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Trigger bar */}
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

function DockIcon({
  mouseX,
  isActive,
  children
}: {
  mouseX: MotionValue;
  isActive: boolean;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const distance = useTransform(mouseX, (val) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });

  const widthSync = useTransform(distance, [-100, 0, 100], [48, 64, 48]);
  const width = useSpring(widthSync, {
    mass: 0.1,
    stiffness: 150,
    damping: 12,
  });

  return (
    <motion.div
      ref={ref}
      style={{ width }}
      className={`relative h-12 rounded-xl overflow-hidden flex items-center justify-center transition-all duration-200 ${
        isActive
          ? 'bg-gradient-to-br from-blue-500 to-purple-500 text-white shadow-lg shadow-blue-500/25'
          : 'bg-white/10 text-slate-400 hover:text-white hover:bg-white/20'
      }`}
    >
      {children}
      {isActive && (
        <motion.div
          layoutId="activeIndicator"
          className="absolute bottom-1 w-1 h-1 rounded-full bg-white"
        />
      )}
    </motion.div>
  );
}
