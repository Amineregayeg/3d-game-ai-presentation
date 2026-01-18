'use client';

import { cn } from '@/lib/utils';
import { motion, useInView } from 'motion/react';
import * as React from 'react';

interface TimelineItem {
  id: string;
  title: string;
  description: string;
  date?: string;
  status?: 'completed' | 'current' | 'upcoming';
  icon?: React.ReactNode;
}

interface TimelineProps {
  items: TimelineItem[];
  className?: string;
}

export function Timeline({ items, className }: TimelineProps) {
  return (
    <div className={cn('relative', className)}>
      {/* Vertical line */}
      <div className="absolute left-6 top-0 h-full w-px bg-gradient-to-b from-primary/50 via-primary to-primary/50" />

      <div className="space-y-8">
        {items.map((item, index) => (
          <TimelineEntry key={item.id} item={item} index={index} />
        ))}
      </div>
    </div>
  );
}

function TimelineEntry({ item, index }: { item: TimelineItem; index: number }) {
  const ref = React.useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  const statusColors = {
    completed: 'bg-green-500 shadow-green-500/30',
    current: 'bg-primary shadow-primary/30 animate-pulse',
    upcoming: 'bg-muted-foreground/50',
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, x: -20 }}
      animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="relative pl-16"
    >
      {/* Node */}
      <div
        className={cn(
          'absolute left-4 top-1 h-5 w-5 rounded-full border-4 border-background shadow-lg',
          statusColors[item.status || 'upcoming']
        )}
      />

      {/* Content */}
      <div className="rounded-xl border border-border/50 bg-card/50 p-5 backdrop-blur-sm">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              {item.icon && (
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  {item.icon}
                </div>
              )}
              <h3 className="font-semibold text-foreground">{item.title}</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {item.description}
            </p>
          </div>
          {item.date && (
            <span className="shrink-0 rounded-full bg-secondary px-3 py-1 text-xs font-medium text-muted-foreground">
              {item.date}
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
}
