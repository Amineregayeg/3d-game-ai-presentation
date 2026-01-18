"use client";

import * as React from "react";
import { motion } from "motion/react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface AvatarItem {
  src: string;
  fallback: string;
  name: string;
}

interface AnimatedAvatarGroupProps {
  avatars: AvatarItem[];
  maxVisible?: number;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeClasses = {
  sm: "size-8",
  md: "size-12",
  lg: "size-16",
};

const overlapClasses = {
  sm: "-space-x-2",
  md: "-space-x-3",
  lg: "-space-x-4",
};

export function AnimatedAvatarGroup({
  avatars,
  maxVisible = 6,
  size = "md",
  className,
}: AnimatedAvatarGroupProps) {
  const visibleAvatars = avatars.slice(0, maxVisible);
  const remainingCount = avatars.length - maxVisible;

  return (
    <TooltipProvider delayDuration={0}>
      <div
        className={cn(
          "flex items-center",
          overlapClasses[size],
          className
        )}
      >
        {visibleAvatars.map((avatar, index) => (
          <Tooltip key={index}>
            <TooltipTrigger asChild>
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1, duration: 0.3 }}
                whileHover={{
                  y: -8,
                  zIndex: 50,
                  transition: { type: "spring", stiffness: 300, damping: 20 }
                }}
                style={{
                  zIndex: visibleAvatars.length - index,
                  position: "relative"
                }}
                className="cursor-pointer"
              >
                <Avatar
                  className={cn(
                    sizeClasses[size],
                    "border-2 border-slate-900 ring-2 ring-slate-800/50 transition-shadow hover:ring-[#00A1E0]/50"
                  )}
                >
                  <AvatarImage src={avatar.src} alt={avatar.name} />
                  <AvatarFallback className="bg-[#00A1E0] text-white text-xs font-medium">
                    {avatar.fallback}
                  </AvatarFallback>
                </Avatar>
              </motion.div>
            </TooltipTrigger>
            <TooltipContent
              side="top"
              className="bg-slate-800 text-white border-slate-700 px-3 py-1.5"
            >
              <p className="text-sm font-medium">{avatar.name}</p>
            </TooltipContent>
          </Tooltip>
        ))}

        {remainingCount > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: maxVisible * 0.1, duration: 0.3 }}
                whileHover={{
                  y: -8,
                  zIndex: 50,
                  transition: { type: "spring", stiffness: 300, damping: 20 }
                }}
                style={{ zIndex: 0, position: "relative" }}
                className="cursor-pointer"
              >
                <div
                  className={cn(
                    sizeClasses[size],
                    "flex items-center justify-center rounded-full bg-[#00A1E0]/20 border-2 border-slate-900 ring-2 ring-slate-800/50 hover:ring-[#00A1E0]/50 transition-shadow"
                  )}
                >
                  <span className="text-xs font-semibold text-[#00A1E0]">
                    +{remainingCount}
                  </span>
                </div>
              </motion.div>
            </TooltipTrigger>
            <TooltipContent
              side="top"
              className="bg-slate-800 text-white border-slate-700 px-3 py-1.5"
            >
              <p className="text-sm font-medium">+{remainingCount} more consultants</p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
    </TooltipProvider>
  );
}
