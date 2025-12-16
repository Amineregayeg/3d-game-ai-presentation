"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname, useSearchParams } from "next/navigation";

interface UseSlideNavigationOptions {
  totalSlides: number;
  onSlideChange?: (slideIndex: number) => void;
}

export function useSlideNavigation({ totalSlides, onSlideChange }: UseSlideNavigationOptions) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [currentSlide, setCurrentSlide] = useState(0);

  // Parse slide from URL on mount and when URL changes
  useEffect(() => {
    const slideParam = searchParams.get("slide");
    if (slideParam) {
      const slideIndex = parseInt(slideParam, 10) - 1; // URL uses 1-based index
      if (!isNaN(slideIndex) && slideIndex >= 0 && slideIndex < totalSlides) {
        setCurrentSlide(slideIndex);
      }
    } else if (typeof window !== "undefined") {
      // Check for hash-based navigation (e.g., #slide-3)
      const hash = window.location.hash;
      const match = hash.match(/^#slide-(\d+)$/);
      if (match) {
        const slideIndex = parseInt(match[1], 10) - 1;
        if (slideIndex >= 0 && slideIndex < totalSlides) {
          setCurrentSlide(slideIndex);
        }
      }
    }
  }, [searchParams, totalSlides]);

  // Update URL when slide changes
  const updateUrl = useCallback((slideIndex: number) => {
    if (typeof window !== "undefined") {
      const newUrl = `${pathname}#slide-${slideIndex + 1}`;
      window.history.replaceState(null, "", newUrl);
    }
  }, [pathname]);

  const goToSlide = useCallback((index: number) => {
    if (index >= 0 && index < totalSlides) {
      setCurrentSlide(index);
      updateUrl(index);
      onSlideChange?.(index);
    }
  }, [totalSlides, updateUrl, onSlideChange]);

  const goToNextSlide = useCallback(() => {
    if (currentSlide < totalSlides - 1) {
      goToSlide(currentSlide + 1);
    }
  }, [currentSlide, totalSlides, goToSlide]);

  const goToPrevSlide = useCallback(() => {
    if (currentSlide > 0) {
      goToSlide(currentSlide - 1);
    }
  }, [currentSlide, goToSlide]);

  const goToFirstSlide = useCallback(() => {
    goToSlide(0);
  }, [goToSlide]);

  const goToLastSlide = useCallback(() => {
    goToSlide(totalSlides - 1);
  }, [totalSlides, goToSlide]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case "ArrowRight":
        case " ":
        case "Enter":
          e.preventDefault();
          goToNextSlide();
          break;
        case "ArrowLeft":
        case "Backspace":
          e.preventDefault();
          goToPrevSlide();
          break;
        case "Home":
          e.preventDefault();
          goToFirstSlide();
          break;
        case "End":
          e.preventDefault();
          goToLastSlide();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goToNextSlide, goToPrevSlide, goToFirstSlide, goToLastSlide]);

  // Handle browser back/forward
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash;
      const match = hash.match(/^#slide-(\d+)$/);
      if (match) {
        const slideIndex = parseInt(match[1], 10) - 1;
        if (slideIndex >= 0 && slideIndex < totalSlides) {
          setCurrentSlide(slideIndex);
        }
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [totalSlides]);

  return {
    currentSlide,
    goToSlide,
    goToNextSlide,
    goToPrevSlide,
    goToFirstSlide,
    goToLastSlide,
    isFirstSlide: currentSlide === 0,
    isLastSlide: currentSlide === totalSlides - 1,
  };
}
