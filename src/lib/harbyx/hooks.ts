"use client";

// Harbyx React Hooks for Governance

import { useState, useCallback } from "react";
import { toast } from "sonner";
import { Decision } from "./types";

export interface GovernanceState {
  isLoading: boolean;
  decision: Decision | null;
  approvalId: string | null;
  reason: string | null;
  error: string | null;
}

export interface UseGovernanceOptions {
  onBlock?: (reason: string) => void;
  onApprovalRequired?: (approvalId: string, reason: string) => void;
  onAllow?: () => void;
  showToasts?: boolean;
}

/**
 * Hook for handling governance decisions in React components
 */
export function useGovernance(options: UseGovernanceOptions = {}) {
  const { showToasts = true } = options;

  const [state, setState] = useState<GovernanceState>({
    isLoading: false,
    decision: null,
    approvalId: null,
    reason: null,
    error: null,
  });

  const handleDecision = useCallback(
    (decision: Decision, reason?: string, approvalId?: string) => {
      setState({
        isLoading: false,
        decision,
        approvalId: approvalId || null,
        reason: reason || null,
        error: null,
      });

      switch (decision) {
        case "allow":
          options.onAllow?.();
          break;

        case "block":
          if (showToasts) {
            toast.error("Action Blocked", {
              description: reason || "This action is not permitted by policy",
            });
          }
          options.onBlock?.(reason || "Blocked by policy");
          break;

        case "require_approval":
          if (showToasts) {
            toast.info("Approval Required", {
              description: reason || "This action requires approval before proceeding",
            });
          }
          options.onApprovalRequired?.(approvalId || "", reason || "Approval required");
          break;
      }
    },
    [options, showToasts]
  );

  const reset = useCallback(() => {
    setState({
      isLoading: false,
      decision: null,
      approvalId: null,
      reason: null,
      error: null,
    });
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    setState((prev) => ({ ...prev, isLoading: loading }));
  }, []);

  const setError = useCallback((error: string) => {
    setState((prev) => ({ ...prev, error, isLoading: false }));
    if (showToasts) {
      toast.error("Governance Error", { description: error });
    }
  }, [showToasts]);

  return {
    ...state,
    handleDecision,
    reset,
    setLoading,
    setError,
    isBlocked: state.decision === "block",
    isPending: state.decision === "require_approval",
    isAllowed: state.decision === "allow",
  };
}

/**
 * Hook for checking approval status
 */
export function useApprovalStatus(approvalId: string | null) {
  const [status, setStatus] = useState<"pending" | "approved" | "rejected" | "expired" | null>(null);
  const [isChecking, setIsChecking] = useState(false);

  const checkStatus = useCallback(async () => {
    if (!approvalId) return;

    setIsChecking(true);
    try {
      const response = await fetch(`/api/harbyx/approvals/${approvalId}/status`);
      const data = await response.json();
      setStatus(data.status);
      return data.status;
    } catch (error) {
      console.error("Failed to check approval status:", error);
      return null;
    } finally {
      setIsChecking(false);
    }
  }, [approvalId]);

  return {
    status,
    isChecking,
    checkStatus,
    isApproved: status === "approved",
    isRejected: status === "rejected",
    isPending: status === "pending",
    isExpired: status === "expired",
  };
}
