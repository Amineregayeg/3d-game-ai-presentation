// Harbyx Policy Sync Service
// Syncs local Salesforce Consultant policies to Harbyx dashboard

import { getHarbyxClient } from './client';
import { HarbyxError, PolicyConfig } from './types';
import {
  salesforceQueryPolicy,
  salesforceWritePolicy,
  salesforceDeployPolicy,
  agentToolPolicy,
  databasePolicy,
  externalApiPolicy,
} from './config';

/**
 * Convert local PolicyConfig to Harbyx API format
 */
function convertPolicyToHarbyxFormat(policy: PolicyConfig, priority: number) {
  return {
    name: policy.name,
    description: policy.description,
    priority,
    rules: policy.rules.map((rule) => ({
      actionType: rule.actionType,
      targetPattern: typeof rule.target === 'string' ? rule.target : rule.target.source,
      effect: rule.decision,
    })),
  };
}

/**
 * Sync all local policies to Harbyx
 * This will create policies in the Harbyx dashboard
 */
export async function syncPoliciesToHarbyx(): Promise<{
  success: boolean;
  created: string[];
  errors: string[];
}> {
  const client = getHarbyxClient();
  const created: string[] = [];
  const errors: string[] = [];

  // Policies in priority order
  const localPolicies: Array<{ policy: PolicyConfig; priority: number }> = [
    { policy: salesforceQueryPolicy, priority: 100 },
    { policy: salesforceWritePolicy, priority: 90 },
    { policy: salesforceDeployPolicy, priority: 80 },
    { policy: agentToolPolicy, priority: 70 },
    { policy: databasePolicy, priority: 60 },
    { policy: externalApiPolicy, priority: 50 },
  ];

  // First, get existing policies to avoid duplicates
  let existingPolicies: Array<{ id: string; name: string }> = [];
  try {
    existingPolicies = await client.listPolicies();
  } catch (error) {
    console.log('[Harbyx Sync] Could not fetch existing policies, will create all');
  }

  const existingNames = new Set(existingPolicies.map((p) => p.name));

  for (const { policy, priority } of localPolicies) {
    try {
      // Skip if policy already exists
      if (existingNames.has(policy.name)) {
        console.log(`[Harbyx Sync] Policy "${policy.name}" already exists, skipping`);
        continue;
      }

      const harbyxPolicy = convertPolicyToHarbyxFormat(policy, priority);
      const result = await client.createPolicy(harbyxPolicy);
      created.push(policy.name);
      console.log(`[Harbyx Sync] Created policy: ${policy.name} (${result.policy_id})`);
    } catch (error) {
      const message = error instanceof HarbyxError ? error.message : 'Unknown error';
      errors.push(`${policy.name}: ${message}`);
      console.error(`[Harbyx Sync] Failed to create policy "${policy.name}":`, message);
    }
  }

  return {
    success: errors.length === 0,
    created,
    errors,
  };
}

/**
 * Get sync status - compare local vs Harbyx policies
 */
export async function getSyncStatus(): Promise<{
  localCount: number;
  remoteCount: number;
  synced: boolean;
  localPolicies: string[];
  remotePolicies: string[];
}> {
  const client = getHarbyxClient();
  const localPolicies = [
    salesforceQueryPolicy,
    salesforceWritePolicy,
    salesforceDeployPolicy,
    agentToolPolicy,
    databasePolicy,
    externalApiPolicy,
  ];
  const localNames = localPolicies.map((p) => p.name);

  let remotePolicies: Array<{ name: string }> = [];
  try {
    remotePolicies = await client.listPolicies();
  } catch (error) {
    console.error('[Harbyx Sync] Failed to fetch remote policies');
  }

  const remoteNames = remotePolicies.map((p) => p.name);

  // Check if all local policies exist remotely
  const allSynced = localNames.every((name) => remoteNames.includes(name));

  return {
    localCount: localPolicies.length,
    remoteCount: remotePolicies.length,
    synced: allSynced,
    localPolicies: localNames,
    remotePolicies: remoteNames,
  };
}

/**
 * Clear all policies from Harbyx (use with caution)
 */
export async function clearHarbyxPolicies(): Promise<{
  success: boolean;
  deleted: number;
}> {
  const client = getHarbyxClient();
  let deleted = 0;

  try {
    const policies = await client.listPolicies();

    for (const policy of policies) {
      try {
        await client.deletePolicy(policy.id);
        deleted++;
        console.log(`[Harbyx Sync] Deleted policy: ${policy.name}`);
      } catch (error) {
        console.error(`[Harbyx Sync] Failed to delete policy ${policy.name}`);
      }
    }

    return { success: true, deleted };
  } catch (error) {
    return { success: false, deleted };
  }
}
