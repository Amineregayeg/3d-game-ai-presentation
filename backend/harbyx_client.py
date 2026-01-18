"""
Harbyx Governance Python Client
Integrates with Harbyx API for policy evaluation and approval management
For Salesforce Consultant Agent
"""

import os
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any

# Configuration from environment
HARBYX_API_KEY = os.environ.get('HARBYX_API_KEY', '')
HARBYX_BASE_URL = os.environ.get('HARBYX_BASE_URL', 'https://app.harbyx.com')
HARBYX_AGENT_ID = os.environ.get('HARBYX_AGENT_ID', 'salesforce-consultant-agent')
HARBYX_TIMEOUT = int(os.environ.get('HARBYX_TIMEOUT', '30'))


class HarbyxError(Exception):
    """Custom exception for Harbyx API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class HarbyxClient:
    """
    Python client for Harbyx Governance API

    Usage:
        client = HarbyxClient()
        result = client.evaluate_action('api_call', 'salesforce:insert')
        if result['decision'] == 'allow':
            # proceed with action
        elif result['decision'] == 'require_approval':
            # wait for approval
        elif result['decision'] == 'block':
            # action not permitted
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 agent_id: Optional[str] = None, timeout: Optional[int] = None):
        self.api_key = api_key or HARBYX_API_KEY
        self.base_url = base_url or HARBYX_BASE_URL
        self.agent_id = agent_id or HARBYX_AGENT_ID
        self.timeout = timeout or HARBYX_TIMEOUT

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an authenticated request to Harbyx API"""
        if not self.api_key:
            raise HarbyxError('HARBYX_API_KEY not configured', 500, 'MISSING_API_KEY')

        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )

            if not response.ok:
                error_data = response.json() if response.text else {}
                raise HarbyxError(
                    error_data.get('message', f'Request failed with status {response.status_code}'),
                    response.status_code,
                    error_data.get('code')
                )

            return response.json()

        except requests.exceptions.Timeout:
            raise HarbyxError('Request timed out', 408, 'TIMEOUT')
        except requests.exceptions.RequestException as e:
            raise HarbyxError(str(e), 500, 'REQUEST_ERROR')

    # ==================== Action Evaluation ====================

    def evaluate_action(self, action_type: str, target: str,
                       params: Optional[Dict] = None,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate an action against Harbyx policies

        Args:
            action_type: Type of action (tool_call, api_call, db_query, file_access, external_request)
            target: Action target (e.g., 'salesforce:insert', 'rag:query')
            params: Optional parameters for the action
            metadata: Optional metadata (timestamp, user, etc.)

        Returns:
            {
                'success': bool,
                'decision': 'allow' | 'block' | 'require_approval',
                'action_id': str,
                'reason': str,
                'approval_id': str | None,
                'policy_id': str | None
            }
        """
        return self._request('POST', '/api/v1/ingest', {
            'agent_id': self.agent_id,
            'action_type': action_type,
            'target': target,
            'params': params or {},
            'metadata': metadata or {'timestamp': datetime.utcnow().isoformat()}
        })

    def evaluate_action_dry_run(self, action_type: str, target: str,
                                params: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate action without side effects (dry run)"""
        return self._request('POST', '/api/v1/ingest?dry_run=true', {
            'agent_id': self.agent_id,
            'action_type': action_type,
            'target': target,
            'params': params or {},
            'metadata': {'dry_run': True}
        })

    def evaluate_batch(self, actions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate multiple actions in batch (max 100)

        Args:
            actions: List of dicts with keys: action_type, target, params, metadata
        """
        if len(actions) > 100:
            raise HarbyxError('Batch size cannot exceed 100 actions', 400, 'BATCH_TOO_LARGE')

        return self._request('POST', '/api/v1/ingest/batch', {
            'actions': [{
                'agent_id': self.agent_id,
                'action_type': a['action_type'],
                'target': a['target'],
                'params': a.get('params', {}),
                'metadata': a.get('metadata', {})
            } for a in actions]
        })

    # ==================== Approval Management ====================

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Fetch all pending approvals"""
        return self._request('GET', '/api/v1/approvals?status=pending')

    def get_approval_status(self, approval_id: str) -> Dict[str, Any]:
        """Check status of a specific approval"""
        return self._request('GET', f'/api/v1/approvals/{approval_id}/status')

    def decide_approval(self, approval_id: str, decision: str,
                       reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Approve or reject an action

        Args:
            approval_id: The approval request ID
            decision: 'approve' or 'reject'
            reason: Optional reason for the decision
        """
        if decision not in ('approve', 'reject'):
            raise HarbyxError('Decision must be approve or reject', 400, 'INVALID_DECISION')

        return self._request('POST', f'/api/v1/approvals/{approval_id}', {
            'decision': decision,
            'reason': reason
        })

    # ==================== Policy Management ====================

    def list_policies(self) -> List[Dict[str, Any]]:
        """List all policies in Harbyx"""
        return self._request('GET', '/api/v1/policies')

    def create_policy(self, name: str, rules: List[Dict],
                     description: Optional[str] = None,
                     priority: int = 100) -> Dict[str, Any]:
        """
        Create a new policy in Harbyx

        Args:
            name: Policy name
            rules: List of rule dicts with keys: actionType, targetPattern, effect
            description: Optional policy description
            priority: Policy priority (higher = evaluated first)
        """
        return self._request('POST', '/api/v1/policies', {
            'name': name,
            'description': description,
            'priority': priority,
            'rules': rules
        })

    def update_policy(self, policy_id: str,
                     name: Optional[str] = None,
                     rules: Optional[List[Dict]] = None,
                     description: Optional[str] = None,
                     priority: Optional[int] = None) -> Dict[str, Any]:
        """Update an existing policy"""
        data = {}
        if name:
            data['name'] = name
        if rules:
            data['rules'] = rules
        if description:
            data['description'] = description
        if priority:
            data['priority'] = priority

        return self._request('PATCH', f'/api/v1/policies/{policy_id}', data)

    def delete_policy(self, policy_id: str) -> Dict[str, Any]:
        """Delete a policy from Harbyx"""
        return self._request('DELETE', f'/api/v1/policies/{policy_id}')

    # ==================== Webhooks ====================

    def register_webhook(self, url: str, events: List[str]) -> Dict[str, Any]:
        """
        Register a webhook for real-time notifications

        Args:
            url: Webhook endpoint URL
            events: List of events to subscribe to:
                    ['approval.created', 'approval.decided', 'policy.updated']
        """
        return self._request('POST', '/api/v1/webhooks', {
            'url': url,
            'events': events
        })

    # ==================== Health Check ====================

    def health_check(self) -> Dict[str, Any]:
        """Check Harbyx connectivity"""
        try:
            self.evaluate_action_dry_run('api_call', 'health')
            return {'status': 'ok', 'connected': True}
        except HarbyxError as e:
            return {'status': e.message, 'connected': False}
        except Exception as e:
            return {'status': str(e), 'connected': False}


# Singleton instance
_client_instance: Optional[HarbyxClient] = None

def get_harbyx_client() -> HarbyxClient:
    """Get singleton Harbyx client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = HarbyxClient()
    return _client_instance


# Convenience functions
def evaluate_action(action_type: str, target: str,
                   params: Optional[Dict] = None) -> Dict[str, Any]:
    """Quick action evaluation"""
    return get_harbyx_client().evaluate_action(action_type, target, params)

def is_action_allowed(action_type: str, target: str) -> bool:
    """Check if action is allowed (returns False for block or require_approval)"""
    try:
        result = get_harbyx_client().evaluate_action_dry_run(action_type, target)
        return result.get('decision') == 'allow'
    except HarbyxError:
        return False


# Salesforce-specific policy definitions for syncing
SALESFORCE_POLICIES = [
    {
        'name': 'Salesforce Query Policy',
        'description': 'Controls read-only Salesforce query operations',
        'priority': 100,
        'rules': [
            {'actionType': 'api_call', 'targetPattern': 'salesforce:query', 'effect': 'allow'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:queryAll', 'effect': 'allow'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:describe', 'effect': 'allow'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:describeGlobal', 'effect': 'allow'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:limits', 'effect': 'allow'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:search', 'effect': 'allow'},
        ]
    },
    {
        'name': 'Salesforce Write Policy',
        'description': 'Controls Salesforce data modification operations',
        'priority': 90,
        'rules': [
            {'actionType': 'api_call', 'targetPattern': 'salesforce:insert', 'effect': 'require_approval'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:update', 'effect': 'require_approval'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:upsert', 'effect': 'require_approval'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:delete', 'effect': 'block'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:hardDelete', 'effect': 'block'},
        ]
    },
    {
        'name': 'Salesforce Deployment Policy',
        'description': 'Controls Salesforce metadata and deployment operations',
        'priority': 80,
        'rules': [
            {'actionType': 'api_call', 'targetPattern': 'salesforce:metadata:read', 'effect': 'allow'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:metadata:write', 'effect': 'require_approval'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:deploy', 'effect': 'require_approval'},
            {'actionType': 'api_call', 'targetPattern': 'salesforce:deploy:production', 'effect': 'block'},
        ]
    },
    {
        'name': 'Agent Tool Policy',
        'description': 'Controls AI agent tool usage',
        'priority': 70,
        'rules': [
            {'actionType': 'tool_call', 'targetPattern': 'rag:*', 'effect': 'allow'},
            {'actionType': 'tool_call', 'targetPattern': 'mcp:salesforce:*:read', 'effect': 'allow'},
            {'actionType': 'tool_call', 'targetPattern': 'mcp:salesforce:*:write', 'effect': 'require_approval'},
            {'actionType': 'tool_call', 'targetPattern': 'code:execute', 'effect': 'require_approval'},
            {'actionType': 'tool_call', 'targetPattern': 'shell:*', 'effect': 'block'},
        ]
    }
]


def sync_policies_to_harbyx() -> Dict[str, Any]:
    """Sync local Salesforce policies to Harbyx"""
    client = get_harbyx_client()
    created = []
    errors = []

    # Get existing policies
    try:
        existing = client.list_policies()
        existing_names = {p['name'] for p in existing}
    except HarbyxError:
        existing_names = set()

    for policy in SALESFORCE_POLICIES:
        if policy['name'] in existing_names:
            print(f"[Harbyx Sync] Policy '{policy['name']}' already exists, skipping")
            continue

        try:
            result = client.create_policy(
                name=policy['name'],
                rules=policy['rules'],
                description=policy.get('description'),
                priority=policy.get('priority', 100)
            )
            created.append(policy['name'])
            print(f"[Harbyx Sync] Created policy: {policy['name']}")
        except HarbyxError as e:
            errors.append(f"{policy['name']}: {e.message}")
            print(f"[Harbyx Sync] Failed to create '{policy['name']}': {e.message}")

    return {
        'success': len(errors) == 0,
        'created': created,
        'errors': errors
    }
