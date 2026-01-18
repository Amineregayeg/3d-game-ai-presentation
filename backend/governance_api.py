"""
Governance API Module - Policies, Approvals, Audit, Knowledge Base, Patterns
Implements Sprint 5-7 features for Consulting Delivery OS
Now integrated with Harbyx Governance Platform
"""

import json
from datetime import datetime
from flask import Blueprint, request, jsonify

# Import models and helpers from main app
from app import (
    db, auth_required, get_current_user, filter_by_org,
    GovernancePolicy, ApprovalRequest, AuditLog, KnowledgeBase, Pattern
)

# Import Harbyx client for governance integration
try:
    from harbyx_client import (
        get_harbyx_client, HarbyxError, sync_policies_to_harbyx
    )
    HARBYX_AVAILABLE = True
except ImportError:
    HARBYX_AVAILABLE = False
    print("[Governance API] Warning: harbyx_client not available")

governance_bp = Blueprint('governance', __name__, url_prefix='/api/governance')

# ============== Audit Logging Helper ==============

def log_audit(action, entity_type, entity_id=None, description=None, changes=None, metadata=None):
    """Create an audit log entry"""
    user = get_current_user()
    if not user:
        return None

    log = AuditLog(
        organization_id=user.organization_id,
        user_id=user.id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        description=description,
        changes=json.dumps(changes) if changes else None,
        metadata=json.dumps(metadata) if metadata else None,
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()
    return log

# ============== Policy Endpoints ==============

@governance_bp.route('/policies', methods=['GET'])
@auth_required
def list_policies():
    """List all policies for the organization"""
    user = get_current_user()
    policies = filter_by_org(GovernancePolicy.query, GovernancePolicy, user).all()

    return jsonify({
        'policies': [{
            'id': p.id,
            'name': p.name,
            'type': p.type,
            'rules': json.loads(p.rules) if p.rules else {},
            'is_active': p.is_active,
            'created_at': p.created_at.isoformat() if p.created_at else None,
            'updated_at': p.updated_at.isoformat() if p.updated_at else None
        } for p in policies]
    })

@governance_bp.route('/policies', methods=['POST'])
@auth_required
def create_policy():
    """Create a new governance policy"""
    user = get_current_user()
    data = request.json

    if not data.get('name'):
        return jsonify({'error': 'Policy name is required'}), 400

    policy = GovernancePolicy(
        organization_id=user.organization_id,
        name=data['name'],
        type=data.get('type', 'type_a'),
        rules=json.dumps(data.get('rules', {})),
        is_active=data.get('is_active', True)
    )
    db.session.add(policy)
    db.session.commit()

    log_audit('create', 'policy', policy.id, f"Created policy: {policy.name}")

    return jsonify({
        'message': 'Policy created',
        'policy': {
            'id': policy.id,
            'name': policy.name,
            'type': policy.type,
            'is_active': policy.is_active
        }
    }), 201

@governance_bp.route('/policies/<int:id>', methods=['GET'])
@auth_required
def get_policy(id):
    """Get a specific policy"""
    user = get_current_user()
    policy = GovernancePolicy.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not policy:
        return jsonify({'error': 'Policy not found'}), 404

    return jsonify({
        'policy': {
            'id': policy.id,
            'name': policy.name,
            'type': policy.type,
            'rules': json.loads(policy.rules) if policy.rules else {},
            'is_active': policy.is_active,
            'created_at': policy.created_at.isoformat() if policy.created_at else None,
            'updated_at': policy.updated_at.isoformat() if policy.updated_at else None
        }
    })

@governance_bp.route('/policies/<int:id>', methods=['PUT'])
@auth_required
def update_policy(id):
    """Update a governance policy"""
    user = get_current_user()
    policy = GovernancePolicy.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not policy:
        return jsonify({'error': 'Policy not found'}), 404

    data = request.json
    old_values = {'name': policy.name, 'type': policy.type, 'is_active': policy.is_active}

    if 'name' in data:
        policy.name = data['name']
    if 'type' in data:
        policy.type = data['type']
    if 'rules' in data:
        policy.rules = json.dumps(data['rules'])
    if 'is_active' in data:
        policy.is_active = data['is_active']

    db.session.commit()

    log_audit('update', 'policy', policy.id, f"Updated policy: {policy.name}",
              changes={'old': old_values, 'new': data})

    return jsonify({'message': 'Policy updated', 'policy': {'id': policy.id, 'name': policy.name}})

@governance_bp.route('/policies/<int:id>', methods=['DELETE'])
@auth_required
def delete_policy(id):
    """Delete a governance policy"""
    user = get_current_user()
    policy = GovernancePolicy.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not policy:
        return jsonify({'error': 'Policy not found'}), 404

    policy_name = policy.name
    db.session.delete(policy)
    db.session.commit()

    log_audit('delete', 'policy', id, f"Deleted policy: {policy_name}")

    return jsonify({'message': 'Policy deleted'})

# ============== Approval Endpoints ==============

@governance_bp.route('/approvals', methods=['GET'])
@auth_required
def list_approvals():
    """List approval requests"""
    user = get_current_user()
    status = request.args.get('status', 'pending')

    query = ApprovalRequest.query.filter_by(organization_id=user.organization_id)
    if status != 'all':
        query = query.filter_by(status=status)

    approvals = query.order_by(ApprovalRequest.created_at.desc()).all()

    return jsonify({
        'approvals': [{
            'id': a.id,
            'action_type': a.action_type,
            'target': a.target,
            'context': json.loads(a.context) if a.context else {},
            'status': a.status,
            'requester': {
                'id': a.requester.id,
                'name': a.requester.name,
                'email': a.requester.email
            } if a.requester else None,
            'policy': {
                'id': a.policy.id,
                'name': a.policy.name
            } if a.policy else None,
            'created_at': a.created_at.isoformat() if a.created_at else None,
            'decided_at': a.decided_at.isoformat() if a.decided_at else None,
            'decision_reason': a.decision_reason
        } for a in approvals]
    })

@governance_bp.route('/approvals', methods=['POST'])
@auth_required
def create_approval():
    """Create a new approval request"""
    user = get_current_user()
    data = request.json

    if not data.get('action_type'):
        return jsonify({'error': 'Action type is required'}), 400

    approval = ApprovalRequest(
        organization_id=user.organization_id,
        user_id=user.id,
        policy_id=data.get('policy_id'),
        action_type=data['action_type'],
        target=data.get('target', ''),
        context=json.dumps(data.get('context', {})),
        status='pending'
    )
    db.session.add(approval)
    db.session.commit()

    log_audit('create', 'approval', approval.id,
              f"Created approval request for: {approval.action_type} - {approval.target}")

    return jsonify({
        'message': 'Approval request created',
        'approval': {'id': approval.id, 'status': approval.status}
    }), 201

@governance_bp.route('/approvals/<int:id>/decide', methods=['POST'])
@auth_required
def decide_approval(id):
    """Approve or reject an approval request"""
    user = get_current_user()
    approval = ApprovalRequest.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not approval:
        return jsonify({'error': 'Approval request not found'}), 404

    if approval.status != 'pending':
        return jsonify({'error': 'Approval has already been decided'}), 400

    data = request.json
    decision = data.get('decision')
    if decision not in ['approved', 'rejected']:
        return jsonify({'error': 'Decision must be approved or rejected'}), 400

    approval.status = decision
    approval.decision_by = user.id
    approval.decision_reason = data.get('reason', '')
    approval.decided_at = datetime.utcnow()
    db.session.commit()

    log_audit(decision.rstrip('d'), 'approval', approval.id,
              f"{decision.capitalize()} request for: {approval.action_type} - {approval.target}")

    return jsonify({'message': f'Approval {decision}', 'approval': {'id': approval.id, 'status': approval.status}})

# ============== Audit Log Endpoints ==============

@governance_bp.route('/audit', methods=['GET'])
@auth_required
def list_audit_logs():
    """List audit logs with filtering"""
    user = get_current_user()

    query = AuditLog.query.filter_by(organization_id=user.organization_id)

    # Filters
    entity_type = request.args.get('entity_type')
    action = request.args.get('action')
    user_id = request.args.get('user_id')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')

    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    if action:
        query = query.filter_by(action=action)
    if user_id:
        query = query.filter_by(user_id=int(user_id))
    if from_date:
        query = query.filter(AuditLog.created_at >= datetime.fromisoformat(from_date))
    if to_date:
        query = query.filter(AuditLog.created_at <= datetime.fromisoformat(to_date))

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)

    pagination = query.order_by(AuditLog.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        'logs': [{
            'id': log.id,
            'action': log.action,
            'entity_type': log.entity_type,
            'entity_id': log.entity_id,
            'description': log.description,
            'changes': json.loads(log.changes) if log.changes else None,
            'metadata': json.loads(log.metadata) if log.metadata else None,
            'ip_address': log.ip_address,
            'user': {
                'id': log.user.id,
                'name': log.user.name,
                'email': log.user.email
            } if log.user else None,
            'created_at': log.created_at.isoformat() if log.created_at else None
        } for log in pagination.items],
        'total': pagination.total,
        'page': pagination.page,
        'pages': pagination.pages,
        'per_page': pagination.per_page
    })

@governance_bp.route('/audit/export', methods=['GET'])
@auth_required
def export_audit_logs():
    """Export audit logs as CSV or JSON"""
    user = get_current_user()
    format = request.args.get('format', 'json')

    query = AuditLog.query.filter_by(organization_id=user.organization_id)

    # Same filters as list
    entity_type = request.args.get('entity_type')
    action = request.args.get('action')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')

    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    if action:
        query = query.filter_by(action=action)
    if from_date:
        query = query.filter(AuditLog.created_at >= datetime.fromisoformat(from_date))
    if to_date:
        query = query.filter(AuditLog.created_at <= datetime.fromisoformat(to_date))

    logs = query.order_by(AuditLog.created_at.desc()).limit(10000).all()

    if format == 'csv':
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Timestamp', 'Action', 'Entity Type', 'Entity ID', 'Description', 'User', 'IP Address'])

        for log in logs:
            writer.writerow([
                log.created_at.isoformat() if log.created_at else '',
                log.action,
                log.entity_type,
                log.entity_id or '',
                log.description or '',
                log.user.email if log.user else '',
                log.ip_address or ''
            ])

        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=audit_logs.csv'}
        )

    return jsonify({
        'logs': [{
            'timestamp': log.created_at.isoformat() if log.created_at else None,
            'action': log.action,
            'entity_type': log.entity_type,
            'entity_id': log.entity_id,
            'description': log.description,
            'user_email': log.user.email if log.user else None,
            'ip_address': log.ip_address
        } for log in logs]
    })

# ============== Knowledge Base Endpoints ==============

@governance_bp.route('/knowledge', methods=['GET'])
@auth_required
def list_knowledge():
    """List knowledge base entries"""
    user = get_current_user()

    query = KnowledgeBase.query.filter_by(organization_id=user.organization_id)

    category = request.args.get('category')
    active_only = request.args.get('active_only', 'true').lower() == 'true'

    if category:
        query = query.filter_by(category=category)
    if active_only:
        query = query.filter_by(is_active=True)

    entries = query.order_by(KnowledgeBase.created_at.desc()).all()

    return jsonify({
        'entries': [{
            'id': e.id,
            'category': e.category,
            'title': e.title,
            'content': json.loads(e.content) if e.content else {},
            'tags': json.loads(e.tags) if e.tags else [],
            'is_active': e.is_active,
            'created_at': e.created_at.isoformat() if e.created_at else None,
            'updated_at': e.updated_at.isoformat() if e.updated_at else None
        } for e in entries]
    })

@governance_bp.route('/knowledge', methods=['POST'])
@auth_required
def create_knowledge():
    """Create a knowledge base entry"""
    user = get_current_user()
    data = request.json

    if not data.get('title'):
        return jsonify({'error': 'Title is required'}), 400

    entry = KnowledgeBase(
        organization_id=user.organization_id,
        category=data.get('category', 'context'),
        title=data['title'],
        content=json.dumps(data.get('content', {})),
        tags=json.dumps(data.get('tags', [])),
        is_active=data.get('is_active', True)
    )
    db.session.add(entry)
    db.session.commit()

    log_audit('create', 'knowledge', entry.id, f"Created knowledge entry: {entry.title}")

    return jsonify({
        'message': 'Knowledge entry created',
        'entry': {'id': entry.id, 'title': entry.title}
    }), 201

@governance_bp.route('/knowledge/<int:id>', methods=['PUT'])
@auth_required
def update_knowledge(id):
    """Update a knowledge base entry"""
    user = get_current_user()
    entry = KnowledgeBase.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not entry:
        return jsonify({'error': 'Knowledge entry not found'}), 404

    data = request.json

    if 'title' in data:
        entry.title = data['title']
    if 'category' in data:
        entry.category = data['category']
    if 'content' in data:
        entry.content = json.dumps(data['content'])
    if 'tags' in data:
        entry.tags = json.dumps(data['tags'])
    if 'is_active' in data:
        entry.is_active = data['is_active']

    db.session.commit()

    log_audit('update', 'knowledge', entry.id, f"Updated knowledge entry: {entry.title}")

    return jsonify({'message': 'Knowledge entry updated'})

@governance_bp.route('/knowledge/<int:id>', methods=['DELETE'])
@auth_required
def delete_knowledge(id):
    """Delete a knowledge base entry"""
    user = get_current_user()
    entry = KnowledgeBase.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not entry:
        return jsonify({'error': 'Knowledge entry not found'}), 404

    title = entry.title
    db.session.delete(entry)
    db.session.commit()

    log_audit('delete', 'knowledge', id, f"Deleted knowledge entry: {title}")

    return jsonify({'message': 'Knowledge entry deleted'})

# ============== Pattern Library Endpoints ==============

@governance_bp.route('/patterns', methods=['GET'])
@auth_required
def list_patterns():
    """List patterns from the pattern library"""
    user = get_current_user()

    query = Pattern.query.filter_by(organization_id=user.organization_id)

    pattern_type = request.args.get('type')
    platform = request.args.get('platform')
    active_only = request.args.get('active_only', 'true').lower() == 'true'

    if pattern_type:
        query = query.filter_by(type=pattern_type)
    if platform:
        query = query.filter_by(platform=platform)
    if active_only:
        query = query.filter_by(is_active=True)

    patterns = query.order_by(Pattern.created_at.desc()).all()

    return jsonify({
        'patterns': [{
            'id': p.id,
            'name': p.name,
            'type': p.type,
            'platform': p.platform,
            'pattern': json.loads(p.pattern) if p.pattern else {},
            'description': p.description,
            'rationale': p.rationale,
            'is_active': p.is_active,
            'created_at': p.created_at.isoformat() if p.created_at else None,
            'updated_at': p.updated_at.isoformat() if p.updated_at else None
        } for p in patterns]
    })

@governance_bp.route('/patterns', methods=['POST'])
@auth_required
def create_pattern():
    """Create a new pattern"""
    user = get_current_user()
    data = request.json

    if not data.get('name'):
        return jsonify({'error': 'Pattern name is required'}), 400

    pattern = Pattern(
        organization_id=user.organization_id,
        name=data['name'],
        type=data.get('type', 'recommended'),
        platform=data.get('platform', 'salesforce'),
        pattern=json.dumps(data.get('pattern', {})),
        description=data.get('description', ''),
        rationale=data.get('rationale', ''),
        is_active=data.get('is_active', True)
    )
    db.session.add(pattern)
    db.session.commit()

    log_audit('create', 'pattern', pattern.id, f"Created pattern: {pattern.name}")

    return jsonify({
        'message': 'Pattern created',
        'pattern': {'id': pattern.id, 'name': pattern.name}
    }), 201

@governance_bp.route('/patterns/<int:id>', methods=['PUT'])
@auth_required
def update_pattern(id):
    """Update a pattern"""
    user = get_current_user()
    pattern = Pattern.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not pattern:
        return jsonify({'error': 'Pattern not found'}), 404

    data = request.json

    if 'name' in data:
        pattern.name = data['name']
    if 'type' in data:
        pattern.type = data['type']
    if 'platform' in data:
        pattern.platform = data['platform']
    if 'pattern' in data:
        pattern.pattern = json.dumps(data['pattern'])
    if 'description' in data:
        pattern.description = data['description']
    if 'rationale' in data:
        pattern.rationale = data['rationale']
    if 'is_active' in data:
        pattern.is_active = data['is_active']

    db.session.commit()

    log_audit('update', 'pattern', pattern.id, f"Updated pattern: {pattern.name}")

    return jsonify({'message': 'Pattern updated'})

@governance_bp.route('/patterns/<int:id>', methods=['DELETE'])
@auth_required
def delete_pattern(id):
    """Delete a pattern"""
    user = get_current_user()
    pattern = Pattern.query.filter_by(
        id=id, organization_id=user.organization_id
    ).first()

    if not pattern:
        return jsonify({'error': 'Pattern not found'}), 404

    name = pattern.name
    db.session.delete(pattern)
    db.session.commit()

    log_audit('delete', 'pattern', id, f"Deleted pattern: {name}")

    return jsonify({'message': 'Pattern deleted'})

# ============== Dashboard Stats ==============

@governance_bp.route('/stats', methods=['GET'])
@auth_required
def get_governance_stats():
    """Get governance dashboard statistics"""
    user = get_current_user()
    org_id = user.organization_id

    # Count policies
    total_policies = GovernancePolicy.query.filter_by(organization_id=org_id).count()
    active_policies = GovernancePolicy.query.filter_by(organization_id=org_id, is_active=True).count()

    # Count approvals
    pending_approvals = ApprovalRequest.query.filter_by(
        organization_id=org_id, status='pending'
    ).count()
    total_approvals = ApprovalRequest.query.filter_by(organization_id=org_id).count()

    # Count knowledge entries
    knowledge_entries = KnowledgeBase.query.filter_by(organization_id=org_id, is_active=True).count()

    # Count patterns
    total_patterns = Pattern.query.filter_by(organization_id=org_id).count()
    by_type = {}
    for t in ['allowed', 'forbidden', 'recommended', 'deprecated']:
        by_type[t] = Pattern.query.filter_by(organization_id=org_id, type=t, is_active=True).count()

    # Recent audit activity
    recent_logs = AuditLog.query.filter_by(organization_id=org_id).count()

    return jsonify({
        'policies': {
            'total': total_policies,
            'active': active_policies
        },
        'approvals': {
            'pending': pending_approvals,
            'total': total_approvals
        },
        'knowledge': {
            'entries': knowledge_entries
        },
        'patterns': {
            'total': total_patterns,
            'by_type': by_type
        },
        'audit': {
            'total_logs': recent_logs
        }
    })


# ============== Harbyx Integration Endpoints ==============

@governance_bp.route('/harbyx/health', methods=['GET'])
def harbyx_health():
    """Check Harbyx connectivity status"""
    if not HARBYX_AVAILABLE:
        return jsonify({
            'status': 'unavailable',
            'message': 'Harbyx client not installed',
            'connected': False
        }), 503

    try:
        client = get_harbyx_client()
        health = client.health_check()
        return jsonify({
            'status': 'ok' if health['connected'] else 'disconnected',
            'harbyx': health,
            'agent_id': 'salesforce-consultant-agent',
            'timestamp': datetime.utcnow().isoformat()
        })
    except HarbyxError as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'connected': False
        }), 500


@governance_bp.route('/harbyx/sync', methods=['POST'])
@auth_required
def harbyx_sync_policies():
    """Sync local Salesforce policies to Harbyx"""
    if not HARBYX_AVAILABLE:
        return jsonify({'error': 'Harbyx client not available'}), 503

    try:
        result = sync_policies_to_harbyx()
        log_audit('sync', 'harbyx', description=f"Synced {len(result['created'])} policies to Harbyx")
        return jsonify(result)
    except HarbyxError as e:
        return jsonify({'error': str(e)}), 500


@governance_bp.route('/harbyx/evaluate', methods=['POST'])
@auth_required
def harbyx_evaluate_action():
    """Evaluate an action against Harbyx policies"""
    if not HARBYX_AVAILABLE:
        return jsonify({'error': 'Harbyx client not available'}), 503

    data = request.json
    action_type = data.get('action_type')
    target = data.get('target')

    if not action_type or not target:
        return jsonify({'error': 'action_type and target are required'}), 400

    try:
        client = get_harbyx_client()
        user = get_current_user()

        result = client.evaluate_action(
            action_type=action_type,
            target=target,
            params=data.get('params', {}),
            metadata={
                'user_id': user.id,
                'user_email': user.email,
                'organization_id': user.organization_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

        # Log the evaluation
        log_audit('evaluate', 'harbyx', description=f"Evaluated {action_type}:{target} -> {result.get('decision')}")

        # If requires approval, create local approval request as well
        if result.get('decision') == 'require_approval' and result.get('approval_id'):
            approval = ApprovalRequest(
                organization_id=user.organization_id,
                user_id=user.id,
                action_type=action_type,
                target=target,
                context=json.dumps({
                    'harbyx_approval_id': result.get('approval_id'),
                    'params': data.get('params', {}),
                    'reason': result.get('reason')
                }),
                status='pending'
            )
            db.session.add(approval)
            db.session.commit()
            result['local_approval_id'] = approval.id

        return jsonify(result)

    except HarbyxError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code or 500


@governance_bp.route('/harbyx/approvals', methods=['GET'])
@auth_required
def harbyx_list_approvals():
    """List pending approvals from Harbyx"""
    if not HARBYX_AVAILABLE:
        return jsonify({'error': 'Harbyx client not available'}), 503

    try:
        client = get_harbyx_client()
        approvals = client.get_pending_approvals()
        return jsonify({
            'approvals': approvals,
            'count': len(approvals) if isinstance(approvals, list) else 0,
            'source': 'harbyx'
        })
    except HarbyxError as e:
        return jsonify({'error': str(e)}), 500


@governance_bp.route('/harbyx/approvals/<approval_id>/decide', methods=['POST'])
@auth_required
def harbyx_decide_approval(approval_id):
    """Approve or reject an action in Harbyx"""
    if not HARBYX_AVAILABLE:
        return jsonify({'error': 'Harbyx client not available'}), 503

    data = request.json
    decision = data.get('decision')
    reason = data.get('reason', '')

    if decision not in ('approve', 'reject'):
        return jsonify({'error': 'Decision must be approve or reject'}), 400

    try:
        client = get_harbyx_client()
        user = get_current_user()

        result = client.decide_approval(approval_id, decision, reason)

        # Update local approval if exists
        local_approval = ApprovalRequest.query.filter(
            ApprovalRequest.context.contains(approval_id)
        ).first()

        if local_approval:
            local_approval.status = 'approved' if decision == 'approve' else 'rejected'
            local_approval.decision_by = user.id
            local_approval.decision_reason = reason
            local_approval.decided_at = datetime.utcnow()
            db.session.commit()

        log_audit(decision, 'harbyx_approval', approval_id,
                  f"{decision.capitalize()} Harbyx approval: {approval_id}")

        return jsonify(result)

    except HarbyxError as e:
        return jsonify({'error': str(e)}), e.status_code or 500


@governance_bp.route('/harbyx/policies', methods=['GET'])
@auth_required
def harbyx_list_policies():
    """List all policies from Harbyx"""
    if not HARBYX_AVAILABLE:
        return jsonify({'error': 'Harbyx client not available'}), 503

    try:
        client = get_harbyx_client()
        policies = client.list_policies()
        return jsonify({
            'policies': policies,
            'count': len(policies) if isinstance(policies, list) else 0,
            'source': 'harbyx'
        })
    except HarbyxError as e:
        return jsonify({'error': str(e)}), 500


@governance_bp.route('/audit', methods=['POST'])
def create_audit_from_webhook():
    """Create audit log entry from webhook (no auth required for webhooks)"""
    data = request.json

    # Basic validation
    if not data.get('entity_type') or not data.get('action'):
        return jsonify({'error': 'entity_type and action are required'}), 400

    try:
        log = AuditLog(
            organization_id=data.get('organization_id', 1),  # Default org for webhooks
            user_id=data.get('user_id'),
            action=data['action'],
            entity_type=data['entity_type'],
            entity_id=data.get('entity_id'),
            description=data.get('description'),
            metadata=json.dumps(data.get('metadata', {})),
            ip_address=request.remote_addr
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({'message': 'Audit log created', 'id': log.id}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500
