"""
Flask Backend for 3D Game AI Presentation Platform
Provides APIs for: secrets vault, tasks, team, activity, milestones, decisions, resources
"""

import os
import json
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import jwt

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://5.249.161.66:3000"])

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Secrets vault password (should be set via environment variable)
VAULT_PASSWORD = os.environ.get('VAULT_PASSWORD', 'admin123')

db = SQLAlchemy(app)

# ============== Models ==============

class Secret(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # api_key, token, credential, env, other
    value = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Task(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    component = db.Column(db.String(50), nullable=False)  # stt, rag, tts-lipsync, mcp
    phase = db.Column(db.String(100))
    status = db.Column(db.String(20), default='todo')  # todo, in_progress, done
    priority = db.Column(db.String(20), default='medium')  # high, medium, low
    assignee = db.Column(db.String(100))
    notes = db.Column(db.Text)
    due_date = db.Column(db.DateTime)
    time_spent = db.Column(db.Integer, default=0)  # minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TeamMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100))
    github = db.Column(db.String(100))
    avatar_url = db.Column(db.String(500))
    components = db.Column(db.Text)  # JSON array of component IDs
    status = db.Column(db.String(20), default='active')  # active, away, offline
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)  # task_update, comment, commit, milestone, decision
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    component = db.Column(db.String(50))
    user = db.Column(db.String(100))
    extra_data = db.Column(db.Text)  # JSON for extra data (renamed from metadata which is reserved)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Milestone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    component = db.Column(db.String(50))
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, completed, delayed
    target_date = db.Column(db.DateTime)
    completed_date = db.Column(db.DateTime)
    progress = db.Column(db.Integer, default=0)  # 0-100
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Decision(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='proposed')  # proposed, accepted, rejected, superseded
    context = db.Column(db.Text)
    decision = db.Column(db.Text)
    consequences = db.Column(db.Text)
    component = db.Column(db.String(50))
    author = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # paper, tutorial, tool, library, docs
    description = db.Column(db.Text)
    component = db.Column(db.String(50))
    tags = db.Column(db.Text)  # JSON array
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Changelog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    changes = db.Column(db.Text)  # JSON array of changes
    component = db.Column(db.String(50))
    release_date = db.Column(db.DateTime, default=datetime.utcnow)
    author = db.Column(db.String(100))

class GlossaryTerm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    term = db.Column(db.String(100), nullable=False, unique=True)
    definition = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))  # ml, audio, graphics, general
    related_terms = db.Column(db.Text)  # JSON array
    component = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ============== Auth Helpers ==============

def generate_vault_token():
    """Generate a JWT token for vault access"""
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=4),
        'iat': datetime.utcnow(),
        'type': 'vault_access'
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def vault_auth_required(f):
    """Decorator to require vault authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

# ============== Secrets Vault Routes ==============

@app.route('/api/vault/auth', methods=['POST'])
def vault_authenticate():
    """Authenticate to access secrets vault"""
    data = request.json
    password = data.get('password', '')

    if password == VAULT_PASSWORD:
        token = generate_vault_token()
        log_activity('vault_access', 'Vault accessed', 'Someone authenticated to the secrets vault')
        return jsonify({'token': token, 'expires_in': 14400})  # 4 hours

    return jsonify({'error': 'Invalid password'}), 401

@app.route('/api/vault/secrets', methods=['GET'])
@vault_auth_required
def get_secrets():
    """Get all secrets (values are masked unless specifically requested)"""
    secrets = Secret.query.all()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'category': s.category,
        'value': mask_secret(s.value),
        'description': s.description,
        'created_at': s.created_at.isoformat(),
        'updated_at': s.updated_at.isoformat()
    } for s in secrets])

@app.route('/api/vault/secrets/<int:secret_id>/reveal', methods=['GET'])
@vault_auth_required
def reveal_secret(secret_id):
    """Reveal the actual value of a secret"""
    secret = Secret.query.get_or_404(secret_id)
    log_activity('secret_revealed', f'Secret revealed: {secret.name}', f'The secret "{secret.name}" was revealed')
    return jsonify({'value': secret.value})

@app.route('/api/vault/secrets', methods=['POST'])
@vault_auth_required
def create_secret():
    """Create a new secret"""
    data = request.json
    secret = Secret(
        name=data['name'],
        category=data['category'],
        value=data['value'],
        description=data.get('description', '')
    )
    db.session.add(secret)
    db.session.commit()
    log_activity('secret_created', f'Secret created: {secret.name}', f'A new secret "{secret.name}" was added')
    return jsonify({'id': secret.id, 'message': 'Secret created'}), 201

@app.route('/api/vault/secrets/<int:secret_id>', methods=['PUT'])
@vault_auth_required
def update_secret(secret_id):
    """Update a secret"""
    secret = Secret.query.get_or_404(secret_id)
    data = request.json

    if 'name' in data:
        secret.name = data['name']
    if 'category' in data:
        secret.category = data['category']
    if 'value' in data:
        secret.value = data['value']
    if 'description' in data:
        secret.description = data['description']

    db.session.commit()
    log_activity('secret_updated', f'Secret updated: {secret.name}', f'The secret "{secret.name}" was modified')
    return jsonify({'message': 'Secret updated'})

@app.route('/api/vault/secrets/<int:secret_id>', methods=['DELETE'])
@vault_auth_required
def delete_secret(secret_id):
    """Delete a secret"""
    secret = Secret.query.get_or_404(secret_id)
    name = secret.name
    db.session.delete(secret)
    db.session.commit()
    log_activity('secret_deleted', f'Secret deleted: {name}', f'The secret "{name}" was removed')
    return jsonify({'message': 'Secret deleted'})

def mask_secret(value):
    """Mask a secret value for display"""
    if len(value) <= 8:
        return '*' * len(value)
    return value[:4] + '*' * (len(value) - 8) + value[-4:]

# ============== Tasks Routes ==============

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks with optional filters"""
    component = request.args.get('component')
    status = request.args.get('status')

    query = Task.query
    if component:
        query = query.filter_by(component=component)
    if status:
        query = query.filter_by(status=status)

    tasks = query.all()
    return jsonify([task_to_dict(t) for t in tasks])

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get a specific task"""
    task = Task.query.get_or_404(task_id)
    return jsonify(task_to_dict(task))

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new task"""
    data = request.json
    task = Task(
        id=data['id'],
        title=data['title'],
        description=data.get('description', ''),
        component=data['component'],
        phase=data.get('phase', ''),
        status=data.get('status', 'todo'),
        priority=data.get('priority', 'medium'),
        assignee=data.get('assignee'),
        notes=data.get('notes'),
        due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None
    )
    db.session.add(task)
    db.session.commit()
    log_activity('task_created', f'Task created: {task.title}', task.description, task.component)
    return jsonify(task_to_dict(task)), 201

@app.route('/api/tasks/<task_id>', methods=['PUT'])
def update_task(task_id):
    """Update a task"""
    task = Task.query.get_or_404(task_id)
    data = request.json

    old_status = task.status

    for field in ['title', 'description', 'component', 'phase', 'status', 'priority', 'assignee', 'notes', 'time_spent']:
        if field in data:
            setattr(task, field, data[field])

    if 'due_date' in data:
        task.due_date = datetime.fromisoformat(data['due_date']) if data['due_date'] else None

    db.session.commit()

    if old_status != task.status:
        log_activity('task_status_changed', f'Task status: {task.title}',
                    f'Changed from {old_status} to {task.status}', task.component)

    return jsonify(task_to_dict(task))

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task"""
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return jsonify({'message': 'Task deleted'})

@app.route('/api/tasks/bulk', methods=['POST'])
def bulk_create_tasks():
    """Bulk create tasks (for initial seeding)"""
    data = request.json
    tasks = data.get('tasks', [])

    for task_data in tasks:
        existing = Task.query.get(task_data['id'])
        if not existing:
            task = Task(
                id=task_data['id'],
                title=task_data['title'],
                description=task_data.get('description', ''),
                component=task_data['component'],
                phase=task_data.get('phase', ''),
                status=task_data.get('status', 'todo'),
                priority=task_data.get('priority', 'medium')
            )
            db.session.add(task)

    db.session.commit()
    return jsonify({'message': f'Created {len(tasks)} tasks'})

def task_to_dict(task):
    return {
        'id': task.id,
        'title': task.title,
        'description': task.description,
        'component': task.component,
        'phase': task.phase,
        'status': task.status,
        'priority': task.priority,
        'assignee': task.assignee,
        'notes': task.notes,
        'due_date': task.due_date.isoformat() if task.due_date else None,
        'time_spent': task.time_spent,
        'created_at': task.created_at.isoformat(),
        'updated_at': task.updated_at.isoformat()
    }

# ============== Team Routes ==============

@app.route('/api/team', methods=['GET'])
def get_team():
    """Get all team members"""
    members = TeamMember.query.all()
    return jsonify([{
        'id': m.id,
        'name': m.name,
        'role': m.role,
        'email': m.email,
        'github': m.github,
        'avatar_url': m.avatar_url,
        'components': json.loads(m.components) if m.components else [],
        'status': m.status,
        'bio': m.bio,
        'created_at': m.created_at.isoformat()
    } for m in members])

@app.route('/api/team', methods=['POST'])
def create_team_member():
    """Create a new team member"""
    data = request.json
    member = TeamMember(
        name=data['name'],
        role=data['role'],
        email=data.get('email'),
        github=data.get('github'),
        avatar_url=data.get('avatar_url'),
        components=json.dumps(data.get('components', [])),
        status=data.get('status', 'active'),
        bio=data.get('bio')
    )
    db.session.add(member)
    db.session.commit()
    log_activity('team_member_added', f'Team member added: {member.name}', f'{member.name} joined as {member.role}')
    return jsonify({'id': member.id, 'message': 'Team member created'}), 201

@app.route('/api/team/<int:member_id>', methods=['PUT'])
def update_team_member(member_id):
    """Update a team member"""
    member = TeamMember.query.get_or_404(member_id)
    data = request.json

    for field in ['name', 'role', 'email', 'github', 'avatar_url', 'status', 'bio']:
        if field in data:
            setattr(member, field, data[field])

    if 'components' in data:
        member.components = json.dumps(data['components'])

    db.session.commit()
    return jsonify({'message': 'Team member updated'})

@app.route('/api/team/<int:member_id>', methods=['DELETE'])
def delete_team_member(member_id):
    """Delete a team member"""
    member = TeamMember.query.get_or_404(member_id)
    db.session.delete(member)
    db.session.commit()
    return jsonify({'message': 'Team member deleted'})

# ============== Activity Routes ==============

@app.route('/api/activity', methods=['GET'])
def get_activity():
    """Get activity feed with optional filters"""
    limit = request.args.get('limit', 50, type=int)
    component = request.args.get('component')
    activity_type = request.args.get('type')

    query = Activity.query.order_by(Activity.created_at.desc())

    if component:
        query = query.filter_by(component=component)
    if activity_type:
        query = query.filter_by(type=activity_type)

    activities = query.limit(limit).all()
    return jsonify([{
        'id': a.id,
        'type': a.type,
        'title': a.title,
        'description': a.description,
        'component': a.component,
        'user': a.user,
        'metadata': json.loads(a.extra_data) if a.extra_data else None,
        'created_at': a.created_at.isoformat()
    } for a in activities])

@app.route('/api/activity', methods=['POST'])
def create_activity():
    """Create a new activity entry"""
    data = request.json
    activity = Activity(
        type=data['type'],
        title=data['title'],
        description=data.get('description'),
        component=data.get('component'),
        user=data.get('user'),
        extra_data=json.dumps(data.get('metadata')) if data.get('metadata') else None
    )
    db.session.add(activity)
    db.session.commit()
    return jsonify({'id': activity.id}), 201

def log_activity(activity_type, title, description=None, component=None, user=None, metadata=None):
    """Helper to log activities"""
    activity = Activity(
        type=activity_type,
        title=title,
        description=description,
        component=component,
        user=user,
        extra_data=json.dumps(metadata) if metadata else None
    )
    db.session.add(activity)
    db.session.commit()

# ============== Milestones Routes ==============

@app.route('/api/milestones', methods=['GET'])
def get_milestones():
    """Get all milestones"""
    component = request.args.get('component')
    query = Milestone.query.order_by(Milestone.target_date)

    if component:
        query = query.filter_by(component=component)

    milestones = query.all()
    return jsonify([{
        'id': m.id,
        'title': m.title,
        'description': m.description,
        'component': m.component,
        'status': m.status,
        'target_date': m.target_date.isoformat() if m.target_date else None,
        'completed_date': m.completed_date.isoformat() if m.completed_date else None,
        'progress': m.progress,
        'created_at': m.created_at.isoformat()
    } for m in milestones])

@app.route('/api/milestones', methods=['POST'])
def create_milestone():
    """Create a new milestone"""
    data = request.json
    milestone = Milestone(
        title=data['title'],
        description=data.get('description'),
        component=data.get('component'),
        status=data.get('status', 'pending'),
        target_date=datetime.fromisoformat(data['target_date']) if data.get('target_date') else None,
        progress=data.get('progress', 0)
    )
    db.session.add(milestone)
    db.session.commit()
    log_activity('milestone_created', f'Milestone created: {milestone.title}', milestone.description, milestone.component)
    return jsonify({'id': milestone.id}), 201

@app.route('/api/milestones/<int:milestone_id>', methods=['PUT'])
def update_milestone(milestone_id):
    """Update a milestone"""
    milestone = Milestone.query.get_or_404(milestone_id)
    data = request.json

    old_status = milestone.status

    for field in ['title', 'description', 'component', 'status', 'progress']:
        if field in data:
            setattr(milestone, field, data[field])

    if 'target_date' in data:
        milestone.target_date = datetime.fromisoformat(data['target_date']) if data['target_date'] else None

    if data.get('status') == 'completed' and old_status != 'completed':
        milestone.completed_date = datetime.utcnow()
        log_activity('milestone_completed', f'Milestone completed: {milestone.title}', None, milestone.component)

    db.session.commit()
    return jsonify({'message': 'Milestone updated'})

@app.route('/api/milestones/<int:milestone_id>', methods=['DELETE'])
def delete_milestone(milestone_id):
    """Delete a milestone"""
    milestone = Milestone.query.get_or_404(milestone_id)
    db.session.delete(milestone)
    db.session.commit()
    return jsonify({'message': 'Milestone deleted'})

# ============== Decisions (ADR) Routes ==============

@app.route('/api/decisions', methods=['GET'])
def get_decisions():
    """Get all architecture decisions"""
    component = request.args.get('component')
    status = request.args.get('status')

    query = Decision.query.order_by(Decision.created_at.desc())

    if component:
        query = query.filter_by(component=component)
    if status:
        query = query.filter_by(status=status)

    decisions = query.all()
    return jsonify([{
        'id': d.id,
        'title': d.title,
        'status': d.status,
        'context': d.context,
        'decision': d.decision,
        'consequences': d.consequences,
        'component': d.component,
        'author': d.author,
        'created_at': d.created_at.isoformat(),
        'updated_at': d.updated_at.isoformat()
    } for d in decisions])

@app.route('/api/decisions', methods=['POST'])
def create_decision():
    """Create a new architecture decision"""
    data = request.json
    decision = Decision(
        title=data['title'],
        status=data.get('status', 'proposed'),
        context=data.get('context'),
        decision=data.get('decision'),
        consequences=data.get('consequences'),
        component=data.get('component'),
        author=data.get('author')
    )
    db.session.add(decision)
    db.session.commit()
    log_activity('decision_created', f'ADR created: {decision.title}', decision.context, decision.component, decision.author)
    return jsonify({'id': decision.id}), 201

@app.route('/api/decisions/<int:decision_id>', methods=['PUT'])
def update_decision(decision_id):
    """Update an architecture decision"""
    decision = Decision.query.get_or_404(decision_id)
    data = request.json

    for field in ['title', 'status', 'context', 'decision', 'consequences', 'component', 'author']:
        if field in data:
            setattr(decision, field, data[field])

    db.session.commit()
    return jsonify({'message': 'Decision updated'})

@app.route('/api/decisions/<int:decision_id>', methods=['DELETE'])
def delete_decision(decision_id):
    """Delete an architecture decision"""
    decision = Decision.query.get_or_404(decision_id)
    db.session.delete(decision)
    db.session.commit()
    return jsonify({'message': 'Decision deleted'})

# ============== Resources Routes ==============

@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get all resources"""
    category = request.args.get('category')
    component = request.args.get('component')

    query = Resource.query.order_by(Resource.created_at.desc())

    if category:
        query = query.filter_by(category=category)
    if component:
        query = query.filter_by(component=component)

    resources = query.all()
    return jsonify([{
        'id': r.id,
        'title': r.title,
        'url': r.url,
        'category': r.category,
        'description': r.description,
        'component': r.component,
        'tags': json.loads(r.tags) if r.tags else [],
        'created_at': r.created_at.isoformat()
    } for r in resources])

@app.route('/api/resources', methods=['POST'])
def create_resource():
    """Create a new resource"""
    data = request.json
    resource = Resource(
        title=data['title'],
        url=data['url'],
        category=data['category'],
        description=data.get('description'),
        component=data.get('component'),
        tags=json.dumps(data.get('tags', []))
    )
    db.session.add(resource)
    db.session.commit()
    return jsonify({'id': resource.id}), 201

@app.route('/api/resources/<int:resource_id>', methods=['DELETE'])
def delete_resource(resource_id):
    """Delete a resource"""
    resource = Resource.query.get_or_404(resource_id)
    db.session.delete(resource)
    db.session.commit()
    return jsonify({'message': 'Resource deleted'})

# ============== Changelog Routes ==============

@app.route('/api/changelog', methods=['GET'])
def get_changelog():
    """Get changelog entries"""
    component = request.args.get('component')

    query = Changelog.query.order_by(Changelog.release_date.desc())

    if component:
        query = query.filter_by(component=component)

    entries = query.all()
    return jsonify([{
        'id': e.id,
        'version': e.version,
        'title': e.title,
        'description': e.description,
        'changes': json.loads(e.changes) if e.changes else [],
        'component': e.component,
        'release_date': e.release_date.isoformat(),
        'author': e.author
    } for e in entries])

@app.route('/api/changelog', methods=['POST'])
def create_changelog():
    """Create a new changelog entry"""
    data = request.json
    entry = Changelog(
        version=data['version'],
        title=data['title'],
        description=data.get('description'),
        changes=json.dumps(data.get('changes', [])),
        component=data.get('component'),
        release_date=datetime.fromisoformat(data['release_date']) if data.get('release_date') else datetime.utcnow(),
        author=data.get('author')
    )
    db.session.add(entry)
    db.session.commit()
    log_activity('changelog_added', f'Changelog: {entry.version} - {entry.title}', entry.description, entry.component, entry.author)
    return jsonify({'id': entry.id}), 201

# ============== Glossary Routes ==============

@app.route('/api/glossary', methods=['GET'])
def get_glossary():
    """Get all glossary terms"""
    category = request.args.get('category')
    component = request.args.get('component')

    query = GlossaryTerm.query.order_by(GlossaryTerm.term)

    if category:
        query = query.filter_by(category=category)
    if component:
        query = query.filter_by(component=component)

    terms = query.all()
    return jsonify([{
        'id': t.id,
        'term': t.term,
        'definition': t.definition,
        'category': t.category,
        'related_terms': json.loads(t.related_terms) if t.related_terms else [],
        'component': t.component,
        'created_at': t.created_at.isoformat()
    } for t in terms])

@app.route('/api/glossary', methods=['POST'])
def create_glossary_term():
    """Create a new glossary term"""
    data = request.json
    term = GlossaryTerm(
        term=data['term'],
        definition=data['definition'],
        category=data.get('category'),
        related_terms=json.dumps(data.get('related_terms', [])),
        component=data.get('component')
    )
    db.session.add(term)
    db.session.commit()
    return jsonify({'id': term.id}), 201

@app.route('/api/glossary/<int:term_id>', methods=['DELETE'])
def delete_glossary_term(term_id):
    """Delete a glossary term"""
    term = GlossaryTerm.query.get_or_404(term_id)
    db.session.delete(term)
    db.session.commit()
    return jsonify({'message': 'Term deleted'})

# ============== LLMs.txt Route ==============

@app.route('/llms.txt', methods=['GET'])
def get_llms_txt():
    """Generate machine-readable context for LLMs"""

    # Get current stats
    total_tasks = Task.query.count()
    completed_tasks = Task.query.filter_by(status='done').count()
    in_progress_tasks = Task.query.filter_by(status='in_progress').count()

    milestones = Milestone.query.filter(Milestone.status != 'completed').order_by(Milestone.target_date).limit(5).all()
    recent_decisions = Decision.query.filter_by(status='accepted').order_by(Decision.created_at.desc()).limit(5).all()

    content = f"""# 3D Game Generation AI Assistant - LLM Context

## Project Overview
A voice-controlled AI system for creating 3D game assets using natural language and advanced AI technologies.
This project combines Speech-to-Text, RAG retrieval, Text-to-Speech, and 3D asset generation.

## Current Status
- Total Tasks: {total_tasks}
- Completed: {completed_tasks} ({round(completed_tasks/total_tasks*100) if total_tasks > 0 else 0}%)
- In Progress: {in_progress_tasks}
- Target Deadline: December 18, 2025

## Components

### 1. VoxFormer STT (Speech-to-Text)
Custom Transformer architecture with Conformer blocks, RoPE embeddings, and CTC loss.
Technologies: PyTorch, Conformer, RoPE, CTC, SwiGLU
Timeline: 12 weeks (6 phases)
Documentation: /docs/technical/STT_ARCHITECTURE_PLAN.md

### 2. Advanced RAG (Retrieval-Augmented Generation)
Hybrid retrieval with BGE-M3 embeddings, HNSW indexing, BM25, and cross-encoder reranking.
Technologies: PostgreSQL, pgvector, BGE-M3, HNSW, BM25, MiniLM
Timeline: 16 weeks (4 phases)
Documentation: /docs/technical/RAG_ARCHITECTURE_PLAN.md

### 3. TTS + LipSync (Text-to-Speech with Lip Synchronization)
ElevenLabs TTS integration with Wav2Lip/SadTalker lip-sync and game engine bridging.
Technologies: ElevenLabs, Wav2Lip, SadTalker, WebSocket, Unity, UE5
Timeline: 5 weeks (5 phases)
Documentation: /docs/technical/TTS_LIPSYNC_ARCHITECTURE_PLAN.md

### 4. Blender MCP (Model Context Protocol)
AI-driven 3D asset creation with Sketchfab, Poly Haven, and Hyper3D Rodin integration.
Technologies: MCP, Blender, Sketchfab, Poly Haven, Hyper3D, Socket
Timeline: 4 weeks (4 phases)
Documentation: /docs/technical/BLENDER_MCP_ARCHITECTURE_PLAN.md

## Upcoming Milestones
"""

    for m in milestones:
        content += f"- [{m.status.upper()}] {m.title}"
        if m.target_date:
            content += f" (Target: {m.target_date.strftime('%Y-%m-%d')})"
        content += "\n"

    content += "\n## Recent Architecture Decisions\n"

    for d in recent_decisions:
        content += f"- ADR-{d.id}: {d.title}\n"
        if d.decision:
            content += f"  Decision: {d.decision[:200]}...\n" if len(d.decision or '') > 200 else f"  Decision: {d.decision}\n"

    content += """
## Project Structure
```
/src/app/           - Next.js App Router routes
/src/components/    - React components (slides, UI)
/docs/technical/    - Architecture specifications
/backend/           - Flask API backend
```

## API Endpoints
- GET /api/tasks - Task management
- GET /api/team - Team directory
- GET /api/activity - Activity feed
- GET /api/milestones - Project milestones
- GET /api/decisions - Architecture decisions
- GET /api/resources - Curated resources
- GET /api/glossary - Technical glossary
- GET /api/changelog - Version history

## Getting Started
1. Frontend: `npm run dev` (port 3000)
2. Backend: `python backend/app.py` (port 5000)

## Contact
For collaboration, check the /team page for team members and their responsibilities.
"""

    return Response(content, mimetype='text/plain')

# ============== Context API Route ==============

@app.route('/api/context', methods=['GET'])
def get_context():
    """Get full project context for LLM consumption"""

    tasks = Task.query.all()
    milestones = Milestone.query.all()
    decisions = Decision.query.filter_by(status='accepted').all()

    components = {
        'stt': {
            'name': 'VoxFormer STT',
            'description': 'Custom Speech-to-Text Transformer with Conformer blocks, RoPE embeddings, and CTC loss',
            'technologies': ['PyTorch', 'Conformer', 'RoPE', 'CTC', 'SwiGLU'],
            'doc_path': '/docs/technical/STT_ARCHITECTURE_PLAN.md',
            'presentation': '/technical',
            'timeline': '12 weeks'
        },
        'rag': {
            'name': 'Advanced RAG',
            'description': 'Hybrid retrieval with BGE-M3 embeddings, HNSW indexing, and cross-encoder reranking',
            'technologies': ['PostgreSQL', 'pgvector', 'BGE-M3', 'HNSW', 'BM25', 'MiniLM'],
            'doc_path': '/docs/technical/RAG_ARCHITECTURE_PLAN.md',
            'presentation': '/rag',
            'timeline': '16 weeks'
        },
        'tts-lipsync': {
            'name': 'TTS + LipSync',
            'description': 'ElevenLabs TTS with Wav2Lip/SadTalker lip-sync and game engine bridging',
            'technologies': ['ElevenLabs', 'Wav2Lip', 'SadTalker', 'WebSocket', 'Unity', 'UE5'],
            'doc_path': '/docs/technical/TTS_LIPSYNC_ARCHITECTURE_PLAN.md',
            'presentation': '/avatar',
            'timeline': '5 weeks'
        },
        'mcp': {
            'name': 'Blender MCP',
            'description': 'AI-driven 3D asset creation with Sketchfab, Poly Haven, and Hyper3D Rodin',
            'technologies': ['MCP', 'Blender', 'Sketchfab', 'Poly Haven', 'Hyper3D', 'Socket'],
            'doc_path': '/docs/technical/BLENDER_MCP_ARCHITECTURE_PLAN.md',
            'presentation': '/mcp',
            'timeline': '4 weeks'
        }
    }

    # Calculate stats per component
    for comp_id in components:
        comp_tasks = [t for t in tasks if t.component == comp_id]
        completed = len([t for t in comp_tasks if t.status == 'done'])
        total = len(comp_tasks)
        components[comp_id]['progress'] = round(completed / total * 100) if total > 0 else 0
        components[comp_id]['tasks'] = {
            'total': total,
            'completed': completed,
            'in_progress': len([t for t in comp_tasks if t.status == 'in_progress'])
        }

    return jsonify({
        'project': {
            'name': '3D Game Generation AI Assistant',
            'description': 'Voice-controlled AI system for creating 3D game assets using natural language',
            'deadline': '2025-12-18',
            'total_tasks': len(tasks),
            'completed_tasks': len([t for t in tasks if t.status == 'done'])
        },
        'components': components,
        'milestones': [{
            'id': m.id,
            'title': m.title,
            'component': m.component,
            'status': m.status,
            'target_date': m.target_date.isoformat() if m.target_date else None,
            'progress': m.progress
        } for m in milestones],
        'decisions': [{
            'id': d.id,
            'title': d.title,
            'component': d.component,
            'decision': d.decision
        } for d in decisions]
    })

# ============== Health Check ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected'
    })

# ============== Seed Data ==============

def seed_initial_data():
    """Seed database with initial data"""

    # Check if already seeded
    if Task.query.count() > 0:
        return

    # Seed initial tasks
    initial_tasks = [
        # STT Tasks
        {"id": "stt-1", "title": "Implement STFT with Hann windowing", "description": "400-sample window, 160-sample hop for audio processing", "component": "stt", "phase": "Phase 1: Core Components", "priority": "high"},
        {"id": "stt-2", "title": "Build Mel filter bank", "description": "80 filters, 80-7600 Hz frequency range", "component": "stt", "phase": "Phase 1: Core Components", "priority": "high"},
        {"id": "stt-3", "title": "Create log-Mel spectrogram pipeline", "description": "Complete audio frontend preprocessing", "component": "stt", "phase": "Phase 1: Core Components", "priority": "high"},
        {"id": "stt-4", "title": "Implement Conv2D subsampling", "description": "4x reduction for efficient processing", "component": "stt", "phase": "Phase 1: Core Components", "priority": "medium"},
        {"id": "stt-5", "title": "Implement RoPE embeddings", "description": "Rotary Position Embeddings for attention", "component": "stt", "phase": "Phase 2: Attention", "priority": "high"},
        {"id": "stt-6", "title": "Build multi-head self-attention", "description": "With RoPE integration", "component": "stt", "phase": "Phase 2: Attention", "priority": "high"},
        {"id": "stt-7", "title": "Implement Conformer convolution module", "description": "31-kernel depthwise conv", "component": "stt", "phase": "Phase 3: Transformer", "priority": "high"},
        {"id": "stt-8", "title": "Build SwiGLU feed-forward network", "description": "Gated linear unit activation", "component": "stt", "phase": "Phase 3: Transformer", "priority": "medium"},
        {"id": "stt-9", "title": "Add CTC projection head", "description": "vocab_size outputs for decoding", "component": "stt", "phase": "Phase 4: Assembly", "priority": "high"},
        {"id": "stt-10", "title": "Implement CTC loss", "description": "Log-space stability for training", "component": "stt", "phase": "Phase 5: Training", "priority": "high"},
        {"id": "stt-11", "title": "Train on LibriSpeech 960h", "description": "Full training run with evaluation", "component": "stt", "phase": "Phase 6: Evaluation", "priority": "high"},

        # RAG Tasks
        {"id": "rag-1", "title": "Set up PostgreSQL with pgvector", "description": "Database with vector extension", "component": "rag", "phase": "Phase 1: Foundation", "priority": "high"},
        {"id": "rag-2", "title": "Design document schema", "description": "With metadata fields for filtering", "component": "rag", "phase": "Phase 1: Foundation", "priority": "high"},
        {"id": "rag-3", "title": "Implement chunking strategies", "description": "Semantic and recursive splitting", "component": "rag", "phase": "Phase 1: Foundation", "priority": "medium"},
        {"id": "rag-4", "title": "Set up BGE-M3 embedding service", "description": "4096-dim dense vectors", "component": "rag", "phase": "Phase 1: Foundation", "priority": "high"},
        {"id": "rag-5", "title": "Implement HNSW index", "description": "Approximate nearest neighbor search", "component": "rag", "phase": "Phase 2: Retrieval", "priority": "high"},
        {"id": "rag-6", "title": "Build BM25 sparse retrieval", "description": "Elasticsearch backend", "component": "rag", "phase": "Phase 2: Retrieval", "priority": "high"},
        {"id": "rag-7", "title": "Implement RRF fusion", "description": "Combine dense + sparse rankings", "component": "rag", "phase": "Phase 2: Retrieval", "priority": "medium"},
        {"id": "rag-8", "title": "Integrate MiniLM reranker", "description": "Cross-encoder for precision", "component": "rag", "phase": "Phase 3: Reranking", "priority": "high"},
        {"id": "rag-9", "title": "Build agentic query transformation", "description": "Query decomposition and expansion", "component": "rag", "phase": "Phase 4: Integration", "priority": "medium"},
        {"id": "rag-10", "title": "Set up RAGAS evaluation", "description": "Quality metrics framework", "component": "rag", "phase": "Phase 4: Integration", "priority": "high"},

        # TTS Tasks
        {"id": "tts-1", "title": "Set up ElevenLabs API", "description": "Authentication and streaming", "component": "tts-lipsync", "phase": "Phase 1: TTS", "priority": "high"},
        {"id": "tts-2", "title": "Implement TTS streaming", "description": "WebSocket audio chunks", "component": "tts-lipsync", "phase": "Phase 1: TTS", "priority": "high"},
        {"id": "tts-3", "title": "Set up Wav2Lip pipeline", "description": "Neural lip-sync inference", "component": "tts-lipsync", "phase": "Phase 2: LipSync", "priority": "high"},
        {"id": "tts-4", "title": "Implement viseme mapping", "description": "22 ARKit-compatible visemes", "component": "tts-lipsync", "phase": "Phase 2: LipSync", "priority": "high"},
        {"id": "tts-5", "title": "Unity C# integration", "description": "Blend shape controller", "component": "tts-lipsync", "phase": "Phase 3: Game Engine", "priority": "high"},
        {"id": "tts-6", "title": "UE5 C++ integration", "description": "Morph target controller", "component": "tts-lipsync", "phase": "Phase 3: Game Engine", "priority": "high"},
        {"id": "tts-7", "title": "Latency optimization", "description": "Target < 200ms end-to-end", "component": "tts-lipsync", "phase": "Phase 4: Testing", "priority": "medium"},

        # MCP Tasks
        {"id": "mcp-1", "title": "Install BlenderMCP addon", "description": "Blender 3.6+ setup", "component": "mcp", "phase": "Phase 1: Setup", "priority": "high"},
        {"id": "mcp-2", "title": "Configure MCP server", "description": "Socket communication on 9876", "component": "mcp", "phase": "Phase 1: Setup", "priority": "high"},
        {"id": "mcp-3", "title": "Test basic commands", "description": "get_scene_info, execute_blender_code", "component": "mcp", "phase": "Phase 1: Setup", "priority": "medium"},
        {"id": "mcp-4", "title": "Configure Sketchfab API", "description": "Model search and download", "component": "mcp", "phase": "Phase 2: Assets", "priority": "high"},
        {"id": "mcp-5", "title": "Configure Poly Haven", "description": "Textures and HDRIs", "component": "mcp", "phase": "Phase 2: Assets", "priority": "medium"},
        {"id": "mcp-6", "title": "Configure Hyper3D Rodin", "description": "AI 3D generation", "component": "mcp", "phase": "Phase 2: Assets", "priority": "low"},
        {"id": "mcp-7", "title": "FBX export for Unity", "description": "Y-up coordinate system", "component": "mcp", "phase": "Phase 3: Export", "priority": "high"},
        {"id": "mcp-8", "title": "FBX export for UE5", "description": "Z-up, Nanite support", "component": "mcp", "phase": "Phase 3: Export", "priority": "high"},
        {"id": "mcp-9", "title": "Security audit", "description": "Code validation and sandboxing", "component": "mcp", "phase": "Phase 4: Production", "priority": "high"},
    ]

    for task_data in initial_tasks:
        task = Task(**task_data, status='todo')
        db.session.add(task)

    # Seed milestones
    milestones_data = [
        {"title": "STT Audio Frontend Complete", "component": "stt", "target_date": datetime(2025, 3, 15), "progress": 0},
        {"title": "STT Model Training Started", "component": "stt", "target_date": datetime(2025, 5, 1), "progress": 0},
        {"title": "RAG Vector Store Operational", "component": "rag", "target_date": datetime(2025, 4, 1), "progress": 0},
        {"title": "RAG Hybrid Retrieval Working", "component": "rag", "target_date": datetime(2025, 6, 1), "progress": 0},
        {"title": "TTS Pipeline Integrated", "component": "tts-lipsync", "target_date": datetime(2025, 3, 1), "progress": 0},
        {"title": "LipSync Demo Ready", "component": "tts-lipsync", "target_date": datetime(2025, 4, 1), "progress": 0},
        {"title": "Blender MCP Basic Tools", "component": "mcp", "target_date": datetime(2025, 2, 15), "progress": 0},
        {"title": "Full System Integration", "component": None, "target_date": datetime(2025, 12, 1), "progress": 0},
        {"title": "Project Deadline", "component": None, "target_date": datetime(2025, 12, 18), "progress": 0},
    ]

    for m_data in milestones_data:
        milestone = Milestone(**m_data)
        db.session.add(milestone)

    # Seed architecture decisions
    decisions_data = [
        {
            "title": "Use Conformer over vanilla Transformer for STT",
            "status": "accepted",
            "context": "Need to choose architecture for speech recognition that balances accuracy and efficiency",
            "decision": "Use Conformer blocks which combine self-attention with convolutions for better local and global pattern capture in audio",
            "consequences": "Better WER on speech benchmarks, slightly higher computational cost than vanilla transformer",
            "component": "stt",
            "author": "Team"
        },
        {
            "title": "Hybrid retrieval with BGE-M3 and BM25",
            "status": "accepted",
            "context": "Pure vector search misses keyword matches; pure BM25 misses semantic similarity",
            "decision": "Implement hybrid retrieval combining BGE-M3 dense vectors with BM25 sparse retrieval, fused via Reciprocal Rank Fusion",
            "consequences": "Better recall across diverse queries, requires maintaining two index types",
            "component": "rag",
            "author": "Team"
        },
        {
            "title": "ElevenLabs for TTS over local models",
            "status": "accepted",
            "context": "Need high-quality, low-latency TTS with voice cloning capabilities",
            "decision": "Use ElevenLabs API for production TTS due to superior voice quality and streaming support",
            "consequences": "API cost dependency, but significantly better voice quality than open-source alternatives",
            "component": "tts-lipsync",
            "author": "Team"
        },
        {
            "title": "Use Model Context Protocol for Blender integration",
            "status": "accepted",
            "context": "Need standardized way for AI to control Blender operations",
            "decision": "Adopt Anthropic's MCP protocol with BlenderMCP addon for AI-driven 3D asset manipulation",
            "consequences": "Standardized interface, but requires MCP server setup and Blender addon installation",
            "component": "mcp",
            "author": "Team"
        },
    ]

    for d_data in decisions_data:
        decision = Decision(**d_data)
        db.session.add(decision)

    # Seed resources
    resources_data = [
        {"title": "Conformer Paper", "url": "https://arxiv.org/abs/2005.08100", "category": "paper", "description": "Conformer: Convolution-augmented Transformer for Speech Recognition", "component": "stt", "tags": json.dumps(["speech", "transformer", "conformer"])},
        {"title": "RoPE Paper", "url": "https://arxiv.org/abs/2104.09864", "category": "paper", "description": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "component": "stt", "tags": json.dumps(["embeddings", "position", "attention"])},
        {"title": "BGE-M3 on HuggingFace", "url": "https://huggingface.co/BAAI/bge-m3", "category": "library", "description": "Multi-Functionality, Multi-Linguality, Multi-Granularity embedding model", "component": "rag", "tags": json.dumps(["embeddings", "multilingual"])},
        {"title": "pgvector Extension", "url": "https://github.com/pgvector/pgvector", "category": "tool", "description": "Open-source vector similarity search for PostgreSQL", "component": "rag", "tags": json.dumps(["vector", "database", "postgresql"])},
        {"title": "ElevenLabs Docs", "url": "https://elevenlabs.io/docs", "category": "docs", "description": "Official ElevenLabs API documentation", "component": "tts-lipsync", "tags": json.dumps(["tts", "api", "streaming"])},
        {"title": "Wav2Lip GitHub", "url": "https://github.com/Rudrabha/Wav2Lip", "category": "tool", "description": "Lip-syncing videos in the wild", "component": "tts-lipsync", "tags": json.dumps(["lipsync", "video", "neural"])},
        {"title": "BlenderMCP Addon", "url": "https://github.com/anthropics/mcp", "category": "tool", "description": "Model Context Protocol for Blender integration", "component": "mcp", "tags": json.dumps(["blender", "mcp", "3d"])},
        {"title": "Advanced RAG Techniques", "url": "https://arxiv.org/abs/2312.10997", "category": "paper", "description": "Survey of advanced RAG techniques", "component": "rag", "tags": json.dumps(["rag", "retrieval", "survey"])},
    ]

    for r_data in resources_data:
        resource = Resource(**r_data)
        db.session.add(resource)

    # Seed glossary
    glossary_data = [
        {"term": "STFT", "definition": "Short-Time Fourier Transform - converts audio signal from time domain to frequency domain using overlapping windows", "category": "audio", "component": "stt"},
        {"term": "Mel Spectrogram", "definition": "Audio representation using Mel-scale frequency bins that approximate human auditory perception", "category": "audio", "component": "stt"},
        {"term": "RoPE", "definition": "Rotary Position Embedding - encodes position information by rotating query/key vectors in attention", "category": "ml", "component": "stt"},
        {"term": "CTC", "definition": "Connectionist Temporal Classification - loss function for sequence-to-sequence tasks without explicit alignment", "category": "ml", "component": "stt"},
        {"term": "Conformer", "definition": "Transformer variant combining self-attention with convolutions for audio processing", "category": "ml", "component": "stt"},
        {"term": "HNSW", "definition": "Hierarchical Navigable Small World - graph-based approximate nearest neighbor search algorithm", "category": "ml", "component": "rag"},
        {"term": "BM25", "definition": "Best Matching 25 - probabilistic ranking function for keyword-based document retrieval", "category": "ml", "component": "rag"},
        {"term": "RRF", "definition": "Reciprocal Rank Fusion - method to combine rankings from multiple retrieval systems", "category": "ml", "component": "rag"},
        {"term": "Cross-Encoder", "definition": "Neural network that jointly encodes query and document for relevance scoring", "category": "ml", "component": "rag"},
        {"term": "Viseme", "definition": "Visual representation of a phoneme - the mouth shape corresponding to a speech sound", "category": "graphics", "component": "tts-lipsync"},
        {"term": "Blend Shape", "definition": "Morph target in 3D graphics that deforms mesh vertices for facial animation", "category": "graphics", "component": "tts-lipsync"},
        {"term": "MCP", "definition": "Model Context Protocol - Anthropic's standard for AI model-tool communication", "category": "general", "component": "mcp"},
        {"term": "FBX", "definition": "Filmbox format - 3D asset exchange format supporting geometry, materials, and animations", "category": "graphics", "component": "mcp"},
    ]

    for g_data in glossary_data:
        term = GlossaryTerm(**g_data)
        db.session.add(term)

    # Seed team members (example)
    team_data = [
        {"name": "Project Lead", "role": "Technical Lead", "email": "lead@project.com", "github": "project-lead", "status": "active", "bio": "Overseeing the full stack implementation of the AI assistant", "components": json.dumps(["stt", "rag", "tts-lipsync", "mcp"])},
    ]

    for t_data in team_data:
        member = TeamMember(**t_data)
        db.session.add(member)

    # Seed initial activity
    log_activity('project_created', 'Project initialized', 'The 3D Game AI Assistant project has been set up')

    db.session.commit()
    print("Database seeded with initial data!")

# ============== Register Blueprints ==============

# Avatar API (TTS + Lip-Sync)
try:
    from avatar_api import avatar_bp
    app.register_blueprint(avatar_bp)
    print("Avatar API blueprint registered")
except ImportError as e:
    print(f"Avatar API not available: {e}")

# RAG API (Agentic RAG System) - Production with GPT-5.1
try:
    from rag_api_production import rag_bp
    app.register_blueprint(rag_bp)
    print("RAG API (Production GPT-5.1) blueprint registered")
except ImportError as e:
    print(f"RAG Production API not available, falling back to demo: {e}")
    try:
        from rag_api import rag_bp
        app.register_blueprint(rag_bp)
        print("RAG API (Demo) blueprint registered")
    except ImportError as e2:
        print(f"RAG API not available: {e2}")

# ============== Main ==============

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_initial_data()

    app.run(debug=True, port=5000)
