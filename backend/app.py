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

# ============== Multi-Tenant Auth Models ==============

class Organization(db.Model):
    """Organization/Tenant model for multi-tenancy"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    slug = db.Column(db.String(50), unique=True, nullable=False)  # URL-friendly identifier
    domain = db.Column(db.String(100))  # Optional custom domain
    plan = db.Column(db.String(20), default='free')  # free, starter, pro, enterprise
    settings = db.Column(db.Text)  # JSON for org-specific settings
    salesforce_instance_url = db.Column(db.String(200))  # Connected Salesforce org
    salesforce_access_token = db.Column(db.Text)  # Encrypted token
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = db.relationship('User', backref='organization', lazy=True)

class User(db.Model):
    """User model with organization membership"""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))  # Nullable for OAuth users
    name = db.Column(db.String(100))
    avatar_url = db.Column(db.String(500))

    # Organization relationship
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    role = db.Column(db.String(20), default='member')  # owner, admin, member, viewer

    # OAuth fields
    oauth_provider = db.Column(db.String(20))  # google, microsoft, null for email
    oauth_id = db.Column(db.String(100))  # Provider's user ID

    # Status and tracking
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RefreshToken(db.Model):
    """Store refresh tokens for JWT authentication"""
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(500), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    revoked = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='refresh_tokens')

# ============== Session & Usage Models ==============

class ConsultantSession(db.Model):
    """Track consultant chat sessions"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), unique=True, nullable=False)  # UUID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))

    # Consultant info
    consultant_id = db.Column(db.String(50), nullable=False)  # e.g., 'alex', 'sarah'
    consultant_name = db.Column(db.String(100))
    language = db.Column(db.String(10), default='en')  # en, fr

    # Session metadata
    title = db.Column(db.String(200))  # Auto-generated from first message
    topic = db.Column(db.String(100))  # e.g., 'SOQL', 'Apex', 'Reports'
    status = db.Column(db.String(20), default='active')  # active, ended, archived

    # Usage tracking
    message_count = db.Column(db.Integer, default=0)
    tokens_used = db.Column(db.Integer, default=0)  # LLM tokens consumed
    duration_seconds = db.Column(db.Integer, default=0)

    # Timestamps
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    last_message_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    user = db.relationship('User', backref='sessions')
    messages = db.relationship('SessionMessage', backref='session', lazy=True, cascade='all, delete-orphan')

class SessionMessage(db.Model):
    """Individual messages within a session"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('consultant_session.id'), nullable=False)

    role = db.Column(db.String(20), nullable=False)  # user, assistant
    content = db.Column(db.Text, nullable=False)
    tokens = db.Column(db.Integer, default=0)

    # For RAG tracking
    sources_used = db.Column(db.Text)  # JSON array of doc references
    confidence_score = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UsageStats(db.Model):
    """Daily usage statistics per organization"""
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)

    # Counts
    sessions_count = db.Column(db.Integer, default=0)
    messages_count = db.Column(db.Integer, default=0)
    unique_users = db.Column(db.Integer, default=0)

    # Token usage
    tokens_used = db.Column(db.Integer, default=0)
    tokens_limit = db.Column(db.Integer)  # Based on plan

    # By consultant
    consultant_breakdown = db.Column(db.Text)  # JSON: {"alex": 10, "sarah": 5}

    # By topic
    topic_breakdown = db.Column(db.Text)  # JSON: {"SOQL": 8, "Apex": 7}

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('organization_id', 'date', name='unique_org_date'),
    )

# ============== Governance Models ==============

class GovernancePolicy(db.Model):
    """Policy for governing agent actions"""
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(20))  # type_a, type_b, type_c
    rules = db.Column(db.Text)  # JSON for policy rules
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = db.relationship('Organization', backref='policies')

class ApprovalRequest(db.Model):
    """Approval requests for governed actions"""
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    policy_id = db.Column(db.Integer, db.ForeignKey('governance_policy.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action_type = db.Column(db.String(50))  # create, update, delete, deploy, etc.
    target = db.Column(db.String(200))  # Target entity (e.g., "Apex Class: MyController")
    context = db.Column(db.Text)  # JSON with action context
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    decision_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    decision_reason = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    decided_at = db.Column(db.DateTime)

    organization = db.relationship('Organization', backref='approval_requests')
    policy = db.relationship('GovernancePolicy', backref='approval_requests')
    requester = db.relationship('User', foreign_keys=[user_id], backref='submitted_approvals')
    approver = db.relationship('User', foreign_keys=[decision_by], backref='decided_approvals')

class AuditLog(db.Model):
    """Audit log for all governance-related actions"""
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(50))  # create, read, update, delete, approve, reject
    entity_type = db.Column(db.String(50))  # policy, approval, pattern, knowledge
    entity_id = db.Column(db.Integer)
    description = db.Column(db.Text)
    changes = db.Column(db.Text)  # JSON for before/after state
    metadata = db.Column(db.Text)  # JSON for extra info
    ip_address = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    organization = db.relationship('Organization', backref='audit_logs')
    user = db.relationship('User', backref='audit_logs')

class KnowledgeBase(db.Model):
    """Knowledge base entries for AI context"""
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    category = db.Column(db.String(50))  # context, standard, constraint, reference
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)  # JSON for structured content
    tags = db.Column(db.Text)  # JSON array of tags
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = db.relationship('Organization', backref='knowledge_entries')

class Pattern(db.Model):
    """Pattern library for code patterns and best practices"""
    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(20))  # allowed, forbidden, recommended, deprecated
    platform = db.Column(db.String(50))  # salesforce, general
    pattern = db.Column(db.Text)  # JSON for pattern definition
    description = db.Column(db.Text)
    rationale = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = db.relationship('Organization', backref='patterns')

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

# ============== User Auth Helpers ==============

import secrets
import re

def generate_access_token(user):
    """Generate a JWT access token for user authentication"""
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow(),
        'type': 'access',
        'user_id': user.id,
        'email': user.email,
        'org_id': user.organization_id,
        'role': user.role
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def generate_refresh_token_value():
    """Generate a secure random refresh token"""
    return secrets.token_urlsafe(64)

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength (min 8 chars, 1 upper, 1 lower, 1 number)"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, None

def create_slug(name):
    """Create URL-friendly slug from organization name"""
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', name.lower())
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    # Add random suffix to ensure uniqueness
    return f"{slug}-{secrets.token_hex(3)}"

def auth_required(f):
    """Decorator to require user authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            if payload.get('type') != 'access':
                return jsonify({'error': 'Invalid token type'}), 401
            # Get user and attach to request
            user = User.query.get(payload['user_id'])
            if not user or not user.is_active:
                return jsonify({'error': 'User not found or inactive'}), 401
            request.current_user = user
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    """Get current authenticated user from request"""
    return getattr(request, 'current_user', None)

def get_current_org_id():
    """Get current organization ID from authenticated user"""
    user = get_current_user()
    return user.organization_id if user else None

def org_required(f):
    """Decorator requiring user to belong to an organization"""
    @wraps(f)
    def decorated(user, *args, **kwargs):
        if not user.organization_id:
            return jsonify({'error': 'Organization required. Please create or join an organization.'}), 403
        return f(user, *args, **kwargs)
    return decorated

def org_admin_required(f):
    """Decorator requiring user to be org admin or owner"""
    @wraps(f)
    def decorated(user, *args, **kwargs):
        if not user.organization_id:
            return jsonify({'error': 'Organization required'}), 403
        if user.role not in ['owner', 'admin']:
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(user, *args, **kwargs)
    return decorated

def filter_by_org(query, model, user):
    """Filter query by user's organization (for multi-tenant data isolation)

    Usage:
        tasks = filter_by_org(Task.query, Task, user).all()
    """
    if hasattr(model, 'organization_id'):
        return query.filter_by(organization_id=user.organization_id)
    return query

class OrgBoundMixin:
    """Mixin for models that are organization-scoped"""
    organization_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=True)

    @classmethod
    def get_for_org(cls, user):
        """Get all records for user's organization"""
        if not user.organization_id:
            return cls.query.filter(False)  # Return empty
        return cls.query.filter_by(organization_id=user.organization_id)

    @classmethod
    def get_one_for_org(cls, id, user):
        """Get single record, ensuring it belongs to user's org"""
        record = cls.query.get(id)
        if not record:
            return None
        if hasattr(record, 'organization_id') and record.organization_id != user.organization_id:
            return None
        return record

# ============== User Auth Routes ==============

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user and create their organization"""
    data = request.json or {}

    # Validate required fields
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    org_name = data.get('organization_name', '').strip()

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    if not validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400

    valid, error = validate_password(password)
    if not valid:
        return jsonify({'error': error}), 400

    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    try:
        # Create organization if name provided, otherwise use email domain
        if not org_name:
            org_name = email.split('@')[1].split('.')[0].title() + ' Workspace'

        org = Organization(
            name=org_name,
            slug=create_slug(org_name),
            plan='free'
        )
        db.session.add(org)
        db.session.flush()  # Get org.id

        # Create user as org owner
        user = User(
            email=email,
            password_hash=hash_password(password),
            name=name or email.split('@')[0],
            organization_id=org.id,
            role='owner',
            is_verified=True  # Skip email verification for now
        )
        db.session.add(user)
        db.session.flush()

        # Generate tokens
        access_token = generate_access_token(user)
        refresh_token_value = generate_refresh_token_value()

        # Store refresh token
        refresh_token = RefreshToken(
            token=refresh_token_value,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        db.session.add(refresh_token)
        db.session.commit()

        log_activity('user_registered', f'User registered: {email}', f'New user {name} registered')

        return jsonify({
            'message': 'Registration successful',
            'access_token': access_token,
            'refresh_token': refresh_token_value,
            'expires_in': 86400,  # 24 hours
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name,
                'role': user.role,
                'organization': {
                    'id': org.id,
                    'name': org.name,
                    'slug': org.slug,
                    'plan': org.plan
                }
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user and return tokens"""
    data = request.json or {}

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    # Find user
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401

    if not user.password_hash:
        return jsonify({'error': 'Please use social login for this account'}), 401

    if not verify_password(password, user.password_hash):
        return jsonify({'error': 'Invalid email or password'}), 401

    if not user.is_active:
        return jsonify({'error': 'Account is deactivated'}), 401

    try:
        # Update last login
        user.last_login = datetime.utcnow()

        # Generate tokens
        access_token = generate_access_token(user)
        refresh_token_value = generate_refresh_token_value()

        # Revoke old refresh tokens for this user (optional: keep last N)
        RefreshToken.query.filter_by(user_id=user.id, revoked=False).update({'revoked': True})

        # Store new refresh token
        refresh_token = RefreshToken(
            token=refresh_token_value,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        db.session.add(refresh_token)
        db.session.commit()

        log_activity('user_login', f'User logged in: {email}', f'{user.name} logged in')

        # Get organization
        org_data = None
        if user.organization:
            org_data = {
                'id': user.organization.id,
                'name': user.organization.name,
                'slug': user.organization.slug,
                'plan': user.organization.plan
            }

        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'refresh_token': refresh_token_value,
            'expires_in': 86400,  # 24 hours
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name,
                'avatar_url': user.avatar_url,
                'role': user.role,
                'organization': org_data
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/refresh', methods=['POST'])
def refresh_token():
    """Refresh access token using refresh token"""
    data = request.json or {}
    token = data.get('refresh_token', '')

    if not token:
        return jsonify({'error': 'Refresh token is required'}), 400

    # Find refresh token
    stored_token = RefreshToken.query.filter_by(token=token, revoked=False).first()

    if not stored_token:
        return jsonify({'error': 'Invalid refresh token'}), 401

    if stored_token.expires_at < datetime.utcnow():
        stored_token.revoked = True
        db.session.commit()
        return jsonify({'error': 'Refresh token expired'}), 401

    user = stored_token.user
    if not user or not user.is_active:
        return jsonify({'error': 'User not found or inactive'}), 401

    try:
        # Generate new access token
        access_token = generate_access_token(user)

        # Optionally rotate refresh token
        new_refresh_token_value = generate_refresh_token_value()
        stored_token.revoked = True

        new_refresh_token = RefreshToken(
            token=new_refresh_token_value,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        db.session.add(new_refresh_token)
        db.session.commit()

        return jsonify({
            'access_token': access_token,
            'refresh_token': new_refresh_token_value,
            'expires_in': 86400
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Token refresh failed: {str(e)}'}), 500

@app.route('/api/auth/me', methods=['GET'])
@auth_required
def get_me():
    """Get current authenticated user's profile"""
    user = get_current_user()

    org_data = None
    if user.organization:
        org_data = {
            'id': user.organization.id,
            'name': user.organization.name,
            'slug': user.organization.slug,
            'plan': user.organization.plan
        }

    return jsonify({
        'id': user.id,
        'email': user.email,
        'name': user.name,
        'avatar_url': user.avatar_url,
        'role': user.role,
        'is_verified': user.is_verified,
        'last_login': user.last_login.isoformat() if user.last_login else None,
        'created_at': user.created_at.isoformat(),
        'organization': org_data
    })

@app.route('/api/auth/me', methods=['PUT'])
@auth_required
def update_me():
    """Update current user's profile"""
    user = get_current_user()
    data = request.json or {}

    if 'name' in data:
        user.name = data['name'].strip()
    if 'avatar_url' in data:
        user.avatar_url = data['avatar_url']

    try:
        db.session.commit()
        return jsonify({'message': 'Profile updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@auth_required
def logout():
    """Logout user by revoking refresh tokens"""
    user = get_current_user()
    data = request.json or {}
    refresh_token = data.get('refresh_token', '')

    try:
        if refresh_token:
            # Revoke specific token
            RefreshToken.query.filter_by(token=refresh_token, user_id=user.id).update({'revoked': True})
        else:
            # Revoke all user's tokens
            RefreshToken.query.filter_by(user_id=user.id, revoked=False).update({'revoked': True})

        db.session.commit()
        log_activity('user_logout', f'User logged out: {user.email}', f'{user.name} logged out')
        return jsonify({'message': 'Logged out successfully'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Logout failed: {str(e)}'}), 500

@app.route('/api/auth/change-password', methods=['POST'])
@auth_required
def change_password():
    """Change current user's password"""
    user = get_current_user()
    data = request.json or {}

    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')

    if not current_password or not new_password:
        return jsonify({'error': 'Current and new password are required'}), 400

    if not user.password_hash:
        return jsonify({'error': 'Cannot change password for OAuth accounts'}), 400

    if not verify_password(current_password, user.password_hash):
        return jsonify({'error': 'Current password is incorrect'}), 401

    valid, error = validate_password(new_password)
    if not valid:
        return jsonify({'error': error}), 400

    try:
        user.password_hash = hash_password(new_password)
        # Revoke all refresh tokens to force re-login
        RefreshToken.query.filter_by(user_id=user.id, revoked=False).update({'revoked': True})
        db.session.commit()

        log_activity('password_changed', f'Password changed: {user.email}', f'{user.name} changed password')
        return jsonify({'message': 'Password changed successfully. Please login again.'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Password change failed: {str(e)}'}), 500

# ============== Organization APIs ==============

@app.route('/api/organizations', methods=['GET'])
@auth_required
def list_organizations(user):
    """List organizations the user belongs to"""
    # For now, return the user's organization
    # In future, support multiple org memberships
    if user.organization:
        org = user.organization
        member_count = User.query.filter_by(organization_id=org.id, is_active=True).count()
        return jsonify({
            'organizations': [{
                'id': org.id,
                'name': org.name,
                'slug': org.slug,
                'domain': org.domain,
                'plan': org.plan,
                'role': user.role,
                'member_count': member_count,
                'created_at': org.created_at.isoformat()
            }]
        })
    return jsonify({'organizations': []})

@app.route('/api/organizations', methods=['POST'])
@auth_required
def create_organization(user):
    """Create a new organization"""
    data = request.json or {}

    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Organization name is required'}), 400

    # Generate slug
    slug = create_slug(name)
    base_slug = slug
    counter = 1
    while Organization.query.filter_by(slug=slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1

    try:
        org = Organization(
            name=name,
            slug=slug,
            domain=data.get('domain'),
            plan=data.get('plan', 'free'),
            settings=json.dumps(data.get('settings', {}))
        )
        db.session.add(org)
        db.session.flush()  # Get org.id

        # Update user to be owner of new org
        user.organization_id = org.id
        user.role = 'owner'
        db.session.commit()

        log_activity('org_created', f'Organization created: {org.name}', f'{user.email} created organization {org.name}')

        return jsonify({
            'id': org.id,
            'name': org.name,
            'slug': org.slug,
            'plan': org.plan,
            'message': 'Organization created successfully'
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to create organization: {str(e)}'}), 500

@app.route('/api/organizations/<int:org_id>', methods=['GET'])
@auth_required
def get_organization(user, org_id):
    """Get organization details"""
    org = Organization.query.get_or_404(org_id)

    # Check user has access
    if user.organization_id != org.id:
        return jsonify({'error': 'Access denied'}), 403

    members = User.query.filter_by(organization_id=org.id, is_active=True).all()

    return jsonify({
        'id': org.id,
        'name': org.name,
        'slug': org.slug,
        'domain': org.domain,
        'plan': org.plan,
        'settings': json.loads(org.settings) if org.settings else {},
        'salesforce_connected': bool(org.salesforce_instance_url),
        'created_at': org.created_at.isoformat(),
        'updated_at': org.updated_at.isoformat(),
        'members': [{
            'id': m.id,
            'email': m.email,
            'name': m.name,
            'role': m.role,
            'avatar_url': m.avatar_url,
            'last_login': m.last_login.isoformat() if m.last_login else None
        } for m in members]
    })

@app.route('/api/organizations/<int:org_id>', methods=['PUT'])
@auth_required
def update_organization(user, org_id):
    """Update organization details"""
    org = Organization.query.get_or_404(org_id)

    # Check user has permission (owner or admin)
    if user.organization_id != org.id or user.role not in ['owner', 'admin']:
        return jsonify({'error': 'Permission denied'}), 403

    data = request.json or {}

    if 'name' in data:
        org.name = data['name'].strip()
    if 'domain' in data:
        org.domain = data['domain']
    if 'settings' in data:
        org.settings = json.dumps(data['settings'])

    # Only owner can change plan (normally done via billing)
    if 'plan' in data and user.role == 'owner':
        org.plan = data['plan']

    try:
        db.session.commit()
        log_activity('org_updated', f'Organization updated: {org.name}', f'{user.email} updated organization settings')
        return jsonify({'message': 'Organization updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500

@app.route('/api/organizations/<int:org_id>', methods=['DELETE'])
@auth_required
def delete_organization(user, org_id):
    """Delete organization (owner only)"""
    org = Organization.query.get_or_404(org_id)

    # Only owner can delete
    if user.organization_id != org.id or user.role != 'owner':
        return jsonify({'error': 'Only organization owner can delete'}), 403

    try:
        org_name = org.name
        # Remove all users from org (don't delete users, just unassign)
        User.query.filter_by(organization_id=org.id).update({'organization_id': None, 'role': 'member'})
        db.session.delete(org)
        db.session.commit()

        log_activity('org_deleted', f'Organization deleted: {org_name}', f'{user.email} deleted organization')
        return jsonify({'message': 'Organization deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500

@app.route('/api/organizations/<int:org_id>/members', methods=['GET'])
@auth_required
def list_organization_members(user, org_id):
    """List organization members"""
    org = Organization.query.get_or_404(org_id)

    if user.organization_id != org.id:
        return jsonify({'error': 'Access denied'}), 403

    members = User.query.filter_by(organization_id=org.id).all()

    return jsonify({
        'members': [{
            'id': m.id,
            'email': m.email,
            'name': m.name,
            'role': m.role,
            'avatar_url': m.avatar_url,
            'is_active': m.is_active,
            'last_login': m.last_login.isoformat() if m.last_login else None,
            'created_at': m.created_at.isoformat()
        } for m in members]
    })

@app.route('/api/organizations/<int:org_id>/members', methods=['POST'])
@auth_required
def invite_member(user, org_id):
    """Invite a new member to the organization"""
    org = Organization.query.get_or_404(org_id)

    # Check permission (owner or admin can invite)
    if user.organization_id != org.id or user.role not in ['owner', 'admin']:
        return jsonify({'error': 'Permission denied'}), 403

    data = request.json or {}
    email = data.get('email', '').strip().lower()
    role = data.get('role', 'member')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Validate role
    if role not in ['admin', 'member', 'viewer']:
        return jsonify({'error': 'Invalid role'}), 400

    # Check if user exists
    existing_user = User.query.filter_by(email=email).first()

    if existing_user:
        if existing_user.organization_id == org.id:
            return jsonify({'error': 'User is already a member'}), 400
        if existing_user.organization_id:
            return jsonify({'error': 'User belongs to another organization'}), 400

        # Add existing user to org
        existing_user.organization_id = org.id
        existing_user.role = role
        db.session.commit()

        log_activity('member_added', f'Member added: {email}', f'{user.email} added {email} to {org.name}')
        return jsonify({'message': f'User {email} added to organization', 'user_id': existing_user.id})

    # Create invitation (for now, create inactive user)
    # In production, send email invitation
    try:
        new_user = User(
            email=email,
            organization_id=org.id,
            role=role,
            is_active=False,  # Will be activated when they accept invite
            is_verified=False
        )
        db.session.add(new_user)
        db.session.commit()

        log_activity('member_invited', f'Member invited: {email}', f'{user.email} invited {email} to {org.name}')
        return jsonify({
            'message': f'Invitation sent to {email}',
            'user_id': new_user.id,
            'note': 'User needs to complete registration'
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Invitation failed: {str(e)}'}), 500

@app.route('/api/organizations/<int:org_id>/members/<int:member_id>', methods=['PUT'])
@auth_required
def update_member(user, org_id, member_id):
    """Update member role"""
    org = Organization.query.get_or_404(org_id)

    # Check permission
    if user.organization_id != org.id or user.role not in ['owner', 'admin']:
        return jsonify({'error': 'Permission denied'}), 403

    member = User.query.get_or_404(member_id)
    if member.organization_id != org.id:
        return jsonify({'error': 'Member not found'}), 404

    data = request.json or {}
    new_role = data.get('role')

    if new_role:
        # Can't change owner role unless you're the owner
        if member.role == 'owner' and user.role != 'owner':
            return jsonify({'error': 'Cannot modify owner role'}), 403

        # Can't promote to owner unless you're the owner
        if new_role == 'owner' and user.role != 'owner':
            return jsonify({'error': 'Only owner can transfer ownership'}), 403

        # If transferring ownership, demote current owner
        if new_role == 'owner':
            user.role = 'admin'

        member.role = new_role
        db.session.commit()

        log_activity('member_updated', f'Member role changed: {member.email}', f'{user.email} changed {member.email} role to {new_role}')
        return jsonify({'message': 'Member updated successfully'})

    return jsonify({'error': 'No changes specified'}), 400

@app.route('/api/organizations/<int:org_id>/members/<int:member_id>', methods=['DELETE'])
@auth_required
def remove_member(user, org_id, member_id):
    """Remove member from organization"""
    org = Organization.query.get_or_404(org_id)

    # Check permission
    if user.organization_id != org.id or user.role not in ['owner', 'admin']:
        return jsonify({'error': 'Permission denied'}), 403

    member = User.query.get_or_404(member_id)
    if member.organization_id != org.id:
        return jsonify({'error': 'Member not found'}), 404

    # Can't remove owner
    if member.role == 'owner':
        return jsonify({'error': 'Cannot remove organization owner'}), 403

    # Can't remove yourself (use leave instead)
    if member.id == user.id:
        return jsonify({'error': 'Use leave endpoint to leave organization'}), 400

    member.organization_id = None
    member.role = 'member'
    db.session.commit()

    log_activity('member_removed', f'Member removed: {member.email}', f'{user.email} removed {member.email} from {org.name}')
    return jsonify({'message': 'Member removed successfully'})

@app.route('/api/organizations/current/leave', methods=['POST'])
@auth_required
def leave_organization(user):
    """Leave current organization"""
    if not user.organization_id:
        return jsonify({'error': 'You are not in an organization'}), 400

    if user.role == 'owner':
        return jsonify({'error': 'Owner cannot leave. Transfer ownership or delete the organization.'}), 400

    org_name = user.organization.name
    user.organization_id = None
    user.role = 'member'
    db.session.commit()

    log_activity('member_left', f'Member left: {user.email}', f'{user.email} left {org_name}')
    return jsonify({'message': 'Successfully left organization'})

# ============== Session APIs ==============

@app.route('/api/sessions', methods=['GET'])
@auth_required
def list_sessions(user):
    """List user's consultant sessions"""
    status_filter = request.args.get('status')  # active, ended, archived
    consultant = request.args.get('consultant')
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = int(request.args.get('offset', 0))

    query = ConsultantSession.query.filter_by(user_id=user.id)

    if user.organization_id:
        query = query.filter_by(organization_id=user.organization_id)

    if status_filter:
        query = query.filter_by(status=status_filter)

    if consultant:
        query = query.filter_by(consultant_id=consultant)

    total = query.count()
    sessions = query.order_by(ConsultantSession.last_message_at.desc())\
                    .offset(offset).limit(limit).all()

    return jsonify({
        'sessions': [{
            'id': s.id,
            'session_id': s.session_id,
            'consultant_id': s.consultant_id,
            'consultant_name': s.consultant_name,
            'language': s.language,
            'title': s.title,
            'topic': s.topic,
            'status': s.status,
            'message_count': s.message_count,
            'tokens_used': s.tokens_used,
            'duration_seconds': s.duration_seconds,
            'started_at': s.started_at.isoformat() if s.started_at else None,
            'ended_at': s.ended_at.isoformat() if s.ended_at else None,
            'last_message_at': s.last_message_at.isoformat() if s.last_message_at else None
        } for s in sessions],
        'total': total,
        'limit': limit,
        'offset': offset
    })

@app.route('/api/sessions', methods=['POST'])
@auth_required
def create_session(user):
    """Start a new consultant session"""
    data = request.json or {}

    consultant_id = data.get('consultant_id')
    if not consultant_id:
        return jsonify({'error': 'consultant_id is required'}), 400

    import uuid
    session = ConsultantSession(
        session_id=str(uuid.uuid4()),
        user_id=user.id,
        organization_id=user.organization_id,
        consultant_id=consultant_id,
        consultant_name=data.get('consultant_name', consultant_id.title()),
        language=data.get('language', 'en'),
        title=data.get('title'),
        topic=data.get('topic'),
        status='active'
    )

    db.session.add(session)
    db.session.commit()

    # Update daily stats
    update_daily_stats(user.organization_id, sessions_delta=1)

    return jsonify({
        'id': session.id,
        'session_id': session.session_id,
        'consultant_id': session.consultant_id,
        'status': session.status,
        'started_at': session.started_at.isoformat()
    }), 201

@app.route('/api/sessions/<session_id>', methods=['GET'])
@auth_required
def get_session(user, session_id):
    """Get session details with messages"""
    session = ConsultantSession.query.filter_by(session_id=session_id).first()

    if not session:
        return jsonify({'error': 'Session not found'}), 404

    if session.user_id != user.id:
        return jsonify({'error': 'Access denied'}), 403

    messages = SessionMessage.query.filter_by(session_id=session.id)\
                                   .order_by(SessionMessage.created_at).all()

    return jsonify({
        'id': session.id,
        'session_id': session.session_id,
        'consultant_id': session.consultant_id,
        'consultant_name': session.consultant_name,
        'language': session.language,
        'title': session.title,
        'topic': session.topic,
        'status': session.status,
        'message_count': session.message_count,
        'tokens_used': session.tokens_used,
        'duration_seconds': session.duration_seconds,
        'started_at': session.started_at.isoformat() if session.started_at else None,
        'ended_at': session.ended_at.isoformat() if session.ended_at else None,
        'messages': [{
            'id': m.id,
            'role': m.role,
            'content': m.content,
            'tokens': m.tokens,
            'sources_used': json.loads(m.sources_used) if m.sources_used else None,
            'confidence_score': m.confidence_score,
            'created_at': m.created_at.isoformat()
        } for m in messages]
    })

@app.route('/api/sessions/<session_id>/messages', methods=['POST'])
@auth_required
def add_message(user, session_id):
    """Add a message to a session"""
    session = ConsultantSession.query.filter_by(session_id=session_id).first()

    if not session:
        return jsonify({'error': 'Session not found'}), 404

    if session.user_id != user.id:
        return jsonify({'error': 'Access denied'}), 403

    data = request.json or {}
    role = data.get('role', 'user')
    content = data.get('content', '').strip()

    if not content:
        return jsonify({'error': 'Content is required'}), 400

    tokens = data.get('tokens', len(content) // 4)  # Rough estimate if not provided

    message = SessionMessage(
        session_id=session.id,
        role=role,
        content=content,
        tokens=tokens,
        sources_used=json.dumps(data.get('sources_used')) if data.get('sources_used') else None,
        confidence_score=data.get('confidence_score')
    )

    db.session.add(message)

    # Update session stats
    session.message_count += 1
    session.tokens_used += tokens
    session.last_message_at = datetime.utcnow()

    # Auto-generate title from first user message
    if not session.title and role == 'user':
        session.title = content[:100] + ('...' if len(content) > 100 else '')

    db.session.commit()

    # Update daily stats
    update_daily_stats(user.organization_id, messages_delta=1, tokens_delta=tokens)

    return jsonify({
        'id': message.id,
        'role': message.role,
        'content': message.content,
        'created_at': message.created_at.isoformat()
    }), 201

@app.route('/api/sessions/<session_id>/end', methods=['POST'])
@auth_required
def end_session(user, session_id):
    """End an active session"""
    session = ConsultantSession.query.filter_by(session_id=session_id).first()

    if not session:
        return jsonify({'error': 'Session not found'}), 404

    if session.user_id != user.id:
        return jsonify({'error': 'Access denied'}), 403

    if session.status != 'active':
        return jsonify({'error': 'Session is not active'}), 400

    session.status = 'ended'
    session.ended_at = datetime.utcnow()
    session.duration_seconds = int((session.ended_at - session.started_at).total_seconds())

    db.session.commit()

    return jsonify({
        'message': 'Session ended',
        'duration_seconds': session.duration_seconds
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@auth_required
def delete_session(user, session_id):
    """Delete/archive a session"""
    session = ConsultantSession.query.filter_by(session_id=session_id).first()

    if not session:
        return jsonify({'error': 'Session not found'}), 404

    if session.user_id != user.id:
        return jsonify({'error': 'Access denied'}), 403

    # Soft delete by archiving
    session.status = 'archived'
    db.session.commit()

    return jsonify({'message': 'Session archived'})

# ============== Usage Statistics APIs ==============

def update_daily_stats(org_id, sessions_delta=0, messages_delta=0, tokens_delta=0):
    """Update or create daily usage stats"""
    if not org_id:
        return

    from datetime import date
    today = date.today()

    stats = UsageStats.query.filter_by(organization_id=org_id, date=today).first()

    if not stats:
        stats = UsageStats(
            organization_id=org_id,
            date=today,
            sessions_count=0,
            messages_count=0,
            tokens_used=0
        )
        db.session.add(stats)

    stats.sessions_count += sessions_delta
    stats.messages_count += messages_delta
    stats.tokens_used += tokens_delta

    db.session.commit()

@app.route('/api/stats/usage', methods=['GET'])
@auth_required
def get_usage_stats(user):
    """Get usage statistics for the organization"""
    if not user.organization_id:
        return jsonify({'error': 'Organization required'}), 403

    from datetime import date, timedelta

    # Get date range from query params
    days = int(request.args.get('days', 30))
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    stats = UsageStats.query.filter(
        UsageStats.organization_id == user.organization_id,
        UsageStats.date >= start_date,
        UsageStats.date <= end_date
    ).order_by(UsageStats.date).all()

    # Calculate totals
    total_sessions = sum(s.sessions_count for s in stats)
    total_messages = sum(s.messages_count for s in stats)
    total_tokens = sum(s.tokens_used for s in stats)

    # Get active sessions count
    active_sessions = ConsultantSession.query.filter_by(
        organization_id=user.organization_id,
        status='active'
    ).count()

    # Get unique users this period
    unique_users = db.session.query(db.func.count(db.distinct(ConsultantSession.user_id)))\
        .filter(
            ConsultantSession.organization_id == user.organization_id,
            ConsultantSession.started_at >= datetime.combine(start_date, datetime.min.time())
        ).scalar() or 0

    return jsonify({
        'period': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'days': days
        },
        'totals': {
            'sessions': total_sessions,
            'messages': total_messages,
            'tokens': total_tokens,
            'unique_users': unique_users,
            'active_sessions': active_sessions
        },
        'daily': [{
            'date': s.date.isoformat(),
            'sessions': s.sessions_count,
            'messages': s.messages_count,
            'tokens': s.tokens_used
        } for s in stats]
    })

@app.route('/api/stats/consultants', methods=['GET'])
@auth_required
def get_consultant_stats(user):
    """Get usage breakdown by consultant"""
    if not user.organization_id:
        return jsonify({'error': 'Organization required'}), 403

    from datetime import date, timedelta
    days = int(request.args.get('days', 30))
    start_date = date.today() - timedelta(days=days)

    # Query sessions grouped by consultant
    results = db.session.query(
        ConsultantSession.consultant_id,
        ConsultantSession.consultant_name,
        db.func.count(ConsultantSession.id).label('session_count'),
        db.func.sum(ConsultantSession.message_count).label('message_count'),
        db.func.sum(ConsultantSession.tokens_used).label('tokens_used')
    ).filter(
        ConsultantSession.organization_id == user.organization_id,
        ConsultantSession.started_at >= datetime.combine(start_date, datetime.min.time())
    ).group_by(
        ConsultantSession.consultant_id,
        ConsultantSession.consultant_name
    ).all()

    return jsonify({
        'consultants': [{
            'consultant_id': r.consultant_id,
            'consultant_name': r.consultant_name,
            'sessions': r.session_count,
            'messages': r.message_count or 0,
            'tokens': r.tokens_used or 0
        } for r in results]
    })

@app.route('/api/stats/topics', methods=['GET'])
@auth_required
def get_topic_stats(user):
    """Get usage breakdown by topic"""
    if not user.organization_id:
        return jsonify({'error': 'Organization required'}), 403

    from datetime import date, timedelta
    days = int(request.args.get('days', 30))
    start_date = date.today() - timedelta(days=days)

    results = db.session.query(
        ConsultantSession.topic,
        db.func.count(ConsultantSession.id).label('session_count'),
        db.func.sum(ConsultantSession.message_count).label('message_count')
    ).filter(
        ConsultantSession.organization_id == user.organization_id,
        ConsultantSession.started_at >= datetime.combine(start_date, datetime.min.time()),
        ConsultantSession.topic.isnot(None)
    ).group_by(
        ConsultantSession.topic
    ).all()

    return jsonify({
        'topics': [{
            'topic': r.topic,
            'sessions': r.session_count,
            'messages': r.message_count or 0
        } for r in results]
    })

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

# Salesforce RAG API (Salesforce Consultant Assistant)
try:
    from salesforce_rag_api import salesforce_rag_bp
    app.register_blueprint(salesforce_rag_bp)
    print("Salesforce RAG API blueprint registered")
except ImportError as e:
    print(f"Salesforce RAG API not available: {e}")

# Salesforce MCP API (Direct Salesforce operations)
try:
    from salesforce_mcp_api import salesforce_mcp_bp
    app.register_blueprint(salesforce_mcp_bp)
    print("Salesforce MCP API blueprint registered")
except ImportError as e:
    print(f"Salesforce MCP API not available: {e}")

# Governance API (Policies, Approvals, Audit, Knowledge, Patterns)
try:
    from governance_api import governance_bp
    app.register_blueprint(governance_bp)
    print("Governance API blueprint registered")
except ImportError as e:
    print(f"Governance API not available: {e}")

# ============== Main ==============

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_initial_data()

    app.run(debug=True, port=5000)
