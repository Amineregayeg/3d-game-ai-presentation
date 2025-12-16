"""
Database Setup Script for Production RAG System
Creates PostgreSQL schema with pgvector support
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ragdb",
    "user": "raguser",
    "password": "<RAG_DB_PASSWORD>"
}

EMBEDDING_DIM = 1024  # BGE-M3 dimension

def create_schema():
    """Create database schema for RAG system"""

    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    print("Creating database schema...")

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✓ pgvector extension enabled")

    # Create documents table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,

            -- Content
            content TEXT NOT NULL,
            title VARCHAR(500),
            embedding vector({EMBEDDING_DIM}),

            -- Metadata - Domain
            source VARCHAR(50) DEFAULT 'blender_docs',
            blender_version VARCHAR(10) DEFAULT '4.2',
            category VARCHAR(50) DEFAULT 'general',
            subcategory VARCHAR(50),

            -- Metadata - Content Type
            language VARCHAR(20) DEFAULT 'english',
            is_code BOOLEAN DEFAULT FALSE,
            modality VARCHAR(50) DEFAULT 'documentation',
            difficulty VARCHAR(20) DEFAULT 'intermediate',

            -- Ranking
            priority FLOAT DEFAULT 0.5,
            relevance_score FLOAT DEFAULT 0.0,

            -- Temporal
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Source tracking
            source_url VARCHAR(1000),
            source_hash VARCHAR(64),

            -- Tags
            tags TEXT[],
            keywords TEXT[],

            -- Full-text search
            tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('english', content || ' ' || COALESCE(title, ''))
            ) STORED
        );
    """)
    print("✓ documents table created")

    # Create HNSW index for vector search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding
        ON documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
    """)
    print("✓ HNSW vector index created")

    # Create GIN index for full-text search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_tsv
        ON documents
        USING GIN (tsv);
    """)
    print("✓ GIN full-text index created")

    # Create other useful indexes
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_source_version
        ON documents(source, blender_version);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_category
        ON documents(category, subcategory);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_is_code
        ON documents(is_code, language);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_tags
        ON documents USING GIN (tags);
    """)
    print("✓ Additional indexes created")

    # Create retrieval history table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS retrieval_history (
            id BIGSERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            query_embedding vector({EMBEDDING_DIM}),
            retrieved_doc_ids BIGINT[],
            final_answer TEXT,
            user_satisfaction FLOAT,
            latency_ms INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(100)
        );
    """)
    print("✓ retrieval_history table created")

    # Create sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR(100) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            history JSONB DEFAULT '[]'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb
        );
    """)
    print("✓ sessions table created")

    cur.close()
    conn.close()

    print("\n✅ Database schema created successfully!")


def insert_sample_documents():
    """Insert sample Blender documentation for testing"""

    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("\nLoading embedding model...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"✓ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")

    # Sample Blender documentation
    sample_docs = [
        {
            "title": "Face Selection in Blender",
            "content": """To select all faces in Blender, follow these steps:
1. Enter Edit Mode by pressing Tab
2. Make sure you're in Face Select mode (press 3 on keyboard)
3. Press A to select all faces
4. Alternatively, use bpy.ops.mesh.select_all(action='SELECT') in Python

The select_all operator works in all edit modes (vertex, edge, face).
You can also use Shift+Click to add to selection, or Ctrl+Click to remove from selection.""",
            "source": "blender_manual",
            "category": "mesh",
            "subcategory": "selection",
            "is_code": False,
            "tags": ["selection", "faces", "edit mode", "mesh"]
        },
        {
            "title": "bpy.ops.mesh.select_all API Reference",
            "content": """The bpy.ops.mesh module contains mesh editing operators:

bpy.ops.mesh.select_all(action='SELECT')

Parameters:
- action (enum in ['TOGGLE', 'SELECT', 'DESELECT', 'INVERT'], default 'TOGGLE')
  - 'TOGGLE': Toggle selection state of all elements
  - 'SELECT': Select all mesh elements
  - 'DESELECT': Deselect all elements
  - 'INVERT': Invert the current selection

Example usage:
```python
import bpy
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
```

Note: This operator requires Edit Mode to be active.""",
            "source": "blender_api",
            "category": "scripting",
            "subcategory": "mesh_operators",
            "is_code": True,
            "tags": ["API", "mesh", "selection", "operators", "python"]
        },
        {
            "title": "Edit Mode Selection Shortcuts",
            "content": """Edit Mode Selection Shortcuts in Blender:

Selection Modes (number keys):
- 1: Vertex Mode - Select individual vertices
- 2: Edge Mode - Select edges between vertices
- 3: Face Mode - Select polygonal faces

Common Selection Shortcuts:
- A: Select All / Deselect All (toggle)
- B: Box Select (drag rectangle)
- C: Circle Select (brush-style)
- L: Select Linked (hover and press)
- Ctrl+L: Select All Linked
- Shift+G: Select Similar (opens menu)
- Alt+Click: Loop Select (edges/faces)
- Ctrl+Alt+Click: Ring Select

Selection Modifiers:
- Shift+Click: Add to selection
- Ctrl+Click: Remove from selection
- Shift+Ctrl+Click: Intersect selection""",
            "source": "blender_manual",
            "category": "mesh",
            "subcategory": "selection",
            "is_code": False,
            "tags": ["shortcuts", "selection", "edit mode", "mesh"]
        },
        {
            "title": "Creating Materials with Python",
            "content": """Creating materials in Blender using Python (bpy):

```python
import bpy

# Create a new material
mat = bpy.data.materials.new(name="MyMaterial")
mat.use_nodes = True

# Get the node tree
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Clear default nodes
nodes.clear()

# Add Principled BSDF shader
bsdf = nodes.new('ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)
bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)  # Red
bsdf.inputs['Roughness'].default_value = 0.4
bsdf.inputs['Metallic'].default_value = 0.0

# Add Material Output
output = nodes.new('ShaderNodeOutputMaterial')
output.location = (300, 0)

# Link BSDF to Output
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# Assign to active object
obj = bpy.context.active_object
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
```""",
            "source": "blender_api",
            "category": "materials",
            "subcategory": "nodes",
            "is_code": True,
            "tags": ["materials", "python", "nodes", "shaders", "principled"]
        },
        {
            "title": "Animation Keyframes in Python",
            "content": """Working with animation keyframes in Blender Python:

```python
import bpy

obj = bpy.context.active_object

# Set initial position and insert keyframe
obj.location = (0, 0, 0)
obj.keyframe_insert(data_path="location", frame=1)

# Set end position and insert keyframe
obj.location = (5, 0, 0)
obj.keyframe_insert(data_path="location", frame=50)

# Rotation keyframes
obj.rotation_euler = (0, 0, 0)
obj.keyframe_insert(data_path="rotation_euler", frame=1)

obj.rotation_euler = (0, 0, 3.14159)  # 180 degrees
obj.keyframe_insert(data_path="rotation_euler", frame=50)

# Scale keyframes
obj.scale = (1, 1, 1)
obj.keyframe_insert(data_path="scale", frame=1)

obj.scale = (2, 2, 2)
obj.keyframe_insert(data_path="scale", frame=50)

# Setting interpolation mode
for fcurve in obj.animation_data.action.fcurves:
    for keyframe in fcurve.keyframe_points:
        keyframe.interpolation = 'BEZIER'  # or 'LINEAR', 'CONSTANT'
```

The keyframe_insert() method works with any animatable property.""",
            "source": "blender_api",
            "category": "animation",
            "subcategory": "keyframes",
            "is_code": True,
            "tags": ["animation", "keyframes", "python", "location", "rotation"]
        },
        {
            "title": "UV Unwrapping Methods",
            "content": """UV Unwrapping in Blender:

Manual Unwrap Methods:
1. Smart UV Project (U > Smart UV Project)
   - Automatic, angle-based projection
   - Good for hard-surface models
   - Settings: Angle Limit, Island Margin

2. Cube Projection (U > Cube Projection)
   - Projects from 6 cube faces
   - Best for boxy objects

3. Cylinder Projection (U > Cylinder Projection)
   - Wraps UVs around cylinder axis
   - Good for cylindrical objects

4. Sphere Projection (U > Sphere Projection)
   - Projects from sphere surface
   - Best for spherical objects

5. Unwrap (U > Unwrap)
   - Uses seams marked by user
   - Most control, best results

Python API:
```python
import bpy

# Select object and enter edit mode
obj = bpy.context.active_object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# Smart UV Project
bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)

# Or standard unwrap (requires seams)
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.02)
```""",
            "source": "blender_manual",
            "category": "uv",
            "subcategory": "unwrapping",
            "is_code": True,
            "tags": ["UV", "unwrap", "texturing", "python"]
        },
        {
            "title": "Extrude Faces in Blender",
            "content": """Extruding faces in Blender:

UI Method:
1. Enter Edit Mode (Tab)
2. Switch to Face Select mode (3)
3. Select faces to extrude
4. Press E to extrude
5. Move mouse to set extrusion distance
6. Click to confirm or Enter

Extrude Options:
- E: Extrude Region (default, moves along normals)
- Alt+E: Extrude menu with more options
  - Extrude Faces Along Normals
  - Extrude Individual Faces
  - Extrude Manifold

Python API:
```python
import bpy

# Enter edit mode and select faces
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='SELECT')

# Extrude along normals
bpy.ops.mesh.extrude_region_move(
    TRANSFORM_OT_translate={
        "value": (0, 0, 1),  # Extrude distance
        "orient_type": 'NORMAL'
    }
)

# Or extrude individual faces
bpy.ops.mesh.extrude_faces_move(
    TRANSFORM_OT_shrink_fatten={"value": 0.5}
)
```""",
            "source": "blender_manual",
            "category": "mesh",
            "subcategory": "modeling",
            "is_code": True,
            "tags": ["extrude", "faces", "modeling", "mesh", "python"]
        },
        {
            "title": "Smooth Shading in Blender",
            "content": """Applying smooth shading in Blender:

Object Mode (entire object):
1. Select object
2. Right-click > Shade Smooth
   OR Object menu > Shade Smooth

Edit Mode (per face):
1. Select faces
2. Mesh menu > Shading > Shade Smooth

Auto Smooth (sharp edges preserved):
1. Apply Shade Smooth first
2. Object Data Properties > Normals > Auto Smooth
3. Set angle threshold (default 30°)

Python API:
```python
import bpy

obj = bpy.context.active_object

# Shade smooth (entire object)
bpy.ops.object.shade_smooth()

# Or per-polygon smooth
for poly in obj.data.polygons:
    poly.use_smooth = True

# Enable auto smooth
obj.data.use_auto_smooth = True
obj.data.auto_smooth_angle = 0.523599  # 30 degrees in radians

# Mark edges as sharp (for auto smooth)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
# Select edges...
bpy.ops.mesh.mark_sharp()
```""",
            "source": "blender_manual",
            "category": "mesh",
            "subcategory": "shading",
            "is_code": True,
            "tags": ["smooth", "shading", "normals", "auto smooth", "python"]
        }
    ]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print(f"\nInserting {len(sample_docs)} sample documents...")

    for doc in sample_docs:
        # Generate embedding
        embedding = model.encode(doc["content"], normalize_embeddings=True)

        cur.execute("""
            INSERT INTO documents (
                title, content, embedding, source, category, subcategory,
                is_code, tags, blender_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            doc["title"],
            doc["content"],
            embedding.tolist(),
            doc["source"],
            doc["category"],
            doc.get("subcategory"),
            doc["is_code"],
            doc["tags"],
            "4.2"
        ))

        doc_id = cur.fetchone()[0]
        print(f"  ✓ Inserted: {doc['title']} (ID: {doc_id})")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n✅ Inserted {len(sample_docs)} documents with embeddings!")


def verify_setup():
    """Verify the database setup"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("\n=== Database Verification ===")

    # Check document count
    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]
    print(f"Documents: {count}")

    # Check vector index
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'documents' AND indexdef LIKE '%hnsw%'
    """)
    hnsw_index = cur.fetchone()
    print(f"HNSW Index: {'✓' if hnsw_index else '✗'}")

    # Check text search index
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'documents' AND indexdef LIKE '%gin%'
    """)
    gin_index = cur.fetchone()
    print(f"GIN Index: {'✓' if gin_index else '✗'}")

    # Test vector search
    cur.execute("""
        SELECT id, title, 1 - (embedding <=> embedding) as self_sim
        FROM documents
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)
    result = cur.fetchone()
    if result:
        print(f"Vector Search Test: ✓ (self-similarity = {result[2]:.4f})")

    # Test text search
    cur.execute("""
        SELECT id, title, ts_rank(tsv, plainto_tsquery('english', 'select faces')) as rank
        FROM documents
        WHERE tsv @@ plainto_tsquery('english', 'select faces')
        ORDER BY rank DESC
        LIMIT 1
    """)
    result = cur.fetchone()
    if result:
        print(f"Text Search Test: ✓ (top result: {result[1]})")

    cur.close()
    conn.close()

    print("\n✅ Database verification complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_setup()
    elif len(sys.argv) > 1 and sys.argv[1] == "--sample":
        insert_sample_documents()
    else:
        create_schema()
        insert_sample_documents()
        verify_setup()
