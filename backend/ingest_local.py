#!/usr/bin/env python3
"""
Document Ingestion Script using Local Embeddings
Uses sentence-transformers all-MiniLM-L6-v2 (384 dimensions, 22M params)
Optimized for CPU inference
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Generator
from dataclasses import dataclass
from html.parser import HTMLParser
import psycopg2

# =============================================================================
# Configuration
# =============================================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "ragdb",
    "user": "raguser",
    "password": "<RAG_DB_PASSWORD>",
}

# Local embedding model (384 dimensions, very fast on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking configuration - smaller for efficiency
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
MIN_CHUNK_SIZE = 50

# Batch sizes
EMBEDDING_BATCH_SIZE = 64
DB_BATCH_SIZE = 100

# =============================================================================
# HTML Parser
# =============================================================================

class BlenderHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript'}
        self.skip_depth = 0
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_depth += 1
        elif tag == 'title':
            self.in_title = True
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.text_parts.append("\n\n### ")

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self.skip_depth > 0:
            self.skip_depth -= 1
        elif tag == 'title':
            self.in_title = False
        elif tag in ('p', 'div', 'li', 'tr'):
            self.text_parts.append("\n")
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self.in_title and not self.title:
            self.title = text
        else:
            self.text_parts.append(text + " ")

    def get_text(self) -> str:
        text = ''.join(self.text_parts)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

# =============================================================================
# Document Processing
# =============================================================================

@dataclass
class DocumentChunk:
    id: str
    content: str
    title: str
    source: str
    category: str
    version: str
    url_path: str
    chunk_index: int
    total_chunks: int

def extract_category_from_path(filepath: str) -> str:
    path_lower = filepath.lower()
    category_mapping = {
        'modeling': 'modeling', 'mesh': 'mesh', 'sculpt': 'sculpting',
        'animation': 'animation', 'rigging': 'rigging', 'armature': 'rigging',
        'render': 'rendering', 'cycles': 'rendering', 'eevee': 'rendering',
        'material': 'materials', 'shader': 'materials', 'texture': 'materials',
        'node': 'nodes', 'geometry_node': 'geometry_nodes',
        'composit': 'compositing', 'video': 'video_editing', 'sequencer': 'video_editing',
        'physics': 'physics', 'cloth': 'physics', 'fluid': 'physics', 'particle': 'physics',
        'uv': 'uv_mapping', 'unwrap': 'uv_mapping',
        'script': 'scripting', 'python': 'scripting', 'addon': 'addons',
        'interface': 'interface', 'editor': 'editors', 'viewport': 'editors',
        'scene': 'scene', 'object': 'objects', 'camera': 'camera', 'light': 'lighting',
        'grease_pencil': 'grease_pencil', 'constraint': 'constraints',
        'modifier': 'modifiers', 'driver': 'drivers', 'asset': 'assets',
        'file': 'files', 'import': 'import_export', 'export': 'import_export',
    }
    for key, category in category_mapping.items():
        if key in path_lower:
            return category
    return 'general'

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) >= MIN_CHUNK_SIZE else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)

        # Try to end at sentence boundary
        for punct in ['. ', '.\n', '? ', '!\n']:
            last_punct = chunk_text.rfind(punct)
            if last_punct > len(chunk_text) * 0.5:
                chunk_text = chunk_text[:last_punct + 1]
                break

        if len(chunk_text.split()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text.strip())

        start = end - overlap
        if start >= len(words):
            break

    return chunks

def process_html_file(filepath: Path, base_path: Path) -> Generator[DocumentChunk, None, None]:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
    except Exception as e:
        return

    parser = BlenderHTMLParser()
    try:
        parser.feed(html_content)
    except Exception:
        return

    text = parser.get_text()
    title = parser.title or filepath.stem.replace('_', ' ').title()

    if len(text.split()) < MIN_CHUNK_SIZE:
        return

    rel_path = filepath.relative_to(base_path)
    url_path = str(rel_path).replace('\\', '/')
    category = extract_category_from_path(str(rel_path))
    chunks = chunk_text(text)

    if not chunks:
        return

    for i, chunk_content in enumerate(chunks):
        if i == 0 and title:
            chunk_content = f"# {title}\n\n{chunk_content}"

        chunk_id = hashlib.md5(f"{url_path}:{i}".encode()).hexdigest()[:16]

        yield DocumentChunk(
            id=chunk_id,
            content=chunk_content,
            title=title,
            source="blender_manual",
            category=category,
            version="5.0",
            url_path=url_path,
            chunk_index=i,
            total_chunks=len(chunks),
        )

# =============================================================================
# Database Operations
# =============================================================================

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def setup_database():
    """Ensure database has correct schema for 384-dim embeddings."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Check current embedding dimension
    cur.execute("""
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'documents'::regclass AND attname = 'embedding'
    """)
    result = cur.fetchone()

    if result is None or result[0] != EMBEDDING_DIM + 4:  # pgvector adds 4 to dimension
        print(f"Updating schema for {EMBEDDING_DIM}-dimensional embeddings...")
        cur.execute("ALTER TABLE documents DROP COLUMN IF EXISTS embedding")
        cur.execute(f"ALTER TABLE documents ADD COLUMN embedding vector({EMBEDDING_DIM})")
        cur.execute("DROP INDEX IF EXISTS documents_embedding_idx")
        cur.execute(f"""
            CREATE INDEX documents_embedding_idx ON documents
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
        """)
        conn.commit()
        print("Schema updated!")

    cur.close()
    conn.close()

def clear_documents(source: str = None):
    conn = get_db_connection()
    cur = conn.cursor()
    if source:
        cur.execute("DELETE FROM documents WHERE source = %s", (source,))
    else:
        cur.execute("DELETE FROM documents")
    conn.commit()
    cur.close()
    conn.close()

def insert_documents_batch(chunks: List[DocumentChunk], embeddings: list):
    """Insert documents in a single transaction."""
    conn = get_db_connection()
    cur = conn.cursor()

    inserted = 0
    for chunk, embedding in zip(chunks, embeddings):
        try:
            cur.execute("""
                INSERT INTO documents (title, content, source, category, version, embedding, metadata, source_hash)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s)
                ON CONFLICT (source_hash) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (
                chunk.title,
                chunk.content,
                chunk.source,
                chunk.category,
                chunk.version,
                embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                json.dumps({
                    "url_path": chunk.url_path,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }),
                chunk.id,
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting: {e}")
            conn.rollback()
            continue

    conn.commit()
    cur.close()
    conn.close()
    return inserted

# =============================================================================
# Main Ingestion Pipeline
# =============================================================================

def ingest_blender_manual(docs_path: str, clear_existing: bool = False):
    print("=" * 60)
    print("Blender Manual Ingestion (Local Embeddings)")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Dimensions: {EMBEDDING_DIM}")
    print("=" * 60)

    docs_path = Path(docs_path)
    if not docs_path.exists():
        print(f"ERROR: Path does not exist: {docs_path}")
        return

    # Setup database
    print("\n[1/5] Setting up database...")
    setup_database()

    if clear_existing:
        print("Clearing existing documents...")
        clear_documents("blender_manual")

    # Find HTML files
    print("\n[2/5] Finding HTML files...")
    html_files = [f for f in docs_path.rglob("*.html") if f.is_file()]
    print(f"Found {len(html_files)} HTML files")

    # Load embedding model
    print("\n[3/5] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded!")

    # Process files
    print("\n[4/5] Processing HTML files...")
    all_chunks = []
    for i, html_file in enumerate(html_files):
        for chunk in process_html_file(html_file, docs_path):
            all_chunks.append(chunk)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(html_files)} files, {len(all_chunks)} chunks")

    print(f"  Total: {len(html_files)} files, {len(all_chunks)} chunks")

    if not all_chunks:
        print("No chunks generated!")
        return

    # Generate embeddings and insert in batches
    print(f"\n[5/5] Generating embeddings and inserting into database...")
    total_inserted = 0

    for i in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):
        batch_chunks = all_chunks[i:i + EMBEDDING_BATCH_SIZE]
        batch_texts = [c.content for c in batch_chunks]

        try:
            # Generate embeddings
            embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

            # Insert into database
            inserted = insert_documents_batch(batch_chunks, embeddings)
            total_inserted += inserted

            pct = 100 * (i + len(batch_chunks)) // len(all_chunks)
            print(f"  Progress: {i + len(batch_chunks)}/{len(all_chunks)} ({pct}%) - Inserted: {total_inserted}")
        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents WHERE source = 'blender_manual'")
    print(f"Total documents: {cur.fetchone()[0]}")

    cur.execute("""
        SELECT category, COUNT(*) FROM documents
        WHERE source = 'blender_manual'
        GROUP BY category ORDER BY COUNT(*) DESC LIMIT 10
    """)
    print("\nTop categories:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-path", default="/home/developer/3d-game-ai/backend/rag_data/blender_manual")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    ingest_blender_manual(args.docs_path, args.clear)
