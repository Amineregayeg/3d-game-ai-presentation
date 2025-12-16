#!/usr/bin/env python3
"""
Document Ingestion Script for RAG System
Processes Blender documentation HTML files, chunks them, generates embeddings,
and stores them in PostgreSQL with pgvector.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Generator
from dataclasses import dataclass
from html.parser import HTMLParser
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

# Embedding model
from sentence_transformers import SentenceTransformer

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

# Embedding model - using fastest model for CPU inference
# all-MiniLM-L6-v2 is only 22M params, very fast on CPU (384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 512  # Target tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep

# Batch sizes
EMBEDDING_BATCH_SIZE = 32
DB_BATCH_SIZE = 100

# =============================================================================
# HTML Parser
# =============================================================================

class BlenderHTMLParser(HTMLParser):
    """Extract clean text from Blender documentation HTML."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript'}
        self.skip_depth = 0
        self.title = ""
        self.in_title = False
        self.in_content = False
        self.headings = []

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)

        if tag in self.skip_tags:
            self.skip_depth += 1
        elif tag == 'title':
            self.in_title = True
        elif tag in ('main', 'article', 'section', 'div'):
            if 'class' in attrs_dict:
                classes = attrs_dict['class']
                if 'document' in classes or 'body' in classes or 'content' in classes:
                    self.in_content = True
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.text_parts.append(f"\n\n### ")

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
        """Get cleaned text content."""
        text = ''.join(self.text_parts)
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        return text.strip()

# =============================================================================
# Document Processing
# =============================================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of documentation."""
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
    """Extract category from file path."""
    path_lower = filepath.lower()

    category_mapping = {
        'modeling': 'modeling',
        'mesh': 'mesh',
        'sculpt': 'sculpting',
        'animation': 'animation',
        'rigging': 'rigging',
        'armature': 'rigging',
        'render': 'rendering',
        'cycles': 'rendering',
        'eevee': 'rendering',
        'material': 'materials',
        'shader': 'materials',
        'texture': 'materials',
        'node': 'nodes',
        'geometry_node': 'geometry_nodes',
        'composit': 'compositing',
        'video': 'video_editing',
        'sequencer': 'video_editing',
        'physics': 'physics',
        'cloth': 'physics',
        'fluid': 'physics',
        'particle': 'physics',
        'uv': 'uv_mapping',
        'unwrap': 'uv_mapping',
        'script': 'scripting',
        'python': 'scripting',
        'addon': 'addons',
        'interface': 'interface',
        'editor': 'editors',
        'viewport': 'editors',
        'scene': 'scene',
        'object': 'objects',
        'camera': 'camera',
        'light': 'lighting',
        'grease_pencil': 'grease_pencil',
        'motion_track': 'motion_tracking',
        'constraint': 'constraints',
        'modifier': 'modifiers',
        'driver': 'drivers',
        'asset': 'assets',
        'file': 'files',
        'import': 'import_export',
        'export': 'import_export',
        'preferences': 'preferences',
        'getting_started': 'getting_started',
        'troubleshoot': 'troubleshooting',
    }

    for key, category in category_mapping.items():
        if key in path_lower:
            return category

    return 'general'

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    # Simple word-based chunking (approximate tokens)
    words = text.split()

    if len(words) <= chunk_size:
        return [text] if len(words) >= MIN_CHUNK_SIZE else []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        # Try to end at sentence boundary
        chunk_text = ' '.join(chunk_words)

        # Find last sentence end
        for punct in ['. ', '.\n', '? ', '!\n']:
            last_punct = chunk_text.rfind(punct)
            if last_punct > len(chunk_text) * 0.5:  # At least halfway through
                chunk_text = chunk_text[:last_punct + 1]
                end = start + len(chunk_text.split())
                break

        if len(chunk_text.split()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text.strip())

        start = end - overlap
        if start >= len(words):
            break

    return chunks

def process_html_file(filepath: Path, base_path: Path) -> Generator[DocumentChunk, None, None]:
    """Process a single HTML file and yield document chunks."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Parse HTML
    parser = BlenderHTMLParser()
    try:
        parser.feed(html_content)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return

    text = parser.get_text()
    title = parser.title or filepath.stem.replace('_', ' ').title()

    # Skip very short documents
    if len(text.split()) < MIN_CHUNK_SIZE:
        return

    # Get relative path for URL
    rel_path = filepath.relative_to(base_path)
    url_path = str(rel_path).replace('\\', '/')

    # Extract category
    category = extract_category_from_path(str(rel_path))

    # Chunk the text
    chunks = chunk_text(text)

    if not chunks:
        return

    # Generate chunks
    for i, chunk_content in enumerate(chunks):
        # Add title context to first chunk
        if i == 0 and title:
            chunk_content = f"# {title}\n\n{chunk_content}"

        # Generate unique ID
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
# Embedding Generation
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings using BGE-M3."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def generate(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        # Add instruction prefix for BGE models
        prefixed_texts = [f"Represent this document for retrieval: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings

# =============================================================================
# Database Operations
# =============================================================================

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)

def clear_existing_documents(source: str = None):
    """Clear existing documents from database."""
    conn = get_db_connection()
    cur = conn.cursor()

    if source:
        cur.execute("DELETE FROM documents WHERE source = %s", (source,))
        print(f"Cleared documents from source: {source}")
    else:
        cur.execute("DELETE FROM documents")
        print("Cleared all documents")

    conn.commit()
    cur.close()
    conn.close()

def insert_documents(chunks: List[DocumentChunk], embeddings: np.ndarray):
    """Insert documents with embeddings into database."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Prepare data
    data = []
    for chunk, embedding in zip(chunks, embeddings):
        data.append((
            chunk.id,
            chunk.title,
            chunk.content,
            chunk.source,
            chunk.category,
            chunk.version,
            embedding.tolist(),
            json.dumps({
                "url_path": chunk.url_path,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }),
        ))

    # Insert using execute_values for efficiency
    insert_query = """
        INSERT INTO documents (id, title, content, source, category, version, embedding, metadata)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """

    execute_values(cur, insert_query, data, template="""
        (%(id)s, %(title)s, %(content)s, %(source)s, %(category)s, %(version)s,
         %(embedding)s::vector, %(metadata)s::jsonb)
    """.replace('%(', '%s, ').replace(')s', ''))

    # Simpler approach - insert one by one for reliability
    cur.execute("BEGIN")
    for item in data:
        cur.execute("""
            INSERT INTO documents (id, title, content, source, category, version, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """, item)

    conn.commit()
    cur.close()
    conn.close()

def insert_documents_batch(chunks: List[DocumentChunk], embeddings: np.ndarray):
    """Insert documents in batches."""
    conn = get_db_connection()
    cur = conn.cursor()

    inserted = 0
    for chunk, embedding in zip(chunks, embeddings):
        try:
            cur.execute("""
                INSERT INTO documents (id, title, content, source, category, version, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (
                chunk.id,
                chunk.title,
                chunk.content,
                chunk.source,
                chunk.category,
                chunk.version,
                embedding.tolist(),
                json.dumps({
                    "url_path": chunk.url_path,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }),
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting chunk {chunk.id}: {e}")
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
    """Main ingestion pipeline for Blender manual."""
    docs_path = Path(docs_path)

    if not docs_path.exists():
        print(f"Error: Path does not exist: {docs_path}")
        return

    # Find all HTML files
    html_files = list(docs_path.rglob("*.html"))
    print(f"Found {len(html_files)} HTML files")

    if not html_files:
        print("No HTML files found!")
        return

    # Clear existing documents if requested
    if clear_existing:
        clear_existing_documents("blender_manual")

    # Initialize embedding generator
    embedder = EmbeddingGenerator()

    # Process files in batches
    all_chunks = []
    processed_files = 0

    print("\n=== Processing HTML files ===")
    for html_file in html_files:
        for chunk in process_html_file(html_file, docs_path):
            all_chunks.append(chunk)

        processed_files += 1
        if processed_files % 100 == 0:
            print(f"Processed {processed_files}/{len(html_files)} files, {len(all_chunks)} chunks")

    print(f"\nTotal: {processed_files} files, {len(all_chunks)} chunks")

    if not all_chunks:
        print("No chunks generated!")
        return

    # Generate embeddings in batches
    print("\n=== Generating embeddings ===")
    chunk_texts = [c.content for c in all_chunks]

    all_embeddings = []
    for i in range(0, len(chunk_texts), EMBEDDING_BATCH_SIZE):
        batch = chunk_texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedder.generate(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"Generated embeddings: {len(all_embeddings)}/{len(chunk_texts)}")

    all_embeddings = np.array(all_embeddings)

    # Insert into database in batches
    print("\n=== Inserting into database ===")
    total_inserted = 0

    for i in range(0, len(all_chunks), DB_BATCH_SIZE):
        batch_chunks = all_chunks[i:i + DB_BATCH_SIZE]
        batch_embeddings = all_embeddings[i:i + DB_BATCH_SIZE]

        inserted = insert_documents_batch(batch_chunks, batch_embeddings)
        total_inserted += inserted
        print(f"Inserted: {total_inserted}/{len(all_chunks)}")

    print(f"\n=== Ingestion complete ===")
    print(f"Total documents inserted: {total_inserted}")

    # Show category breakdown
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        WHERE source = 'blender_manual'
        GROUP BY category
        ORDER BY count DESC
    """)

    print("\nDocuments by category:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cur.close()
    conn.close()

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Blender documentation into RAG database")
    parser.add_argument("--docs-path", type=str, default="/home/developer/3d-game-ai/backend/rag_data/blender_manual",
                        help="Path to extracted Blender manual HTML files")
    parser.add_argument("--clear", action="store_true", help="Clear existing documents before ingesting")

    args = parser.parse_args()

    ingest_blender_manual(args.docs_path, clear_existing=args.clear)
