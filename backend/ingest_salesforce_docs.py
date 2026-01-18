#!/usr/bin/env python3
"""
Ingest Salesforce Documentation PDFs into RAG Vector Database
Uses: PyPDF2 for PDF parsing, sentence-transformers for embeddings
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# PDF parsing
try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    os.system("pip install PyPDF2")
    import PyPDF2

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

# Database
try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Installing psycopg2-binary...")
    os.system("pip install psycopg2-binary")
    import psycopg2
    from psycopg2.extras import execute_values

# Configuration
DOCS_DIR = Path(__file__).parent / "salesforce_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast
CHUNK_SIZE = 512  # characters per chunk
CHUNK_OVERLAP = 64  # overlap between chunks
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/salesforce_rag")

# Category to source mapping
CATEGORY_TO_SOURCE = {
    "admin": "admin_guide",
    "sales": "help_docs",
    "marketing": "help_docs",
    "service": "help_docs",
    "automation": "help_docs",
    "development": "apex_guide",
    "security": "admin_guide",
    "data": "admin_guide",
    "analytics": "help_docs",
    "certification": "trailhead",
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  Error reading {pdf_path.name}: {e}")
        return ""

    # Clean up text
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks with metadata."""
    if not text:
        return []

    chunks = []
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "char_start": len("".join(c["text"] for c in chunks)),
                })
                chunk_index += 1
                # Keep overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_index": chunk_index,
            "char_start": len("".join(c["text"] for c in chunks)),
        })

    return chunks


def get_embedding_model():
    """Load the embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(model, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def init_database(conn):
    """Initialize the database tables."""
    cursor = conn.cursor()

    # Create extension if not exists
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS salesforce_documents (
            id SERIAL PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            content TEXT NOT NULL,
            source VARCHAR(50) NOT NULL,
            category VARCHAR(50),
            pdf_filename VARCHAR(255),
            chunk_index INTEGER DEFAULT 0,
            char_start INTEGER DEFAULT 0,
            content_hash VARCHAR(64) UNIQUE,
            embedding vector(384),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_docs_source ON salesforce_documents(source);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_docs_category ON salesforce_documents(category);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_docs_embedding ON salesforce_documents
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)

    # Create full-text search index
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_docs_fts ON salesforce_documents
        USING gin(to_tsvector('english', content));
    """)

    conn.commit()
    print("✓ Database initialized")


def ingest_pdf(conn, model, pdf_path: Path, category: str):
    """Ingest a single PDF into the database."""
    cursor = conn.cursor()

    # Extract text
    print(f"  Extracting text from {pdf_path.name}...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"    ✗ No text extracted")
        return 0

    # Create chunks
    chunks = chunk_text(text)
    if not chunks:
        print(f"    ✗ No chunks created")
        return 0

    print(f"    Created {len(chunks)} chunks")

    # Generate embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(model, chunk_texts)

    # Prepare title from filename
    title_base = pdf_path.stem.replace("_", " ").title()

    # Get source from category
    source = CATEGORY_TO_SOURCE.get(category, "help_docs")

    # Insert into database
    inserted = 0
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        content_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()

        try:
            cursor.execute("""
                INSERT INTO salesforce_documents
                (title, content, source, category, pdf_filename, chunk_index, char_start, content_hash, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
            """, (
                f"{title_base} (Part {i+1})",
                chunk["text"],
                source,
                category,
                pdf_path.name,
                chunk["chunk_index"],
                chunk["char_start"],
                content_hash,
                embedding,
                json.dumps({
                    "pdf_filename": pdf_path.name,
                    "chunk_of_total": f"{i+1}/{len(chunks)}",
                    "ingested_at": datetime.now().isoformat(),
                })
            ))
            inserted += cursor.rowcount
        except Exception as e:
            print(f"    Error inserting chunk {i}: {e}")

    conn.commit()
    return inserted


def main():
    print("=" * 60)
    print("SALESFORCE DOCUMENTATION INGESTION")
    print("=" * 60)

    # Check for PDFs
    if not DOCS_DIR.exists():
        print(f"\n✗ Directory not found: {DOCS_DIR}")
        print("Run 'python download_salesforce_docs.py' first")
        return

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\n✗ No PDFs found in {DOCS_DIR}")
        print("Run 'python download_salesforce_docs.py' first")
        return

    print(f"\nFound {len(pdfs)} PDFs to ingest")
    print(f"Database: {DATABASE_URL}")

    # Load embedding model
    model = get_embedding_model()

    # Connect to database
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("✓ Connected to database")
    except Exception as e:
        print(f"\n✗ Database connection failed: {e}")
        print("\nMake sure PostgreSQL is running with pgvector extension.")
        print("Or set DATABASE_URL environment variable.")
        return

    # Initialize database
    init_database(conn)

    # Process each PDF
    total_chunks = 0
    for pdf_path in sorted(pdfs):
        print(f"\n[{pdf_path.name}]")

        # Extract category from filename (e.g., "admin_basics.pdf" -> "admin")
        category = pdf_path.stem.split("_")[0]

        inserted = ingest_pdf(conn, model, pdf_path, category)
        total_chunks += inserted
        print(f"    ✓ Inserted {inserted} chunks")

    conn.close()

    print("\n" + "=" * 60)
    print(f"INGESTION COMPLETE: {total_chunks} total chunks inserted")
    print("=" * 60)


if __name__ == "__main__":
    main()
