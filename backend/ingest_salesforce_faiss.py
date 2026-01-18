#!/usr/bin/env python3
"""
Ingest Salesforce Documentation PDFs into FAISS Vector Database
Uses: PyPDF2 for PDF parsing, sentence-transformers for embeddings, FAISS for vector store
"""

import os
import re
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict
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

# FAISS for vector storage
try:
    import faiss
    import numpy as np
except ImportError:
    print("Installing faiss-cpu...")
    os.system("pip install faiss-cpu numpy")
    import faiss
    import numpy as np

# Configuration
DOCS_DIR = Path(__file__).parent / "salesforce_docs"
DATA_DIR = Path(__file__).parent / "salesforce_data"
DATA_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast
CHUNK_SIZE = 512  # characters per chunk
CHUNK_OVERLAP = 64  # overlap between chunks

# Category to source mapping
CATEGORY_TO_SOURCE = {
    "admin": "admin_guide",
    "sales": "help_docs",
    "marketing": "marketing_cloud",
    "service": "service_cloud",
    "automation": "flow_guide",
    "development": "apex_guide",
    "security": "security_guide",
    "data": "data_guide",
    "analytics": "reports_guide",
    "certification": "cert_guide",
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
        })

    return chunks


def detect_content_type(text: str) -> Dict:
    """Detect if content contains code examples."""
    text_lower = text.lower()
    return {
        "is_apex": "trigger " in text_lower or "class " in text_lower or "public void" in text_lower,
        "is_soql": "select " in text_lower and "from " in text_lower,
        "is_flow": "flow" in text_lower or "process builder" in text_lower,
        "is_lwc": "lightning-" in text_lower or "lwc" in text_lower,
    }


def get_embedding_model():
    """Load the embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    # Force CPU to avoid CUDA compatibility issues
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def create_faiss_index(dimension: int = 384):
    """Create a FAISS index for similarity search."""
    # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    return index


def main():
    print("=" * 60)
    print("SALESFORCE DOCUMENTATION INGESTION (FAISS)")
    print("=" * 60)

    # Check for PDFs
    if not DOCS_DIR.exists():
        print(f"\nDirectory not found: {DOCS_DIR}")
        print("Run 'python download_salesforce_docs.py' first")
        return

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\nNo PDFs found in {DOCS_DIR}")
        print("Run 'python download_salesforce_docs.py' first")
        return

    print(f"\nFound {len(pdfs)} PDFs to ingest")
    print(f"Output directory: {DATA_DIR}")

    # Load embedding model
    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dimension}")

    # Create FAISS index
    index = create_faiss_index(dimension)

    # Storage for documents
    documents = []
    all_embeddings = []

    # Process each PDF
    for pdf_path in sorted(pdfs):
        print(f"\n[{pdf_path.name}]")

        # Extract category from filename (e.g., "admin_basics.pdf" -> "admin")
        category = pdf_path.stem.split("_")[0]
        source = CATEGORY_TO_SOURCE.get(category, "help_docs")

        # Extract text
        print(f"  Extracting text...")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"    No text extracted")
            continue

        # Create chunks
        chunks = chunk_text(text)
        if not chunks:
            print(f"    No chunks created")
            continue

        print(f"    Created {len(chunks)} chunks")

        # Prepare title from filename
        title_base = pdf_path.stem.replace("_", " ").title()

        # Generate embeddings for all chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)

        # Store documents and embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            content_type = detect_content_type(chunk["text"])
            doc_id = len(documents)

            doc = {
                "id": doc_id,
                "title": f"{title_base} (Part {i+1}/{len(chunks)})",
                "content": chunk["text"],
                "source": source,
                "category": category,
                "pdf_filename": pdf_path.name,
                "chunk_index": chunk["chunk_index"],
                **content_type,
                "created_at": datetime.now().isoformat(),
            }
            documents.append(doc)
            all_embeddings.append(embedding)

        print(f"    Added {len(chunks)} documents")

    # Build FAISS index
    if all_embeddings:
        embeddings_array = np.array(all_embeddings).astype('float32')
        index.add(embeddings_array)
        print(f"\nFAISS index built with {index.ntotal} vectors")

    # Save everything
    # Save FAISS index
    faiss_path = DATA_DIR / "salesforce.index"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index: {faiss_path}")

    # Save documents metadata
    docs_path = DATA_DIR / "salesforce_documents.json"
    with open(docs_path, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved documents metadata: {docs_path}")

    # Save embeddings for reloading
    embeddings_path = DATA_DIR / "salesforce_embeddings.npy"
    np.save(embeddings_path, np.array(all_embeddings).astype('float32'))
    print(f"Saved embeddings: {embeddings_path}")

    # Create BM25 index for sparse retrieval
    print("\nBuilding BM25 index for sparse retrieval...")
    try:
        from rank_bm25 import BM25Okapi

        # Tokenize documents
        tokenized_docs = [doc["content"].lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        bm25_path = DATA_DIR / "salesforce_bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump((bm25, tokenized_docs), f)
        print(f"Saved BM25 index: {bm25_path}")
    except ImportError:
        print("Installing rank_bm25...")
        os.system("pip install rank_bm25")
        from rank_bm25 import BM25Okapi
        tokenized_docs = [doc["content"].lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_path = DATA_DIR / "salesforce_bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump((bm25, tokenized_docs), f)
        print(f"Saved BM25 index: {bm25_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total documents: {len(documents)}")
    print(f"Total PDFs processed: {len(pdfs)}")

    # Category breakdown
    categories = {}
    for doc in documents:
        cat = doc["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nDocuments by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print(f"\nData saved to: {DATA_DIR}")
    print("\nNext step: Run the Flask backend with 'python app.py'")


if __name__ == "__main__":
    main()
