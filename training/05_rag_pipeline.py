"""
LoanSentry - Training Script 5: RAG Pipeline
=============================================
Covers:
- Knowledge base document loading
- Paragraph-based chunking
- Sentence transformer embeddings
- FAISS index building
- Saving index and chunks

Instructions:
1. Place your .txt knowledge base files in the KB_PATH folder
2. Run this script
3. Copy faiss_index.index and chunks.pkl to your app's rag/ folder
"""

import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
KB_PATH   = "knowledge_base/"   # folder containing .txt files
SAVE_PATH = "output/"           # where to save index and chunks

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DOCUMENTS
# ─────────────────────────────────────────────
def load_documents(kb_path):
    documents = []
    for filename in sorted(os.listdir(kb_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "filename": filename,
                "content":  content
            })
    print(f"Loaded {len(documents)} documents")
    return documents

# ─────────────────────────────────────────────
# 2. CHUNK BY PARAGRAPH
# ─────────────────────────────────────────────
def chunk_documents(documents, chunk_size=500, overlap=50):
    """
    Chunks documents by paragraph to preserve structure.
    Much better than word-count chunking for prose summarization.
    """
    chunks = []
    for doc in documents:
        content    = doc["content"]
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        current_chunk  = []
        current_length = 0

        for para in paragraphs:
            para_words = para.split()
            if current_length + len(para_words) > chunk_size and current_chunk:
                chunks.append({
                    "text":     "\n\n".join(current_chunk),
                    "source":   doc["filename"].replace(".txt", "").replace("_", " ").title(),
                    "chunk_id": len(chunks)
                })
                current_chunk  = current_chunk[-2:] if overlap else []
                current_length = sum(len(p.split()) for p in current_chunk)

            current_chunk.append(para)
            current_length += len(para_words)

        if current_chunk:
            chunks.append({
                "text":     "\n\n".join(current_chunk),
                "source":   doc["filename"].replace(".txt", "").replace("_", " ").title(),
                "chunk_id": len(chunks)
            })

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

documents = load_documents(KB_PATH)
chunks    = chunk_documents(documents)

print(f"\nSample chunk:")
print(f"Source: {chunks[0]['source']}")
print(f"Text preview: {chunks[0]['text'][:200]}...")

# ─────────────────────────────────────────────
# 3. GENERATE EMBEDDINGS
# ─────────────────────────────────────────────
print("\nLoading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
texts      = [chunk["text"] for chunk in chunks]
embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)

print(f"Embeddings shape: {embeddings.shape}")

# ─────────────────────────────────────────────
# 4. BUILD FAISS INDEX
# ─────────────────────────────────────────────
embedding_dim = embeddings.shape[1]
index         = faiss.IndexFlatL2(embedding_dim)

faiss.normalize_L2(embeddings)
index.add(embeddings.astype("float32"))

print(f"FAISS index built - total vectors: {index.ntotal}")

# ─────────────────────────────────────────────
# 5. TEST RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(query, index, chunks, embedder, top_k=3):
    query_embedding = embedder.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding.astype("float32"), top_k)
    return [{
        "rank":     i + 1,
        "source":   chunks[idx]["source"],
        "text":     chunks[idx]["text"],
        "distance": round(float(distances[0][i]), 4)
    } for i, idx in enumerate(indices[0])]

test_queries = [
    "what is the maximum DTI ratio for loan approval",
    "high risk borrower low credit score",
    "loan grade and default rate",
    "FICO score impact on default risk",
]

print("\nTesting retrieval...")
for query in test_queries:
    print(f"\nQuery: {query}")
    results = retrieve(query, index, chunks, embedder, top_k=2)
    for r in results:
        print(f"  Rank {r['rank']} | {r['source']} | dist={r['distance']:.4f}")
        print(f"  {r['text'][:120]}...")

# ─────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────
faiss.write_index(index, SAVE_PATH + "faiss_index.index")

with open(SAVE_PATH + "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

np.save(SAVE_PATH + "embeddings.npy", embeddings)

print(f"\nRAG components saved to {SAVE_PATH}")
print(f"  - faiss_index.index ({index.ntotal} vectors)")
print(f"  - chunks.pkl ({len(chunks)} chunks)")
print(f"  - embeddings.npy {embeddings.shape}")

print("\nCopy faiss_index.index and chunks.pkl to your app's rag/ folder.")