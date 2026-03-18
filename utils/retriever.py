import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RAG_PATH = "rag/"

def load_rag():
    index = faiss.read_index(RAG_PATH + "faiss_index.index")
    with open(RAG_PATH + "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return {"index": index, "chunks": chunks, "embedder": embedder}


def retrieve(query, rag, top_k=3):
    embedder = rag["embedder"]
    index    = rag["index"]
    chunks   = rag["chunks"]

    query_embedding = embedder.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding.astype("float32"), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank":     i + 1,
            "source":   chunks[idx]["source"],
            "text":     chunks[idx]["text"],
            "distance": round(float(distances[0][i]), 4)
        })

    return results


def build_applicant_query(input_dict, risk_category, prob):
    return f"""
    Loan applicant risk assessment:
    Risk category: {risk_category}
    Default probability: {prob:.1%}
    Annual income: ${input_dict.get('annual_inc', 0):,.0f}
    Loan amount: ${input_dict.get('loan_amnt', 0):,.0f}
    DTI ratio: {input_dict.get('dti', 0):.1f}%
    FICO score: {input_dict.get('FICO_AVG', 0):.0f}
    Employment years: {input_dict.get('emp_length', 0):.0f}
    Interest rate: {input_dict.get('int_rate', 0):.1f}%
    Loan purpose: {input_dict.get('purpose', 'unknown')}
    Loan grade: {input_dict.get('grade', 'unknown')}
    Loan to income ratio: {input_dict.get('LOAN_TO_INCOME', 0):.2f}
    """