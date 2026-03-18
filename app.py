import os
import streamlit as st
from dotenv import load_dotenv

from utils.predictor import load_models
from utils.retriever import load_rag
from views import (
    tab_assessment,
    tab_comparison,
    tab_performance,
    tab_simulation,
    tab_explainability,
    tab_knowledge,
    tab_logs,
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSentry",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }

    /* Background */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #FAF9F6 !important;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .app-header h1 { margin: 0; font-size: 28px; font-weight: 700; }
    .app-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2a5298;
        margin-bottom: 12px;
    }
    .metric-card.low    { border-left-color: #00C851; }
    .metric-card.medium { border-left-color: #FF8800; }
    .metric-card.high   { border-left-color: #C6613F; }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #1e3c72;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin: 20px 0 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f0f2f6;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button { border-radius: 8px; font-weight: 500; }
    .stNumberInput input, .stSelectbox select { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS AND RAG
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and knowledge base...")
def load_all():
    return load_models(), load_rag()

models, rag = load_all()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🏦 LoanSentry - Intelligent Credit Risk Assessment</h1>
    <p>AI-powered credit risk assessment combining Machine Learning,
       Retrieval-Augmented Generation, and Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🔍 Risk Assessment",
    "👥 Applicant Comparison",
    "📊 Model Performance",
    "📈 Risk Trend Simulation",
    "🧠 Explainability",
    "📚 Knowledge Base",
    "📋 Logs & Feedback"
])

with tabs[0]: tab_assessment.render(models, rag, GROQ_API_KEY)
with tabs[1]: tab_comparison.render(models, GROQ_API_KEY)
with tabs[2]: tab_performance.render()
with tabs[3]: tab_simulation.render(models)
with tabs[4]: tab_explainability.render(models, GROQ_API_KEY)
with tabs[5]: tab_knowledge.render(rag, GROQ_API_KEY)
with tabs[6]: tab_logs.render()