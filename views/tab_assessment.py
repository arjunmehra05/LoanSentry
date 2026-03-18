import os
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

from components.ui import (
    render_risk_gauge, render_shap_chart,
    risk_color, build_input_dict,
    GRADE_OPTIONS, PURPOSE_OPTIONS
)
from utils.predictor import predict, get_shap_values
from utils.retriever import retrieve, build_applicant_query
from utils.explainer import generate_explanation
from utils.synthetic import generate_synthetic_profile
from utils.validator import validate_input
from utils.logger import log_prediction
from utils.feedback import log_feedback
from utils.comparison import get_all_model_probs
from utils.pdf_report import generate_pdf_report


def render(models, rag, groq_api_key=""):

    st.markdown('<div class="section-header">Applicant Details</div>',
                unsafe_allow_html=True)

    # ── Generate buttons ──────────────────────────────────────────────────
    col_gen1, _ = st.columns([1, 6])

    if col_gen1.button("🎲 Random Profile", use_container_width=True):
        _apply_profile(generate_synthetic_profile("random"), "main")
        st.rerun()

    # ── Input form ────────────────────────────────────────────────────────
    _init_defaults("main")

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt  = st.number_input("Loan Amount ($)",    500,    40000, step=500,  key="main_loan_amnt")
        annual_inc = st.number_input("Annual Income ($)", 1000, 1000000,  step=1000, key="main_annual_inc")
        int_rate   = st.number_input("Interest Rate (%)",  1.0,    40.0,  step=0.1,  key="main_int_rate")
    with col2:
        dti        = st.number_input("Debt to Income (%)", 0.0,   100.0,  step=0.1,  key="main_dti")
        fico       = st.number_input("FICO Score",         300,     850,  step=1,    key="main_fico")
        emp_length = st.number_input("Employment Years",     0,      40,  step=1,    key="main_emp_length")
    with col3:
        grade   = st.selectbox("Loan Grade",    GRADE_OPTIONS,   key="main_grade")
        purpose = st.selectbox("Loan Purpose",  PURPOSE_OPTIONS, key="main_purpose")
        term    = st.selectbox("Term (months)", [36, 60],
                               index=0 if st.session_state.get("main_term", 36) == 36 else 1,
                               key="main_term_select")

    input_dict = build_input_dict(
        loan_amnt, annual_inc, int_rate, dti, fico, emp_length,
        GRADE_OPTIONS.index(grade),
        PURPOSE_OPTIONS.index(purpose),
        term
    )

    st.markdown("---")
    assess_btn = st.button("🔍 Run Risk Assessment",
                           type="primary", use_container_width=True)

    # ── Run pipeline when assess button clicked ───────────────────────────
    if assess_btn:
        errors = validate_input(input_dict)
        if errors:
            for e in errors:
                st.error(f"⚠️ {e}")
            st.stop()

        with st.spinner("Running risk assessment..."):
            st.info("⚙️ Step 1/4 - Running model predictions...")
            results = predict(input_dict, models)

            st.info("⚙️ Step 2/4 - Computing SHAP values...")
            shap_vals, feat_names = get_shap_values(results["input_scaled"], models)

            st.info("⚙️ Step 3/4 - Retrieving policy documents...")
            query = build_applicant_query(
                input_dict, results["risk_category"], results["prob_ensemble"])
            docs = retrieve(query, rag, top_k=3)
            log_prediction(input_dict, results)

            st.info("⚙️ Step 4/4 - Generating explanation...")
            explanation = generate_explanation(
                groq_api_key, input_dict, results, shap_vals, feat_names, docs)

            model_probs = get_all_model_probs(results["input_scaled"], models)

            st.session_state.update({
                "last_results":      results,
                "last_input":        input_dict,
                "last_docs":         docs,
                "last_explanation":  explanation,
                "last_shap":         (shap_vals, feat_names),
                "last_model_probs":  model_probs,
                "assessment_done":   True,
            })

    # ── Show results from session state ───────────────────────────────────
    if not st.session_state.get("assessment_done"):
        return

    results     = st.session_state["last_results"]
    input_dict  = st.session_state["last_input"]
    docs        = st.session_state["last_docs"]
    explanation = st.session_state["last_explanation"]
    shap_vals, feat_names = st.session_state["last_shap"]

    st.success("✅ Assessment complete")

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Assessment Results</div>',
                unsafe_allow_html=True)

    col_gauge, col_details = st.columns(2)
    with col_gauge:
        st.plotly_chart(
            render_risk_gauge(results["prob_ensemble"], results["risk_category"]),
            use_container_width=True)

    with col_details:
        rc    = results["risk_category"]
        color = risk_color(rc)
        # New - based on risk category
        rc = results["risk_category"]
        if rc == "Low":
            dec = "✅ Approve"
        elif rc == "Medium":
            dec = "🔶 Review"
        else:
            dec = "⛔ Decline"

        st.markdown(f"""
        <div class="metric-card {rc.lower()}">
            <h2 style="color:{color};margin:0 0 12px">{rc} Risk</h2>
            <table style="width:100%;font-size:15px">
                <tr><td style="padding:4px 0;color:#555">Default Probability</td>
                    <td style="font-weight:600">{results['prob_ensemble']:.1%}</td></tr>
                <tr><td style="padding:4px 0;color:#555">Model Confidence</td>
                    <td style="font-weight:600">{results['confidence']:.1f}%</td></tr>
                <tr><td style="padding:4px 0;color:#555">Recommendation</td>
                    <td style="font-weight:600">{dec}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
        if results["confidence"] < 30:
            st.warning("⚠️ Borderline case - manual review recommended.")
        elif results["confidence"] < 60:
            st.info("ℹ️ Moderate confidence.")
        else:
            st.success("✅ High confidence prediction.")

    # ── Individual model predictions ──────────────────────────────────────
    st.markdown('<div class="section-header">Individual Model Predictions</div>',
                unsafe_allow_html=True)
    model_probs = get_all_model_probs(results["input_scaled"], models)
    cols = st.columns(len(model_probs))
    for col, (name, prob) in zip(cols, model_probs.items()):
        c = risk_color("Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High")
        col.markdown(f"""
        <div style="text-align:center;padding:12px;background:white;border-radius:8px;
             box-shadow:0 2px 6px rgba(0,0,0,0.07);border-top:3px solid {c}">
            <div style="font-size:11px;color:#888;margin-bottom:4px">{name}</div>
            <div style="font-size:22px;font-weight:700;color:{c}">{prob:.1%}</div>
        </div>""", unsafe_allow_html=True)

    # ── SHAP ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance (SHAP)</div>',
                unsafe_allow_html=True)
    st.pyplot(render_shap_chart(shap_vals, feat_names,
                                "Top Features Driving This Prediction"))
    plt.close()

    # ── Explanation ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Risk Explanation</div>',
                unsafe_allow_html=True)
    st.markdown(explanation)

    # ── Feedback ──────────────────────────────────────────────────────────
    st.markdown("**Was this assessment helpful?**")
    fb1, fb2, _ = st.columns([1, 1, 6])
    with fb1:
        if st.button("👍 Yes", key="fb_pos"):
            log_feedback(results["risk_category"],
                         results["prob_ensemble"], "positive")
            st.success("Thanks!")
    with fb2:
        if st.button("👎 No", key="fb_neg"):
            log_feedback(results["risk_category"],
                         results["prob_ensemble"], "negative")
            st.info("Thanks for the feedback.")

    # ── Report ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Download Report</div>',
                unsafe_allow_html=True)

    shap_vals, feat_names = st.session_state["last_shap"]
    pdf_buffer = generate_pdf_report(
        input_dict, results, explanation, shap_vals, feat_names)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name=f"loansense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _init_defaults(prefix):
    defaults = {
        f"{prefix}_loan_amnt":  10000,
        f"{prefix}_annual_inc": 50000,
        f"{prefix}_int_rate":   12.0,
        f"{prefix}_dti":        20.0,
        f"{prefix}_fico":       680,
        f"{prefix}_emp_length": 3,
        f"{prefix}_grade":      "B",
        f"{prefix}_purpose":    "Debt Consolidation",
        f"{prefix}_term":       36,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _apply_profile(profile: dict, prefix: str):
    grade_idx   = min(int(profile.get("grade",   1)), len(GRADE_OPTIONS) - 1)
    purpose_idx = min(int(profile.get("purpose", 2)), len(PURPOSE_OPTIONS) - 1)

    st.session_state[f"{prefix}_loan_amnt"]  = int(profile["loan_amnt"])
    st.session_state[f"{prefix}_annual_inc"] = int(profile["annual_inc"])
    st.session_state[f"{prefix}_int_rate"]   = float(profile["int_rate"])
    st.session_state[f"{prefix}_dti"]        = float(profile["dti"])
    st.session_state[f"{prefix}_fico"]       = int(profile["FICO_AVG"])
    st.session_state[f"{prefix}_emp_length"] = int(profile["emp_length"])
    st.session_state[f"{prefix}_grade"]      = GRADE_OPTIONS[grade_idx]
    st.session_state[f"{prefix}_purpose"]    = PURPOSE_OPTIONS[purpose_idx]
    st.session_state[f"{prefix}_term"]       = int(profile["term"])