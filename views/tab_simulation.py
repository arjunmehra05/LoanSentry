import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.ui import (
    SIMULATE_FEATURES, SIMULATE_FEATURE_MAP,
    GRADE_OPTIONS, PURPOSE_OPTIONS, build_input_dict
)
from utils.predictor import predict
from utils.synthetic import generate_synthetic_profile


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


def _apply_profile(profile, prefix):
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


def render(models):
    st.markdown('<div class="section-header">Risk Trend Simulation</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Set a base profile then simulate how changing one feature "
        "affects the risk score in real time.")

    st.markdown("#### Base Profile")

    # ── Generate buttons ──────────────────────────────────────────────────
    col_gen1, _ = st.columns([1, 6])

    if col_gen1.button("🎲 Random Profile", key="sim_gen_random",
                       use_container_width=True):
        _apply_profile(generate_synthetic_profile("random"), "trend")
        st.rerun()

    # ── Inline form ───────────────────────────────────────────────────────
    _init_defaults("trend")

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt  = st.number_input("Loan Amount ($)",       500,    40000, step=500,  key="trend_loan_amnt")
        annual_inc = st.number_input("Annual Income ($)",    1000, 1000000,  step=1000, key="trend_annual_inc")
        int_rate   = st.number_input("Interest Rate (%)",     1.0,    40.0,  step=0.1,  key="trend_int_rate")
    with col2:
        dti        = st.number_input("Debt to Income (%)",    0.0,   100.0,  step=0.1,  key="trend_dti")
        fico       = st.number_input("FICO Score",            300,     850,  step=1,    key="trend_fico")
        emp_length = st.number_input("Employment Years",        0,      40,  step=1,    key="trend_emp_length")
    with col3:
        grade   = st.selectbox("Loan Grade",    GRADE_OPTIONS,   key="trend_grade")
        purpose = st.selectbox("Loan Purpose",  PURPOSE_OPTIONS, key="trend_purpose")
        term    = st.selectbox("Term (months)", [36, 60],
                               index=0 if st.session_state.get("trend_term", 36) == 36 else 1,
                               key="trend_term_select")

    base = build_input_dict(
        loan_amnt, annual_inc, int_rate, dti, fico, emp_length,
        GRADE_OPTIONS.index(grade),
        PURPOSE_OPTIONS.index(purpose),
        term
    )

    st.markdown("---")
    st.markdown("#### Feature to Simulate")
    feature_label = st.selectbox("Select feature", SIMULATE_FEATURES)
    feat_key, mn, mx, step = SIMULATE_FEATURE_MAP[feature_label]
    values = list(np.arange(mn, mx + step, step))

    if not st.button("▶ Run Simulation", type="primary"):
        return

    progress = st.progress(0, text="Starting simulation...")
    probs    = []
    total    = len(values)

    for i, val in enumerate(values):
        temp = base.copy()
        temp[feat_key] = val
        temp["LOAN_TO_INCOME"] = round(
            temp["loan_amnt"] / (temp["annual_inc"] + 1), 4)
        temp["INSTALLMENT_TO_INCOME"] = round(
            temp["installment"] / (temp["annual_inc"] / 12 + 1), 4)
        res = predict(temp, models)
        probs.append(res["prob_ensemble"])
        progress.progress(
            int((i + 1) / total * 100),
            text=f"Simulating {feature_label}... {i+1}/{total} values")

    progress.empty()
    st.success("✅ Simulation complete")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=[p * 100 for p in probs],
        mode="lines+markers",
        line=dict(color="#2a5298", width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(42,82,152,0.08)",
        name="Default Probability"
    ))
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#e8f5e9", opacity=0.3, line_width=0)
    fig.add_hrect(y0=30, y1=60,  fillcolor="#fff3e0", opacity=0.3, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="#fdf0eb", opacity=0.3, line_width=0)
    fig.add_hline(y=30, line_dash="dash", line_color="#00C851",
                  annotation_text="Low / Medium")
    fig.add_hline(y=60, line_dash="dash", line_color="#C6613F",
                  annotation_text="Medium / High")
    fig.update_layout(
        title=f"Default Probability vs {feature_label}",
        xaxis_title=feature_label,
        yaxis_title="Default Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    min_idx = int(np.argmin(probs))
    max_idx = int(np.argmax(probs))
    c1, c2  = st.columns(2)
    c1.success(
        f"✅ Lowest risk: {feature_label} = {values[min_idx]:,} "
        f"→ {min(probs):.1%} default probability")
    c2.error(
        f"⚠️ Highest risk: {feature_label} = {values[max_idx]:,} "
        f"→ {max(probs):.1%} default probability")