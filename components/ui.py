import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]

PURPOSE_OPTIONS = [
    "Car", "Credit Card", "Debt Consolidation", "Home Improvement",
    "Major Purchase", "Medical", "Moving", "Small Business",
    "Vacation", "Other"
]

SIMULATE_FEATURES = [
    "Annual Income ($)", "Debt to Income Ratio (%)", "Employment Years",
    "FICO Score", "Interest Rate (%)", "Loan Amount ($)",
]

SIMULATE_FEATURE_MAP = {
    "Annual Income ($)":        ("annual_inc",  10000, 200000, 5000),
    "Debt to Income Ratio (%)": ("dti",         0,     80,     1),
    "Employment Years":         ("emp_length",  0,     30,     1),
    "FICO Score":               ("FICO_AVG",    300,   850,    5),
    "Interest Rate (%)":        ("int_rate",    1.0,   35.0,   0.5),
    "Loan Amount ($)":          ("loan_amnt",   1000,  40000,  1000),
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_color(risk_category):
    return {
        "Low":    "#00C851",
        "Medium": "#FF8800",
        "High":   "#C6613F"
    }.get(risk_category, "#888888")


def render_risk_gauge(prob, risk_category):
    color = risk_color(risk_category)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": f"Default Probability<br>"
                       f"<span style='color:{color};font-size:18px'>"
                       f"{risk_category} Risk</span>",
               "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  30],  "color": "#e8f5e9"},
                {"range": [30, 60],  "color": "#fff3e0"},
                {"range": [60, 100], "color": "#fdf0eb"},
            ],
            "threshold": {
                "line":      {"color": "black", "width": 3},
                "thickness": 0.75,
                "value":     prob * 100
            }
        },
        number={"suffix": "%", "font": {"size": 28}}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_input_dict(loan_amnt, annual_inc, int_rate, dti, fico,
                     emp_length, grade_idx, purpose_idx, term):
    funded_amnt           = loan_amnt
    installment           = round(
        (loan_amnt * (int_rate / 1200)) /
        (1 - (1 + int_rate / 1200) ** -36), 2)
    loan_to_income        = round(loan_amnt / (annual_inc + 1), 4)
    installment_to_income = round(installment / (annual_inc / 12 + 1), 4)

    return {
        "loan_amnt":             loan_amnt,
        "funded_amnt":           funded_amnt,
        "term":                  term,
        "int_rate":              int_rate,
        "installment":           installment,
        "grade":                 grade_idx,
        "sub_grade":             grade_idx * 5,
        "emp_length":            emp_length,
        "home_ownership":        0,
        "annual_inc":            annual_inc,
        "verification_status":   1,
        "purpose":               purpose_idx,
        "dti":                   dti,
        "delinq_2yrs":           0,
        "FICO_AVG":              fico,
        "inq_last_6mths":        0,
        "open_acc":              5,
        "pub_rec":               0,
        "revol_bal":             5000,
        "revol_util":            30.0,
        "total_acc":             10,
        "initial_list_status":   0,
        "application_type":      0,
        "mort_acc":              0,
        "pub_rec_bankruptcies":  0,
        "LOAN_TO_INCOME":        loan_to_income,
        "INSTALLMENT_TO_INCOME": installment_to_income,
        "FUNDED_RATIO":          1.0,
    }


def _set_default(key, value):
    """Only sets session state if key not already present."""
    if key not in st.session_state:
        st.session_state[key] = value


def render_input_form(prefix, defaults=None):
    """
    Renders the input form. Session state keys take priority over defaults.
    Call _load_profile_to_state() before st.rerun() to pre-fill from a profile.
    """
    d = defaults or {}

    # Set defaults only if not already in session state
    _set_default(f"{prefix}_loan_amnt",  int(d.get("loan_amnt",  10000)))
    _set_default(f"{prefix}_annual_inc", int(d.get("annual_inc", 50000)))
    _set_default(f"{prefix}_int_rate",   float(d.get("int_rate",  12.0)))
    _set_default(f"{prefix}_dti",        float(d.get("dti",       20.0)))
    _set_default(f"{prefix}_fico",       int(d.get("FICO_AVG",   680)))
    _set_default(f"{prefix}_emp_length", int(d.get("emp_length",   3)))
    _set_default(f"{prefix}_grade",
        GRADE_OPTIONS[min(int(d.get("grade", 1)), len(GRADE_OPTIONS)-1)])
    _set_default(f"{prefix}_purpose",
        PURPOSE_OPTIONS[min(int(d.get("purpose", 2)), len(PURPOSE_OPTIONS)-1)])
    _set_default(f"{prefix}_term",       int(d.get("term", 36)))

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt  = st.number_input(
            "Loan Amount ($)", 500, 40000, step=500,
            key=f"{prefix}_loan_amnt")
        annual_inc = st.number_input(
            "Annual Income ($)", 1000, 1000000, step=1000,
            key=f"{prefix}_annual_inc")
        int_rate   = st.number_input(
            "Interest Rate (%)", 1.0, 40.0, step=0.1,
            key=f"{prefix}_int_rate")

    with col2:
        dti        = st.number_input(
            "Debt to Income Ratio (%)", 0.0, 100.0, step=0.1,
            key=f"{prefix}_dti")
        fico       = st.number_input(
            "FICO Score", 300, 850, step=1,
            key=f"{prefix}_fico")
        emp_length = st.number_input(
            "Employment Years", 0, 40, step=1,
            key=f"{prefix}_emp_length")

    with col3:
        grade   = st.selectbox(
            "Loan Grade", GRADE_OPTIONS,
            key=f"{prefix}_grade")
        purpose = st.selectbox(
            "Loan Purpose", PURPOSE_OPTIONS,
            key=f"{prefix}_purpose")
        term_val = st.session_state.get(f"{prefix}_term", 36)
        term    = st.selectbox(
            "Term (months)", [36, 60],
            index=0 if term_val == 36 else 1,
            key=f"{prefix}_term_select_{prefix}")

    grade_idx   = GRADE_OPTIONS.index(grade)
    purpose_idx = PURPOSE_OPTIONS.index(purpose)

    return build_input_dict(
        loan_amnt, annual_inc, int_rate, dti,
        fico, emp_length, grade_idx, purpose_idx, term)


def render_shap_chart(shap_vals, feat_names,
                      title="SHAP Feature Importance"):
    vals    = shap_vals[0] if len(np.array(shap_vals).shape) > 1 \
              else shap_vals
    top_idx = np.argsort(np.abs(vals))[-12:][::-1]
    top_feats = [feat_names[i] for i in top_idx]
    top_vals  = [vals[i] for i in top_idx]
    colors    = ["#C6613F" if v > 0 else "#00C851" for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_feats[::-1], top_vals[::-1],
            color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Value (Red = increases default risk)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig