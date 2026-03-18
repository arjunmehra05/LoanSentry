import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.ui import render_input_form, render_risk_gauge, risk_color
from utils.predictor import predict, get_shap_values
from utils.synthetic import generate_synthetic_profile
from utils.validator import validate_input

FEATURE_LABELS = {
    "annual_inc":             "Annual Income",
    "loan_amnt":              "Loan Amount",
    "int_rate":               "Interest Rate",
    "dti":                    "Debt-to-Income Ratio",
    "FICO_AVG":               "FICO Score",
    "emp_length":             "Employment Length",
    "revol_util":             "Revolving Utilization",
    "LOAN_TO_INCOME":         "Loan-to-Income Ratio",
    "INSTALLMENT_TO_INCOME":  "Installment-to-Income Ratio",
    "delinq_2yrs":            "Delinquencies (2yr)",
    "pub_rec":                "Public Records",
    "inq_last_6mths":         "Credit Inquiries (6mo)",
}


def render(models, groq_api_key=""):
    st.markdown('<div class="section-header">Compare Two Applicants</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Applicant A")
        if st.button("🎲 Generate Profile A", key="gen_a"):
            st.session_state["synth_a"] = generate_synthetic_profile("random")
            st.rerun()
        input_a = render_input_form("cmp_a", st.session_state.get("synth_a", {}))

    with col_b:
        st.markdown("#### Applicant B")
        if st.button("🎲 Generate Profile B", key="gen_b"):
            st.session_state["synth_b"] = generate_synthetic_profile("random")
            st.rerun()
        input_b = render_input_form("cmp_b", st.session_state.get("synth_b", {}))

    if not st.button("🔍 Compare", type="primary", use_container_width=True):
        return

    errors = validate_input(input_a) + validate_input(input_b)
    if errors:
        for e in errors:
            st.error(e)
        return

    with st.spinner("Comparing applicants..."):
        st.info("⚙️ Running predictions for both applicants...")
        res_a = predict(input_a, models)
        res_b = predict(input_b, models)

        st.info("⚙️ Computing SHAP values...")
        sv_a, fn_a = get_shap_values(res_a["input_scaled"], models)
        sv_b, fn_b = get_shap_values(res_b["input_scaled"], models)

    st.success("✅ Comparison complete")

    # ── Gauges ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Comparison Results</div>',
                unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            render_risk_gauge(res_a["prob_ensemble"], res_a["risk_category"]),
            use_container_width=True)
        ca = risk_color(res_a["risk_category"])
        st.markdown(
            f"<h4 style='color:{ca};text-align:center'>"
            f"Applicant A - {res_a['risk_category']} Risk</h4>",
            unsafe_allow_html=True)

    with g2:
        st.plotly_chart(
            render_risk_gauge(res_b["prob_ensemble"], res_b["risk_category"]),
            use_container_width=True)
        cb = risk_color(res_b["risk_category"])
        st.markdown(
            f"<h4 style='color:{cb};text-align:center'>"
            f"Applicant B - {res_b['risk_category']} Risk</h4>",
            unsafe_allow_html=True)

    # ── Metrics table ─────────────────────────────────────────────────────
    compare_df = pd.DataFrame({
        "Metric": ["Default Probability", "Risk Category",
                   "Confidence", "Recommendation"],
        "Applicant A": [
            f"{res_a['prob_ensemble']:.1%}", res_a["risk_category"],
            f"{res_a['confidence']:.1f}%",
            "Decline" if res_a["prediction"] == 1 else "Approve"],
        "Applicant B": [
            f"{res_b['prob_ensemble']:.1%}", res_b["risk_category"],
            f"{res_b['confidence']:.1f}%",
            "Decline" if res_b["prediction"] == 1 else "Approve"],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # ── SHAP side by side ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">SHAP Comparison</div>',
                unsafe_allow_html=True)

    vals_a  = sv_a[0] if len(np.array(sv_a).shape) > 1 else sv_a
    vals_b  = sv_b[0] if len(np.array(sv_b).shape) > 1 else sv_b
    top_idx = np.argsort(np.abs(vals_a) + np.abs(vals_b))[-10:][::-1] 

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_a = ["#CC0000" if vals_a[i] > 0 else "#00C851" for i in top_idx]
    colors_b = ["#CC0000" if vals_b[i] > 0 else "#00C851" for i in top_idx]

    axes[0].barh([fn_a[i] for i in top_idx[::-1]],
                 [vals_a[i] for i in top_idx[::-1]], color=colors_a[::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Applicant A", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh([fn_b[i] for i in top_idx[::-1]],
                 [vals_b[i] for i in top_idx[::-1]], color=colors_b[::-1])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Applicant B", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── AI Interpretation ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">AI Interpretation</div>',
                unsafe_allow_html=True)

    _render_comparison_interpretation(
        input_a, input_b, res_a, res_b,
        vals_a, vals_b, fn_a, groq_api_key
    )


def _render_comparison_interpretation(input_a, input_b, res_a, res_b,
                                       vals_a, vals_b, feat_names,
                                       groq_api_key):

    def top_shap_summary(vals, names, n=3):
        indices = np.abs(vals).argsort()[-n:][::-1]
        return "\n".join([
            f"- {FEATURE_LABELS.get(names[i], names[i])}: "
            f"{'increases' if vals[i] > 0 else 'reduces'} risk "
            f"(SHAP: {vals[i]:+.4f})"
            for i in indices
        ])

    def profile_summary(inp, res):
        return (
            f"FICO: {inp.get('FICO_AVG', 0):.0f}, "
            f"DTI: {inp.get('dti', 0):.1f}%, "
            f"Income: ${inp.get('annual_inc', 0):,.0f}, "
            f"Loan: ${inp.get('loan_amnt', 0):,.0f}, "
            f"Employment: {inp.get('emp_length', 0):.0f} yrs, "
            f"Rate: {inp.get('int_rate', 0):.1f}%, "
            f"Default Prob: {res['prob_ensemble']:.1%}, "
            f"Risk: {res['risk_category']}"
        )

    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)

            prompt = f"""You are a senior credit risk analyst comparing two loan applications.

APPLICANT A:
{profile_summary(input_a, res_a)}
Top SHAP factors:
{top_shap_summary(vals_a, feat_names)}

APPLICANT B:
{profile_summary(input_b, res_b)}
Top SHAP factors:
{top_shap_summary(vals_b, feat_names)}

Provide your response in exactly this format:

### Overall Comparison
[2-3 sentences comparing the two applicants and which is lower risk and why]

---
### Key Differences
- **[Factor]** - [1 sentence explaining how they differ and which is better]
- **[Factor]** - [1 sentence explaining how they differ and which is better]
- **[Factor]** - [1 sentence explaining how they differ and which is better]

---
### Recommendation
**Applicant A:** [✅ Approve / 🔶 Approve with Conditions / ⛔ Decline] - [1 sentence reason]
**Applicant B:** [✅ Approve / 🔶 Approve with Conditions / ⛔ Decline] - [1 sentence reason]

---
### Who Should Be Approved?
[1-2 sentences with a clear, direct recommendation if only one can be approved]

Keep it concise and data-driven."""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior credit risk analyst. "
                            "Be concise, direct, and reference specific numbers."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.2
            )
            st.markdown(response.choices[0].message.content)
            return

        except Exception as e:
            st.warning(f"Groq unavailable: {e}. Showing rule-based comparison.")

    # ── Rule-based fallback ───────────────────────────────────────────────
    prob_a = res_a["prob_ensemble"]
    prob_b = res_b["prob_ensemble"]
    better = "A" if prob_a < prob_b else "B"
    worse  = "B" if better == "A" else "A"

    fico_a = input_a.get("FICO_AVG", 0)
    fico_b = input_b.get("FICO_AVG", 0)
    dti_a  = input_a.get("dti", 0)
    dti_b  = input_b.get("dti", 0)

    st.markdown(f"""
### Overall Comparison
Applicant {better} presents lower default risk ({min(prob_a, prob_b):.1%}) 
compared to Applicant {worse} ({max(prob_a, prob_b):.1%}).

### Key Differences
- **FICO Score** - A: {fico_a:.0f} vs B: {fico_b:.0f}.
  {'Applicant A has a stronger credit profile.' if fico_a > fico_b else 'Applicant B has a stronger credit profile.'}
- **DTI Ratio** - A: {dti_a:.1f}% vs B: {dti_b:.1f}%.

### Recommendation
**Applicant A:** {"✅ Approve" if res_a["prediction"] == 0 else "⛔ Decline"}  
**Applicant B:** {"✅ Approve" if res_b["prediction"] == 0 else "⛔ Decline"}
""")