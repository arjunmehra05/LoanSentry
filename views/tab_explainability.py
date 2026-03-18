import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from utils.synthetic import generate_synthetic_profile


def render(models, groq_api_key=""):
    st.markdown('<div class="section-header">Global Feature Importance</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Analyse SHAP values across multiple synthetic profiles to "
        "understand which features drive default risk globally.")

    n_samples = st.slider("Number of synthetic profiles", 10, 100, 30)

    if not st.button("Generate Global SHAP Analysis", type="primary"):
        return

    feat_names = models["feature_names"]

    progress = st.progress(0, text="Generating synthetic profiles...")
    profiles = []
    for i in range(n_samples):
        profiles.append(generate_synthetic_profile("random"))
        progress.progress(
            int((i + 1) / n_samples * 40),
            text=f"Generating profiles... {i+1}/{n_samples}")

    progress.progress(50, text="Scaling features...")
    scaled = np.array([
        models["scaler"].transform(
            np.array([[p.get(f, 0) for f in feat_names]]))[0]
        for p in profiles
    ])

    progress.progress(70, text="Computing SHAP values...")
    explainer = shap.TreeExplainer(models["xgb_booster"])
    shap_vals = explainer.shap_values(scaled)

    progress.progress(100, text="Rendering plots...")
    progress.empty()
    st.success(f"✅ Analysis complete across {n_samples} synthetic profiles")

    # ── Side by side plots ────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### SHAP Summary Plot")
        fig1, _ = plt.subplots(figsize=(6, 5))
        shap.summary_plot(shap_vals, scaled, feature_names=feat_names,
                          show=False, max_display=15)
        st.pyplot(fig1)
        plt.close()

    with col_right:
        st.markdown("#### Mean Absolute SHAP Values")
        mean_shap = pd.DataFrame({
            "Feature":           feat_names,
            "Mean |SHAP Value|": np.abs(shap_vals).mean(axis=0)
        }).sort_values("Mean |SHAP Value|", ascending=False).head(15)

        fig2, ax = plt.subplots(figsize=(6, 5))
        ax.barh(mean_shap["Feature"], mean_shap["Mean |SHAP Value|"],
                color="#2a5298", edgecolor="white")
        ax.set_title("Top 15 Most Important Features",
                     fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── AI Interpretation ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">AI Interpretation</div>',
                unsafe_allow_html=True)

    top_features = mean_shap.head(8)
    shap_summary = "\n".join([
        f"- {row['Feature']}: mean |SHAP| = {row['Mean |SHAP Value|']:.4f}"
        for _, row in top_features.iterrows()
    ])

    mean_shap_signed = shap_vals.mean(axis=0)
    top_positive = [(feat_names[i], mean_shap_signed[i])
                    for i in np.argsort(mean_shap_signed)[-3:][::-1]]
    top_negative = [(feat_names[i], mean_shap_signed[i])
                    for i in np.argsort(mean_shap_signed)[:3]]

    pos_str = "\n".join([f"- {f}: {v:+.4f}" for f, v in top_positive])
    neg_str = "\n".join([f"- {f}: {v:+.4f}" for f, v in top_negative])

    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)

            prompt = f"""You are a senior credit risk analyst interpreting global SHAP feature importance results from a loan default prediction model trained on {n_samples} synthetic profiles.

TOP FEATURES BY MEAN |SHAP VALUE| (overall importance):
{shap_summary}

FEATURES THAT INCREASE DEFAULT RISK ON AVERAGE (positive SHAP):
{pos_str}

FEATURES THAT REDUCE DEFAULT RISK ON AVERAGE (negative SHAP):
{neg_str}

Provide your response in exactly this format:

### Global Model Insights
[2-3 sentences summarizing what the model considers most important overall and why this makes sense from a credit risk perspective]

---
### Top Risk Drivers
- **[Feature]** - [1 sentence explaining why this feature strongly predicts default]
- **[Feature]** - [1 sentence explaining why this feature strongly predicts default]
- **[Feature]** - [1 sentence explaining why this feature strongly predicts default]

---
### Protective Factors
- **[Feature]** - [1 sentence explaining how this feature reduces default risk]
- **[Feature]** - [1 sentence explaining how this feature reduces default risk]

---
### Key Takeaway
[1-2 sentences with the most important insight a credit analyst should take from this analysis]

Keep it concise and grounded in credit risk domain knowledge."""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior credit risk analyst with deep knowledge "
                            "of machine learning model interpretation. Be concise and "
                            "reference specific feature names from the data provided."
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
            st.warning(f"Groq unavailable: {e}. Showing rule-based summary.")

    # ── Rule-based fallback ───────────────────────────────────────────────
    top1 = mean_shap.iloc[0]["Feature"]
    top2 = mean_shap.iloc[1]["Feature"]
    top3 = mean_shap.iloc[2]["Feature"]

    st.markdown(f"""
### Global Model Insights
Across {n_samples} profiles, the model relies most heavily on **{top1}**, **{top2}**,
and **{top3}** to predict default risk. These features consistently show the
largest impact on predictions regardless of the applicant profile.

### Top Risk Drivers
- **{top_positive[0][0]}** - increases default probability on average across profiles
- **{top_positive[1][0]}** - second strongest positive predictor of default
- **{top_positive[2][0]}** - third strongest positive predictor of default

### Protective Factors
- **{top_negative[0][0]}** - reduces default probability on average
- **{top_negative[1][0]}** - second strongest protective factor

### Key Takeaway
Focus underwriting attention on **{top1}** and **{top2}** as they are the
most influential factors in this model's predictions.
""")