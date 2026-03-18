import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render():
    st.markdown('<div class="section-header">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    comparison_path = "models/full_comparison.csv"
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)
    else:
        df = pd.DataFrame({
            "Model":    ["Logistic Regression", "Random Forest",
                         "XGBoost", "Neural Network", "Weighted Ensemble"],
            "Accuracy": [0.6949, 0.7698, 0.7562, 0.6438, 0.6745],
            "F1 Score": [0.3838, 0.3827, 0.4359, 0.4374, 0.4406],
            "ROC-AUC":  [0.6716, 0.7157, 0.7325, 0.7169, 0.7228]
        })
        st.info(
            "Showing training results. Copy full_comparison.csv "
            "from Google Drive into models/ for live data.")

    st.dataframe(
        df.style.highlight_max(
            subset=["Accuracy", "F1 Score", "ROC-AUC"], color="#d4edda"),
        use_container_width=True, hide_index=True)

    colors  = ["#2a5298", "#1e7e34", "#c82333", "#e67e22", "#8e44ad", "#16a085", "#d35400"]
    metrics = [c for c in ["Accuracy", "F1 Score", "ROC-AUC"] if c in df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df["Model"], df[metric],
                      color=colors[:len(df)], edgecolor="white", linewidth=0.5)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=40)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()