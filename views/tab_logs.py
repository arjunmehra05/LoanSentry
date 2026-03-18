import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render():
    st.markdown('<div class="section-header">Prediction Logs & Feedback</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Log")
        log_path = "logs/predictions_log.csv"
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            if not log_df.empty:
                st.dataframe(log_df.tail(20),
                             use_container_width=True, hide_index=True)
                if "risk_category" in log_df.columns:
                    dist      = log_df["risk_category"].value_counts()
                    color_map = {
                        "Low": "#00C851", "Medium": "#FF8800", "High": "#CC0000"}
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.pie(dist.values, labels=dist.index, autopct="%1.1f%%",
                           colors=[color_map.get(k, "#888") for k in dist.index],
                           startangle=90)
                    ax.set_title("Risk Distribution")
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("No predictions logged yet.")
        else:
            st.info("No prediction log found.")

    with col2:
        st.markdown("#### Feedback Log")
        fb_path = "feedback_log.csv"
        if os.path.exists(fb_path):
            fb_df = pd.read_csv(fb_path)
            if not fb_df.empty:
                st.dataframe(fb_df.tail(20),
                             use_container_width=True, hide_index=True)
                pos = (fb_df["feedback"] == "positive").sum()
                neg = (fb_df["feedback"] == "negative").sum()
                m1, m2 = st.columns(2)
                m1.metric("👍 Positive", int(pos))
                m2.metric("👎 Negative", int(neg))
            else:
                st.info("No feedback logged yet.")
        else:
            st.info("No feedback log found.")

    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        if os.path.exists("logs/predictions_log.csv"):
            with open("logs/predictions_log.csv", "rb") as f:
                st.download_button(
                    "📥 Download Prediction Log", f,
                    "predictions_log.csv", "text/csv")
    with d2:
        if os.path.exists("feedback_log.csv"):
            with open("feedback_log.csv", "rb") as f:
                st.download_button(
                    "📥 Download Feedback Log", f,
                    "feedback_log.csv", "text/csv")