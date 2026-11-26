# Home page of the app.
import pandas as pd
import streamlit as st

import utils

# Set the browser tab title and use the full page width.
st.set_page_config(page_title="How Few Sensors?", layout="wide")

# Add the custom look (fonts, background, cards).
utils.apply_style()

# Show the hero card at the top of the home page.
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">How Few Sensors Does a Wearable Need?</div>'
    '<div class="hero-text">We test how much sensor data a wearable really needs to know what you are doing.</div>'
    '<div class="hero-pills">'
    '<span class="hero-pill">Public data</span>'
    '<span class="hero-pill">3 models</span>'
    '<span class="hero-pill">Leave one person out test</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Warn if the result files or models are test data.
utils.show_banner(["results_summary.csv", "models"])

# Show the main question in a highlighted box.
st.info(
    "What is the smallest set of sensor features that can still classify "
    "human activity reliably, and does it still work for a person the model "
    "has never seen?"
)

# Show four facts about the data set.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Volunteers", "30")
col2.metric("Activities", "6")
col3.metric("Windows", "10,299")
col4.metric("Features", "561")

# Show the main results for logistic regression. The rule check uses the LOSO numbers.
st.subheader("Main results")
with st.container(key="main_results"):
    summary_path = utils.BASE_DIR / "results" / "results_summary.csv"
    if not summary_path.exists():
        st.info("The file results/results_summary.csv is missing, so the main results are not shown.")
    else:
        summary = utils.load_csv("results_summary.csv")
        needed = ["model", "group", "kfold_acc_mean", "loso_acc_mean"]
        rows = None
        if all(c in summary.columns for c in needed):
            rows = summary[summary["model"] == "logreg"].drop_duplicates(subset=["group"]).set_index("group")
        if rows is None or "both" not in rows.index or "accel_only" not in rows.index:
            st.info("The main results need the logistic regression rows for both and accel_only.")
        else:
            # Row 1: 5 fold accuracy for three groups.
            st.caption(f"{utils.MODELS['logreg']}, 5 fold accuracy")
            k1, k2, k3 = st.columns(3)
            for col, group in zip((k1, k2, k3), ("both", "accel_only", "gyro_only")):
                value = rows["kfold_acc_mean"].get(group)
                col.metric(utils.GROUPS[group], utils.pct(value) if pd.notna(value) else "n/a")

            # Row 2: LOSO numbers for the accelerometer only, and the rule check.
            st.caption(f"{utils.MODELS['logreg']}, new person (LOSO), {utils.GROUPS['accel_only']}")
            loso_acc = rows.loc["accel_only", "loso_acc_mean"] * 100
            loso_full = rows.loc["both", "loso_acc_mean"] * 100
            reliable, borderline = utils.check_reliable(loso_acc, loso_full)
            if borderline:
                rule_text = "Reliable, but borderline"
            elif reliable:
                rule_text = "Reliable"
            else:
                rule_text = "Not reliable"
            l1, l2, l3 = st.columns(3)
            l1.metric("LOSO accuracy", f"{loso_acc:.1f}%")
            l2.metric("Points below the full set", f"{loso_full - loso_acc:.1f}")
            l3.container(key="rule_check").metric("Rule check", rule_text)

# Explain each page of the app in one line.
st.subheader("Pages in this app")
st.markdown(
    """
- **Trade-off Explorer:** compare accuracy and number of features for each model and sensor group.
- **Try a Prediction:** pick a test window and see what a model predicts.
- **New Person Test:** see how the models do on a person they never saw.
- **Confusion Explorer:** see which activities get mixed up.
- **Sensor Tiers:** the cheapest setup that is reliable for a new person.
- **About and Limits:** read how this project was done and what its limits are.
"""
)

# Explain the words used for accuracy on every page.
with st.container(border=True):
    st.subheader("How to read the numbers")
    st.markdown(
        """
- **Accuracy:** the share of windows the model labels correctly.
- **5 fold:** the standard test. The same person can appear in training and test.
- **LOSO:** leave one subject out. The model is tested on a person it never saw.
"""
    )
