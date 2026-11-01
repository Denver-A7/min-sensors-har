# Home page of the app.
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

# Explain each page of the app in one line.
st.subheader("Pages in this app")
st.markdown(
    """
- **Trade-off Explorer:** compare accuracy and number of features for each model and sensor group.
- **Try a Prediction:** pick a test window and see what a model predicts.
- **New Person Test:** see how the models do on a person they never saw.
- **Confusion Explorer:** see which activities get mixed up.
- **Sensor Tiers:** compare sensor choices by accuracy and power use.
- **About and Limits:** read how this project was done and what its limits are.
"""
)

# Explain the words used for accuracy on every page.
with st.container(border=True):
    st.subheader("How to read the numbers")
    st.markdown(
        """
- **Accuracy:** the share of windows the model labels correctly.
- **K fold:** the standard test. The same person can appear in training and test.
- **LOSO:** leave one subject out. The model is tested on a person it never saw.
"""
    )
