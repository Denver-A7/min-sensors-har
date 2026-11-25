# About and Limits page: key facts, project summary, limits, sources, team and file status.
import pandas as pd
import streamlit as st

import utils

# Set the browser tab title and icon.
st.set_page_config(page_title="About and Limits", page_icon="ℹ️", layout="wide")

# Add the custom look (fonts, background, cards).
utils.apply_style()

# Show the page title.
st.title("About and Limits")

# Show the four big dataset facts in one row.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Volunteers", "30")
col2.metric("Activities", "6")
col3.metric("Windows", "10,299")
col4.metric("Features", "561")

# The five project sentences, one per line.
PROJECT_LINES = [
    "We ask how few sensors a wearable needs to recognize 6 daily activities.",
    "We use the public UCI HAR dataset (30 volunteers, 10,299 windows, 561 features).",
    "We compare 3 models: logistic regression, random forest, and SVM.",
    "We test 5 feature groups and 3 tests, including leave one person out.",
    "We link the results to the cost and power of the sensors.",
]

# Each limit is a (short title, one simple sentence) pair.
LIMITS = [
    ("One dataset", "All our results come from one dataset, so they may not hold for other data."),
    ("Phone worn at the waist", "A wearable on the wrist or chest can give different signals."),
    ("Only 30 volunteers", "A small group may not show how all people move."),
    ("Six simple activities", "Real life has many more activities than these six."),
    ("Pre calculated features", "The dataset authors made the 561 features, so we remove groups of features, not real sensors."),
    ("5 fold can be too optimistic", "The same person can be in training and test, so the score can look better than it is."),
    ("Power from one chip", "The gyroscope only power is calculated, not read from the datasheet."),
    ("Summary features only", "The models do not read the order of the raw readings."),
    ("Sensor current only", "It is not the processor, radio or whole device."),
    ("Accelerometer only is borderline", "It is reliable, but only just."),
]

# Each source is one line, written exactly as given.
SOURCES = [
    "[1] Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). "
    "A public domain dataset for human activity recognition using smartphones. "
    "Proceedings of ESANN 2013.",
    "[2] UCI Machine Learning Repository. Human Activity Recognition Using Smartphones "
    "(dataset 240). https://archive.ics.uci.edu/dataset/240",
    "[3] Rehman, S. U., Ali, A., Khan, A. M., and Okpala, C. (2024). Human activity "
    "recognition: a comparative study of validation methods and impact of feature "
    "extraction in wearable sensors. Algorithms, 17(12), 556. "
    "https://doi.org/10.3390/a17120556",
    "[4] Nuran, A. S. (2026). A SHAP-guided framework for sensor feature minimization in "
    "human activity recognition. Journal of Electrical Engineering and Informatics "
    "(JEENI), 3(2), 150-163. https://doi.org/10.59562/jeeni.v3i2.12788",
    "[5] STMicroelectronics. LSM6DSL iNEMO inertial module: always-on 3D accelerometer and "
    "3D gyroscope (datasheet, Rev 7, September 2017). "
    "https://www.st.com/en/mems-and-sensors/lsm6dsl.html",
    "[6] Cortes, C., and Vapnik, V. (1995). Support-vector networks. "
    "Machine Learning, 20(3), 273-297.",
    "[7] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. "
    "Journal of Machine Learning Research, 12, 2825-2830.",
]


# Build a File / Status table from the status dictionary.
def build_status_table(real):
    rows = [
        {"File": name, "Status": "Real" if value is True else "Test"}
        for name, value in real.items()
    ]
    return pd.DataFrame(rows, columns=["File", "Status"])


# Show the file status table, or a friendly message if the status file is missing or empty.
def show_status_table():
    status_path = utils.BASE_DIR / "results" / "run_status.json"
    if not status_path.exists():
        st.info("The file results/run_status.json is missing, so we cannot tell which files are real.")
        return
    try:
        real_files = utils.load_status().get("real", {})
    except ValueError:
        st.info("The file results/run_status.json could not be read.")
        return
    if not real_files:
        st.info("The file results/run_status.json has no file list yet.")
        return
    st.dataframe(build_status_table(real_files), hide_index=True, use_container_width=True)


# Four tabs, one for each part of the page.
tab_project, tab_limits, tab_sources, tab_team = st.tabs(
    ["The project", "Limits", "Sources", "Files"]
)

# Tab 1: the project in 5 short numbered lines.
with tab_project:
    st.subheader("The project in 5 sentences")
    for number, line in enumerate(PROJECT_LINES, start=1):
        with st.container(border=True):
            st.markdown(f"**{number}.** {line}")

# Tab 2: the limits as cards in a 2 column grid.
with tab_limits:
    st.subheader("Limits")
    grid = st.columns(2)
    for i, (title, reason) in enumerate(LIMITS):
        with grid[i % 2].container(border=True):
            st.markdown(f"**{title}**")
            st.write(reason)

# Tab 3: the numbered source list. A backslash keeps "[1]" from being read as a link.
with tab_sources:
    st.subheader("Sources")
    for source in SOURCES:
        st.markdown("\\" + source)

# Tab 4: team, course, and which result files are real.
with tab_team:
    st.subheader("Which result files are real")
    show_status_table()
