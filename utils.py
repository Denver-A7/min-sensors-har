# Shared helper functions for all pages.
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Folder of this file, so paths work from any page.
BASE_DIR = Path(__file__).parent

# Model code -> display name.
MODELS = {
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "svm": "SVM (support vector machine)",
}

# Feature group code -> display name, in the order we show them.
GROUPS = {
    "both": "Both sensors (all 561 features)",
    "accel_only": "Accelerometer only",
    "gyro_only": "Gyroscope only",
    "time_only": "Time domain only",
    "freq_only": "Frequency domain only",
}

# The six activities, in the order we show them.
ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# Main chart colors.
BLUE = "#1F4E9C"
YELLOW = "#F2B705"


# Read run_status.json. If it is missing, say that nothing is real.
def load_status():
    path = BASE_DIR / "results" / "run_status.json"
    if not path.exists():
        return {"real": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# True only if the status file marks this name as real.
def is_real(name):
    return load_status().get("real", {}).get(name) is True


# Show a yellow warning at the top of the page if any file is not real data.
def show_banner(names):
    fake = [name for name in names if not is_real(name)]
    if fake:
        st.warning(
            "TEST DATA. These are not our results.\n\n"
            "Files that are not real: " + ", ".join(fake)
        )


# Read a CSV file with caching. Called only after we know the file exists.
@st.cache_data
def _read_csv(path):
    return pd.read_csv(path)


# Read results/<filename>. If it is missing, show an error and stop the page.
def load_csv(filename):
    path = BASE_DIR / "results" / filename
    if not path.exists():
        st.error(f"The file results/{filename} is missing. Please add it and reload.")
        st.stop()
    return _read_csv(str(path))


# Read the table that says which feature belongs to which group.
def load_groups():
    path = BASE_DIR / "data_out" / "feature_groups.csv"
    if not path.exists():
        st.error("The file data_out/feature_groups.csv is missing. Please add it and reload.")
        st.stop()
    return _read_csv(str(path))


# Return the list of column numbers that belong to one group.
def get_group_columns(group_code):
    groups = load_groups()
    column = "in_" + group_code
    if column not in groups.columns:
        st.error(f"Unknown feature group: {group_code}")
        st.stop()
    return groups.loc[groups[column].astype(str).str.lower() == "true", "col_index"].tolist()


# Load one trained model from disk. Cached so it loads only once.
@st.cache_resource
def _load_joblib(path):
    return joblib.load(path)


# Load models/demo_<model>_<group>.joblib. If it is missing, show an error and stop.
def load_model(model_code, group_code):
    path = BASE_DIR / "models" / f"demo_{model_code}_{group_code}.joblib"
    if not path.exists():
        st.error(f"The model file models/{path.name} is missing. Please add it and reload.")
        st.stop()
    return _load_joblib(str(path))


# Read the demo features and the info about each row, cached.
@st.cache_data
def _read_demo_files(x_path, meta_path):
    return pd.read_csv(x_path), pd.read_csv(meta_path)


# Return (X, meta) from demo_data/. If a file is missing, show an error and stop.
def load_demo_data():
    x_path = BASE_DIR / "demo_data" / "X_test_demo.csv.gz"
    meta_path = BASE_DIR / "demo_data" / "meta_test_demo.csv"
    for path in (x_path, meta_path):
        if not path.exists():
            st.error(f"The file demo_data/{path.name} is missing. Please add it and reload.")
            st.stop()
    return _read_demo_files(str(x_path), str(meta_path))


# Show a fraction as a percent with one decimal, e.g. 0.923 -> "92.3%".
def pct(x):
    return f"{x * 100:.1f}%"
