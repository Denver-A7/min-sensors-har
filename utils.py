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
BLUE = "#5B9BFF"
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


# Give a plotly chart the app font and very faint grid lines. Data and labels stay the same.
def style_chart(fig):
    fig.update_layout(font=dict(family="Inter, sans-serif"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")


# Inject the CSS for the modern look (fonts, background, cards, tabs, alerts). Design only.
def apply_style():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --blue: #5B9BFF;
    --yellow: #F2B705;
    --soft-text: rgba(255, 255, 255, 0.70);
    --card-bg: rgba(255, 255, 255, 0.04);
    --card-border: rgba(91, 155, 255, 0.25);
    --card-border-hover: rgba(91, 155, 255, 0.75);
}

/* Background: navy base, blue glow top right, yellow glow bottom left, faint grid. */
.stApp {
    background:
        radial-gradient(900px 600px at 100% 0%, rgba(91, 155, 255, 0.22), transparent 60%),
        radial-gradient(700px 500px at 0% 100%, rgba(242, 183, 5, 0.07), transparent 60%),
        linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px) 0 0 / 40px 40px,
        linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px) 0 0 / 40px 40px,
        #0E1A4A;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }

/* Body text. Icon fonts are not touched. */
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
.stApp textarea, .stApp td, .stApp th, .stApp [data-baseweb] {
    font-family: 'Inter', 'Source Sans', sans-serif;
}
.stApp p, .stApp li, .stApp label {
    font-size: 1rem;
    line-height: 1.65;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: var(--soft-text);
    font-size: 0.9rem;
}

/* Headings. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
}
.stApp h1 {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.stApp h1::after, .hero-title::after {
    content: "";
    display: block;
    width: 48px;
    height: 4px;
    margin-top: 0.6rem;
    border-radius: 4px;
    background: var(--yellow);
}
.stApp h2, .stApp h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* Sidebar: slightly darker navy gradient and a yellow bar on the active link. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1440 0%, #070E2E 100%);
    border-right: 1px solid rgba(91, 155, 255, 0.15);
}
[data-testid="stSidebar"] > div:first-child { background: transparent; }
.stApp [data-testid="stSidebarNavLink"] {
    border-radius: 10px;
    border-left: 4px solid transparent;
    transition: background-color 0.2s ease;
}
.stApp [data-testid="stSidebarNavLink"]:hover { background: rgba(91, 155, 255, 0.12); }
.stApp [data-testid="stSidebarNavLink"][aria-current="page"] {
    border-left-color: var(--yellow);
    background: rgba(91, 155, 255, 0.16);
}

/* Glass cards: bordered containers, metrics and expanders. */
.stApp [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"],
.stApp [data-testid="stMetric"],
.stApp [data-testid="stExpander"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.stApp [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:hover,
.stApp [data-testid="stMetric"]:hover,
.stApp [data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    border-color: var(--card-border-hover);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
}
.stApp [data-testid="stMetric"] { padding: 1rem 1.25rem; }
.stApp [data-testid="stExpander"] details {
    border: none;
    background: transparent;
}

/* Metrics: small uppercase label, large mono value. Long values wrap instead of being cut. */
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--soft-text);
}
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 500;
    line-height: 1.2;
    white-space: normal;
    overflow-wrap: anywhere;
    text-overflow: clip;
}

/* Tabs: pills, blue when active, yellow underline. */
.stApp [role="tablist"] { gap: 0.4rem; }
.stApp [data-testid="stTab"] {
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    background: var(--card-bg);
    border: 1px solid transparent;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}
.stApp [data-testid="stTab"]:hover { background: rgba(91, 155, 255, 0.14); }
.stApp [data-testid="stTab"][aria-selected="true"] {
    background: rgba(91, 155, 255, 0.28);
    border-color: rgba(91, 155, 255, 0.6);
}
.stApp [data-testid="stTab"] p { font-weight: 500; }
.stApp .react-aria-SelectionIndicator {
    background-color: var(--yellow);
    height: 3px;
    border-radius: 3px;
}
.stApp [role="tablist"]::after { background-color: rgba(255, 255, 255, 0.08); }
.stApp [data-testid="stTabPanel"] { padding-top: 1rem; }

/* Buttons: rounded blue gradient with dark text for good contrast. */
.stApp button[data-testid="stBaseButton-secondary"],
.stApp button[data-testid="stBaseButton-primary"] {
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #5B9BFF 0%, #3E7DE6 100%);
    box-shadow: 0 4px 14px rgba(91, 155, 255, 0.35);
    transition: filter 0.2s ease, transform 0.2s ease;
}
.stApp button[data-testid="stBaseButton-secondary"] p,
.stApp button[data-testid="stBaseButton-primary"] p {
    color: #0B1537;
    font-weight: 600;
}
.stApp button[data-testid="stBaseButton-secondary"]:hover,
.stApp button[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.15);
    transform: translateY(-1px);
}

/* Messages: rounded, with a 4px colored bar on the left. The background stays as it is. */
.stApp [data-testid="stAlertContainer"] {
    border-radius: 12px;
    border-left: 4px solid var(--blue);
}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: #3DD68C; }
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color: var(--yellow); }
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { border-left-color: #FF6B6B; }
.stApp [data-testid="stAlertContainer"] p { color: rgba(255, 255, 255, 0.95); }

/* Tables: rounded corners. */
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--card-border);
}
.stApp [data-testid="stTable"] td,
.stApp [data-testid="stTable"] th {
    font-family: 'IBM Plex Mono', monospace;
}

/* Home page hero card. */
.hero {
    background: linear-gradient(135deg, rgba(91, 155, 255, 0.20), rgba(255, 255, 255, 0.04));
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}
.hero-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
    color: #FFFFFF;
}
.hero-text {
    margin: 1rem 0 1.2rem 0;
    font-size: 1.1rem;
    line-height: 1.65;
    color: rgba(255, 255, 255, 0.85);
}
.hero-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.hero-pill {
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(242, 183, 5, 0.55);
    background: rgba(242, 183, 5, 0.12);
    color: var(--yellow);
    font-size: 0.9rem;
    font-weight: 600;
}

/* Hide only the "Made with Streamlit" footer. */
footer, [data-testid="stFooter"] { visibility: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )
