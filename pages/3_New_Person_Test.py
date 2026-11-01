# New Person Test page: how well does a model work on a volunteer it has never seen (LOSO)?
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import utils

# Set the browser tab title and icon.
st.set_page_config(page_title="New Person Test", page_icon="🧑", layout="wide")

# Show the page title.
st.title("New Person Test")

# Warn if one of the result files is test data.
utils.show_banner(["results_summary.csv", "loso_by_volunteer.csv"])

# Read both tables. Each call stops the page if its file is missing.
summary = utils.load_csv("results_summary.csv")
loso = utils.load_csv("loso_by_volunteer.csv").dropna(subset=["accuracy"])

# Keep only the (model, group) pairs that have LOSO rows.
pairs = loso[["model", "group"]].drop_duplicates()
if pairs.empty:
    st.info("LOSO has not been run yet. Please come back after the LOSO results are added.")
    st.stop()

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

# Models that have LOSO rows, in our usual order.
model_options = [code for code in utils.MODELS if code in set(pairs["model"])]
model = st.sidebar.selectbox("Model", model_options, format_func=lambda c: utils.MODELS[c])

# Groups that have LOSO rows for the chosen model, in our usual order.
group_options = [code for code in utils.GROUPS if code in set(pairs.loc[pairs["model"] == model, "group"])]
group = st.sidebar.selectbox("Feature group", group_options, format_func=lambda c: utils.GROUPS[c])

# ---------- Summary numbers ----------
st.caption("LOSO holds out one whole volunteer, trains on the others, and tests on that volunteer.")
st.subheader(f"{utils.MODELS[model]} with {utils.GROUPS[group]}")

# The summary row for this choice. It can be missing, so we check.
rows = summary[(summary["model"] == model) & (summary["group"] == group)]
kfold_acc = None


# Show a fraction as a percent, or "n/a" if there is no number.
def pct_or_na(value):
    return utils.pct(value) if pd.notna(value) else "n/a"


# Three boxes: K fold accuracy, LOSO accuracy, and the gap between them.
if rows.empty:
    st.info("This choice is not in results_summary.csv, so the three numbers are not shown.")
else:
    row = rows.iloc[0]
    if pd.notna(row["kfold_acc_mean"]):
        kfold_acc = row["kfold_acc_mean"]
    col1, col2, col3 = st.columns(3)
    col1.metric("K fold accuracy", pct_or_na(row["kfold_acc_mean"]))
    col2.metric("LOSO accuracy", pct_or_na(row["loso_acc_mean"]))
    col3.metric("Gap in points", f"{row['gap_points']:.1f}" if pd.notna(row["gap_points"]) else "n/a")

# ---------- Chart: one bar per volunteer ----------
data = loso[(loso["model"] == model) & (loso["group"] == group)].sort_values("volunteer").copy()
data["acc_pct"] = data["accuracy"] * 100
# Text labels on the x axis make the bars equal and evenly spaced.
data["volunteer_label"] = data["volunteer"].astype(int).astype(str)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=data["volunteer_label"],
        y=data["acc_pct"],
        name="LOSO accuracy",
        marker_color=utils.BLUE,
        customdata=data["n_samples"],
        hovertemplate="Volunteer %{x}<br>Accuracy: %{y:.1f}%<br>Samples: %{customdata}<extra></extra>",
    )
)

# Yellow line at the k fold accuracy, to compare with each volunteer.
if kfold_acc is not None:
    fig.add_hline(
        y=kfold_acc * 100,
        line_dash="dash",
        line_color=utils.YELLOW,
        line_width=3,
        annotation_text=f"K fold accuracy: {utils.pct(kfold_acc)}",
        annotation_position="top left",
        annotation_font=dict(size=16, color=utils.YELLOW),
    )

# Dark template with transparent backgrounds so it matches the app theme.
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title=dict(text="LOSO accuracy for each volunteer", font=dict(size=24)),
    xaxis_title="Volunteer number",
    yaxis_title="Accuracy in %",
    xaxis=dict(type="category"),
    yaxis=dict(range=[0, 105]),
    font=dict(size=16),
    height=550,
)
st.plotly_chart(fig, width="stretch")

# ---------- One volunteer ----------
st.subheader("Look at one volunteer")

# Pick a volunteer from the ones that have rows for this choice.
volunteer = st.selectbox("Volunteer", data["volunteer_label"].tolist())
one = data[data["volunteer_label"] == volunteer].iloc[0]
col_a, col_b = st.columns(2)
col_a.metric("Accuracy", utils.pct(one["accuracy"]))
col_b.metric("Number of samples", f"{int(one['n_samples'])}")

# Best and worst volunteer in one sentence.
best = data.loc[data["accuracy"].idxmax()]
worst = data.loc[data["accuracy"].idxmin()]
st.write(
    f"The best volunteer is {best['volunteer_label']} with {utils.pct(best['accuracy'])}. "
    f"The worst volunteer is {worst['volunteer_label']} with {utils.pct(worst['accuracy'])}."
)

# ---------- Note ----------
st.info(
    "An earlier study found a drop from 89% to 76% (13 points) "
    "when it used LOSO instead of standard cross validation [3]."
)
