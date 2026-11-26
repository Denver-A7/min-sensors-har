# Trade-off Explorer page: accuracy vs. number of features for each model and sensor group.
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import utils

# Set the browser tab title and icon, and use the full page width for the chart.
st.set_page_config(page_title="Trade-off Explorer", page_icon="⚖️", layout="wide")

# Add the custom look (fonts, background, cards).
utils.apply_style()

# Show the page title.
st.title("Trade-off Explorer")

# Warn if the results file is test data.
utils.show_banner(["results_summary.csv"])
is_test = not utils.is_real("results_summary.csv")

# Read the summary table. Stops the page if the file is missing.
df = utils.load_csv("results_summary.csv")

# One color per model. White is the third color because we only have two main colors.
MODEL_COLORS = {"logreg": utils.BLUE, "rf": utils.YELLOW, "svm": "#FFFFFF"}

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

# One checkbox per model. All on by default.
st.sidebar.subheader("Models")
selected_models = [
    code for code, name in utils.MODELS.items() if st.sidebar.checkbox(name, value=True)
]

# Choose the test type. New person (LOSO) is first, so it is the default.
# LOSO is disabled when that column has no numbers.
LOSO_LABEL = "New person (LOSO)"
KFOLD_LABEL = "5 fold cross validation (optimistic: the same person is in train and test)"
loso_empty = "loso_acc_mean" not in df.columns or df["loso_acc_mean"].isna().all()
test_type = st.sidebar.radio("Test type", [LOSO_LABEL, KFOLD_LABEL], disabled=loso_empty)
if loso_empty:
    st.sidebar.caption("LOSO is not available yet. The loso_acc_mean column is empty.")
    test_type = KFOLD_LABEL
acc_col = "loso_acc_mean" if test_type == LOSO_LABEL else "kfold_acc_mean"

# Short name of the test, for chart and heading titles.
test_short = LOSO_LABEL if test_type == LOSO_LABEL else "5 fold"

# Targets for the two status checks. Target 1 is also the accuracy floor for Target 2.
target1 = st.sidebar.slider("Target 1: full set accuracy", 50, 100, utils.RULE_MIN_ACC, 1)
allowed_drop = st.sidebar.slider("Allowed drop in points", 0.0, 20.0, utils.RULE_MAX_DROP, 0.5)

# ---------- Prepare the data ----------
# Keep only the chosen models and rows that have an accuracy value.
data = df[df["model"].isin(selected_models)].dropna(subset=[acc_col]).copy()
if data.empty:
    st.info("Pick at least one model in the sidebar to see the chart.")
    st.stop()

# What each group removes from the full set: sensors or calculations.
REMOVES = {
    "accel_only": "Sensor",
    "gyro_only": "Sensor",
    "time_only": "Calculation",
    "freq_only": "Calculation",
    "both": "Full set",
}

# Accuracy in percent, the group display name, what it removes, and a label with both.
data["acc_pct"] = data[acc_col] * 100
data["group_name"] = data["group"].map(utils.GROUPS).fillna(data["group"])
data["removes"] = data["group"].map(REMOVES).fillna("")
data["group_label"] = data.apply(
    lambda r: f"{r['group_name']} ({r['removes'].lower()})" if r["removes"] == "Full set"
    else f"{r['group_name']} (removes: {r['removes'].lower()})",
    axis=1,
)

# Full set ("both") accuracy per model, used for the drop and the shaded bands.
full_acc = data[data["group"] == "both"].set_index("model")["acc_pct"]

# Drop from the full set in percentage points (positive = worse than the full set).
data["drop_points"] = data["model"].map(full_acc) - data["acc_pct"]


# Say a drop in simple words. A negative drop means the group is better than the full set.
def drop_text(drop):
    if drop < 0:
        return f"{-drop:.1f} points higher than the full set"
    return f"a drop of {drop:.1f} points"


# ---------- Chart ----------
title = f"Accuracy vs. number of features ({test_short})"
if is_test:
    title += " (TEST DATA)"

fig = go.Figure()

for code in selected_models:
    sub = data[data["model"] == code].sort_values("n_features")
    if sub.empty:
        continue
    color = MODEL_COLORS[code]

    # Shaded band: from (full set accuracy - allowed drop) up to the full set accuracy.
    if code in full_acc.index:
        fig.add_hrect(
            y0=full_acc[code] - allowed_drop,
            y1=full_acc[code],
            fillcolor=color,
            opacity=0.12,
            line_width=0,
        )

    # Markers only, because the groups are separate setups and not a trend. Group name shows on hover.
    fig.add_trace(
        go.Scatter(
            x=sub["n_features"],
            y=sub["acc_pct"],
            mode="markers",
            name=utils.MODELS[code],
            customdata=sub[["group_label", "removes"]],
            marker=dict(size=12, color=color),
            hovertemplate=(
                "%{customdata[0]}<br>Removes: %{customdata[1]}<br>"
                "Features: %{x}<br>Accuracy: %{y:.1f}%<extra></extra>"
            ),
        )
    )

# Dashed line at the Target 1 value.
fig.add_hline(
    y=target1,
    line_dash="dash",
    line_color="#FFFFFF",
    annotation_text=f"Target 1: {target1}%",
    annotation_position="top left",
    annotation_font=dict(size=14, color="#FFFFFF"),
)

# Dark template with transparent backgrounds so it matches the app theme.
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title=dict(text=title, font=dict(size=18)),
    xaxis_title="Number of features",
    yaxis_title=f"Accuracy in % ({test_short})",
    font=dict(size=13),
    legend=dict(
    title=dict(text=title, font=dict(size=18)),
    xaxis_title="Number of features",
    yaxis_title=f"Accuracy in % ({test_short})",
    font=dict(size=13),
    legend=dict(
        title=dict(text="Model", font=dict(size=11), side="top center"),
        font=dict(size=11),
        orientation="h",
        yanchor="top",
        y=-0.3,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(b=170),
    height=600,
)
utils.style_chart(fig)

st.plotly_chart(fig, width="stretch", config=utils.MOBILE_CHART_CONFIG)

# ---------- Table ----------
st.subheader("All results")

# Build a readable table with the same numbers as the chart.
table = pd.DataFrame(
    {
        "Model": data["model"].map(utils.MODELS),
        "Group": data["group_label"],
        "Removes": data["removes"],
        "Features": data["n_features"],
        "Accuracy": data[acc_col].map(utils.pct),
        "Drop from full set (points)": data["drop_points"].round(1),
    }
).sort_values(["Model", "Features"], ascending=[True, False])
st.dataframe(table, hide_index=True, width="stretch")

# ---------- Status boxes ----------
st.subheader(f"Targets ({test_short})")

for code in selected_models:
    sub = data[data["model"] == code]
    if sub.empty:
        continue
    st.markdown(f"**{utils.MODELS[code]}**")
    col1, col2 = st.columns(2)

    # Without a "both" row we cannot check either target.
    if code not in full_acc.index:
        col1.warning("No 'both' row for this model. Cannot check the targets.")
        continue
    both_acc = full_acc[code]

    # Target 1: the full set reaches the slider value.
    if both_acc >= target1:
        col1.success(f"Target 1 met: full set accuracy is {both_acc:.1f}% (target {target1}%).")
    else:
        col1.warning(f"Target 1 not met: full set accuracy is {both_acc:.1f}% (target {target1}%).")

    # Target 2: the best reduced group is at least Target 1 and within the allowed drop of the full set.
    reduced = sub[sub["group"] != "both"]
    if reduced.empty:
        col2.warning("No reduced groups for this model. Cannot check Target 2.")
        continue
    best = reduced.loc[reduced["acc_pct"].idxmax()]
    best_drop = both_acc - best["acc_pct"]
    best_text = f"Best reduced group: {best['group_label']} ({best['n_features']} features), {drop_text(best_drop)}."
    rule_text = f"(at least {target1}% and a drop of at most {allowed_drop} points)"
    reliable, borderline = utils.check_reliable(best["acc_pct"], both_acc, target1, allowed_drop)
    if reliable and borderline:
        col2.warning(f"Target 2 met, but borderline {rule_text}. {best_text}")
    elif reliable:
        col2.success(f"Target 2 met {rule_text}. {best_text}")
    else:
        col2.warning(f"Target 2 not met {rule_text}. {best_text}")

# ---------- Largest drop between neighbors ----------
# Sort each model's groups by n_features and find the biggest loss between two neighbors.
largest = None
for code in selected_models:
    sub = data[data["model"] == code].sort_values("n_features").reset_index(drop=True)
    for i in range(len(sub) - 1):
        bigger, smaller = sub.loc[i + 1], sub.loc[i]
        loss = bigger["acc_pct"] - smaller["acc_pct"]
        if largest is None or loss > largest[0]:
            largest = (loss, code, bigger, smaller)

if largest is not None:
    loss, code, bigger, smaller = largest
    st.write(
        f"The largest drop between neighbors is for {utils.MODELS[code]}: going from "
        f"{bigger['group_name']} ({bigger['n_features']} features) to "
        f"{smaller['group_name']} ({smaller['n_features']} features) loses {loss:.1f} points."
    )
