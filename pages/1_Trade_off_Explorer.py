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

# Choose the test type. LOSO is disabled when that column has no numbers.
loso_empty = "loso_acc_mean" not in df.columns or df["loso_acc_mean"].isna().all()
test_type = st.sidebar.radio("Test type", ["K fold", "LOSO"], disabled=loso_empty)
if loso_empty:
    st.sidebar.caption("LOSO is not available yet. The loso_acc_mean column is empty.")
    test_type = "K fold"
acc_col = "kfold_acc_mean" if test_type == "K fold" else "loso_acc_mean"

# Targets for the two status checks.
target1 = st.sidebar.slider("Target 1: full set accuracy", 50, 100, 90, 1)
allowed_drop = st.sidebar.slider("Allowed drop in points", 0.0, 20.0, 5.0, 0.5)

# ---------- Prepare the data ----------
# Keep only the chosen models and rows that have an accuracy value.
data = df[df["model"].isin(selected_models)].dropna(subset=[acc_col]).copy()
if data.empty:
    st.info("Pick at least one model in the sidebar to see the chart.")
    st.stop()

# Accuracy in percent and the group display name.
data["acc_pct"] = data[acc_col] * 100
data["group_name"] = data["group"].map(utils.GROUPS).fillna(data["group"])

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
title = f"Accuracy vs. number of features ({test_type})"
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
            customdata=sub["group_name"],
            marker=dict(size=12, color=color),
            hovertemplate="%{customdata}<br>Features: %{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
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
    title=dict(text=title, font=dict(size=24)),
    xaxis_title="Number of features",
    yaxis_title=f"Accuracy in % ({test_type})",
    font=dict(size=16),
    legend=dict(title="Model"),
    height=600,
)
utils.style_chart(fig)

st.plotly_chart(fig, width="stretch")

# ---------- Table ----------
st.subheader("All results")

# Build a readable table with the same numbers as the chart.
table = pd.DataFrame(
    {
        "Model": data["model"].map(utils.MODELS),
        "Group": data["group_name"],
        "Features": data["n_features"],
        "Accuracy": data[acc_col].map(utils.pct),
        "Drop from full set (points)": data["drop_points"].round(1),
    }
).sort_values(["Model", "Features"], ascending=[True, False])
st.dataframe(table, hide_index=True, width="stretch")

# ---------- Status boxes ----------
st.subheader("Targets")

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

    # Target 2: the best reduced group stays within the allowed drop of the full set.
    reduced = sub[sub["group"] != "both"]
    if reduced.empty:
        col2.warning("No reduced groups for this model. Cannot check Target 2.")
        continue
    best = reduced.loc[reduced["acc_pct"].idxmax()]
    best_drop = both_acc - best["acc_pct"]
    best_text = f"Best reduced group: {best['group_name']} ({best['n_features']} features), {drop_text(best_drop)}."
    if best_drop <= allowed_drop:
        col2.success(f"Target 2 met (allowed drop {allowed_drop} points). {best_text}")
    else:
        col2.warning(f"Target 2 not met (allowed drop {allowed_drop} points). {best_text}")

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
