# Confusion Explorer page: see which activities the models mix up.
import plotly.graph_objects as go
import streamlit as st

import utils

# Set the browser tab title and icon.
st.set_page_config(page_title="Confusion Explorer", page_icon="🔍", layout="wide")

# Add the custom look (fonts, background, cards).
utils.apply_style()

# Show the page title.
st.title("Confusion Explorer")

# Warn if any of the files is test data.
utils.show_banner(["confusion_matrices.csv", "per_class_metrics.csv"])

# Read both tables. Stops the page if a file is missing.
conf = utils.load_csv("confusion_matrices.csv")
metrics = utils.load_csv("per_class_metrics.csv")

if conf.empty:
    st.info("The confusion matrix file has no rows yet.")
    st.stop()

# Names of the two test types.
SPLIT_NAMES = {"kfold": "K fold", "loso": "LOSO"}

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

# Models that exist in the file, in our usual order.
model_options = [m for m in utils.MODELS if m in set(conf["model"])]
if not model_options:
    st.info("No known models were found in the confusion matrix file.")
    st.stop()
model = st.sidebar.selectbox(
    "Model", model_options, format_func=lambda code: utils.MODELS[code]
)

# Only the groups that exist for the chosen model.
group_options = [
    g for g in utils.GROUPS if g in set(conf.loc[conf["model"] == model, "group"])
]
if not group_options:
    st.info("There are no results for this model.")
    st.stop()
group = st.sidebar.selectbox(
    "Feature group", group_options, format_func=lambda code: utils.GROUPS[code]
)

# Only the test types that exist for the chosen model and group.
available_splits = set(
    conf.loc[(conf["model"] == model) & (conf["group"] == group), "split"]
)
split_options = [s for s in SPLIT_NAMES if s in available_splits]
if not split_options:
    st.info("There are no results for this choice.")
    st.stop()
split = st.sidebar.selectbox(
    "Test type", split_options, format_func=lambda code: SPLIT_NAMES[code]
)

# Toggle between counts and row percent.
show_percent = st.sidebar.toggle("Show row percent", value=False)

# ---------- Confusion matrix ----------
# Keep only the rows for this choice.
chosen = conf[
    (conf["model"] == model) & (conf["group"] == group) & (conf["split"] == split)
]
if chosen.empty:
    st.info("There are no results for this choice.")
    st.stop()

# Rows = true activity, columns = predicted activity. Missing cells become 0.
matrix = chosen.pivot_table(
    index="true_activity",
    columns="predicted_activity",
    values="count",
    aggfunc="sum",
    fill_value=0,
).reindex(index=utils.ACTIVITIES, columns=utils.ACTIVITIES, fill_value=0)

# Row percent: each row is divided by its own total. Empty rows stay 0.
row_totals = matrix.sum(axis=1).replace(0, float("nan"))
percent_matrix = matrix.div(row_totals, axis=0).fillna(0) * 100

if show_percent:
    z = percent_matrix
    text_format = "%{z:.1f}%"
    hover_format = "%{z:.1f}%"
    bar_title = "Row percent"
else:
    z = matrix
    text_format = "%{z}"
    hover_format = "%{z}"
    bar_title = "Count"

st.subheader(
    f"{utils.MODELS[model]} | {utils.GROUPS[group]} | {SPLIT_NAMES[split]}"
)

# Heatmap with a blue color scale, from dark navy to blue.
fig = go.Figure(
    go.Heatmap(
        z=z.values,
        x=utils.ACTIVITIES,
        y=utils.ACTIVITIES,
        colorscale=[[0, "#0B1B3A"], [1, utils.BLUE]],
        texttemplate=text_format,
        textfont=dict(size=16),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: "
        + hover_format
        + "<extra></extra>",
        colorbar=dict(title=dict(text=bar_title, font=dict(size=16))),
    )
)

# Dark template with transparent backgrounds. First activity at the top.
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=16),
    height=650,
    xaxis=dict(title=dict(text="Predicted activity", font=dict(size=18)), tickangle=-30),
    yaxis=dict(
        title=dict(text="True activity", font=dict(size=18)),
        autorange="reversed",
    ),
)
utils.style_chart(fig)
st.plotly_chart(fig, use_container_width=True)

# ---------- Most confused pairs ----------
st.subheader("Most confused pairs")

# Off-diagonal cells only: the model said a different activity than the true one.
pairs = matrix.stack().rename("count").reset_index()
pairs.columns = ["true_activity", "predicted_activity", "count"]
pairs = pairs[pairs["true_activity"] != pairs["predicted_activity"]]
pairs = pairs[pairs["count"] > 0].nlargest(3, "count")

if pairs.empty:
    st.success("No mistakes found for this choice.")
else:
    for _, row in pairs.iterrows():
        st.write(
            f"**{row['true_activity']}** was predicted as "
            f"**{row['predicted_activity']}** {int(row['count'])} times"
        )

# ---------- Precision, recall and F1 ----------
st.subheader("Precision, recall and F1 for each activity")

# Keep only the rows for the same model, group and split.
per_class = metrics[
    (metrics["model"] == model)
    & (metrics["group"] == group)
    & (metrics["split"] == split)
]
if per_class.empty:
    st.info("There are no per-activity numbers for this choice.")
else:
    # Show the activities in our fixed order and the values as percent.
    per_class = per_class.set_index("activity").reindex(utils.ACTIVITIES).dropna(how="all")
    table = per_class[["precision", "recall", "f1"]].map(utils.pct)
    table.index.name = "Activity"
    table.columns = ["Precision", "Recall", "F1"]
    st.dataframe(table, use_container_width=True)
