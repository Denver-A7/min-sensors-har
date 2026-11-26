# Try a Prediction page: pick one window from a new volunteer and let a model guess the activity.
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import utils

# Set the browser tab title and icon.
st.set_page_config(page_title="Try a Prediction", page_icon="🎯")

# Add the custom look (fonts, background, cards).
utils.apply_style()

# Show the page title.
st.title("Try a Prediction")

# Warn if the models are not the final ones.
if not utils.is_real("models"):
    st.warning("PRELIMINARY MODELS. These models are not our final results.")

# The models give a number from 1 to 6. This maps the number to the activity name.
CLASS_TO_ACTIVITY = {i + 1: name for i, name in enumerate(utils.ACTIVITIES)}

# Read the demo features (X) and the info about each row (meta). Stops if a file is missing.
X, meta = utils.load_demo_data()


# Load a model, or return None with a friendly message if the file is missing.
# We do not call utils.load_model on a missing file because it stops the whole page.
def try_load_model(model_code, group_code):
    path = utils.BASE_DIR / "models" / f"demo_{model_code}_{group_code}.joblib"
    if not path.exists():
        st.info(f"The model file models/{path.name} is missing, so this result is skipped.")
        return None
    return utils.load_model(model_code, group_code)


# Predict one window with one model and group.
# Returns (predicted activity, confidence as a fraction, or None when the model has no probabilities).
def predict_window(model, group_code, row_position):
    cols = utils.get_group_columns(group_code)
    # Select by column position, because the X columns are only numbered.
    x_row = X.iloc[[row_position], cols].to_numpy()
    predicted = CLASS_TO_ACTIVITY.get(int(model.predict(x_row)[0]), "unknown")
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(x_row)[0].max())
    return predicted, confidence


# Show the confidence as a percent, or "n/a" for models without probabilities.
def confidence_text(confidence):
    return "n/a" if confidence is None else utils.pct(confidence)


# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

# Pick the model and the sensor setup. We show names but keep the codes.
model_code = st.sidebar.selectbox(
    "Model", list(utils.MODELS.keys()), format_func=lambda code: utils.MODELS[code]
)
group_code = st.sidebar.selectbox(
    "Sensor setup", list(utils.GROUPS.keys()), format_func=lambda code: utils.GROUPS[code]
)

# Pick a volunteer from the demo file.
volunteers = sorted(meta["volunteer"].unique().tolist())
volunteer = st.sidebar.selectbox("Volunteer", volunteers)

# All windows (row positions in X) of this volunteer.
windows = meta.index[meta["volunteer"] == volunteer].tolist()
if not windows:
    st.info("This volunteer has no windows in the demo data.")
    st.stop()

# If the volunteer changed, forget the old window so the list does not break.
if st.session_state.get("window_volunteer") != volunteer:
    st.session_state["window_volunteer"] = volunteer
    st.session_state["window"] = windows[0]

# The button must come before the selectbox, so it can change the stored window.
if st.sidebar.button("Random window"):
    st.session_state["window"] = random.choice(windows)

# Pick one window. We show the original row_index from the meta file.
row_position = st.sidebar.selectbox(
    "Window",
    windows,
    key="window",
    format_func=lambda pos: f"Row {meta.loc[pos, 'row_index']}",
)

# ---------- Main prediction ----------
st.write(
    "These models were trained on the official training volunteers. "
    "This window comes from a volunteer the model never saw."
)

true_activity = meta.loc[row_position, "true_activity"]

# Load the chosen model and predict this one window.
model = try_load_model(model_code, group_code)
if model is None:
    st.stop()
predicted, confidence = predict_window(model, group_code, row_position)

# Show the true and predicted activity side by side.
col1, col2 = st.columns(2)
col1.metric("True activity", true_activity)
col2.metric("Predicted activity", predicted)

# Green for a correct guess, yellow for a wrong one.
if predicted == true_activity:
    st.success("Correct")
else:
    st.warning("Wrong")

# Probability chart for models that have one. The SVM only gives a class.
if hasattr(model, "predict_proba"):
    cols = utils.get_group_columns(group_code)
    probs = model.predict_proba(X.iloc[[row_position], cols].to_numpy())[0]
    names = [CLASS_TO_ACTIVITY.get(int(c), str(c)) for c in model.classes_]
    # The true activity bar is yellow, all other bars are blue.
    colors = [utils.YELLOW if name == true_activity else utils.BLUE for name in names]

    fig = go.Figure(
        go.Bar(
            x=probs * 100,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Probability of each activity", font=dict(size=17)),
        xaxis_title="Probability in %",
        yaxis_title="Activity",
        xaxis=dict(range=[0, 110]),
        font=dict(size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    utils.style_chart(fig)
    st.plotly_chart(fig, width="stretch", config=utils.MOBILE_CHART_CONFIG)
    st.caption("Yellow bar = the true activity.")
else:
    st.info("The SVM gives only a predicted activity, no probability.")

# ---------- Compare all five setups ----------
if st.button("Compare all five setups"):
    rows = []
    for code, name in utils.GROUPS.items():
        other = try_load_model(model_code, code)
        if other is None:
            continue
        pred, conf = predict_window(other, code, row_position)
        rows.append(
            {
                "Setup": name,
                "Predicted activity": pred,
                "Confidence (%)": confidence_text(conf),
                "Correct": "yes" if pred == true_activity else "no",
            }
        )
    st.subheader(f"{utils.MODELS[model_code]} with each sensor setup")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------- Compare all three models ----------
if st.button("Compare all three models"):
    rows = []
    for code, name in utils.MODELS.items():
        other = try_load_model(code, group_code)
        if other is None:
            continue
        pred, conf = predict_window(other, group_code, row_position)
        rows.append(
            {
                "Model": name,
                "Predicted activity": pred,
                "Confidence (%)": confidence_text(conf),
                "Correct": "yes" if pred == true_activity else "no",
            }
        )
    st.subheader(f"Each model with {utils.GROUPS[group_code]}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
