# Sensor Tiers page: find the cheapest sensor tier that keeps enough accuracy.
import pandas as pd
import streamlit as st

import utils

# Set the browser tab title and icon.
st.set_page_config(page_title="Sensor Tiers", page_icon="📡", layout="wide")

# Add the custom look (fonts, background, cards).
utils.apply_style()

# Show the page title.
st.title("Sensor Tiers")

# The tier file may not exist yet. This is not an error, so show a friendly note and stop.
if not (utils.BASE_DIR / "results" / "sensor_tiers.csv").exists():
    st.info(
        "Coming soon: the sensor tier table needs power numbers from datasheets, "
        "which the team is still collecting."
    )
    st.stop()

# Warn if any of the files is test data.
utils.show_banner(["sensor_tiers.csv", "results_summary.csv"])

# Read both tables. Each call stops the page if its file is missing.
tiers = utils.load_csv("sensor_tiers.csv")
summary = utils.load_csv("results_summary.csv")

# Check that the columns we need are there, so the page never crashes.
tier_columns = ["tier", "name", "group", "kfold_acc", "loso_acc",
                "power_value", "power_unit", "power_mode", "datasheet_ref", "note"]
missing = [c for c in tier_columns if c not in tiers.columns]
if missing:
    st.error("The file results/sensor_tiers.csv is missing these columns: " + ", ".join(missing))
    st.stop()
if tiers.empty:
    st.info("The file results/sensor_tiers.csv has no rows yet.")
    st.stop()


# Return text for a table cell, or the given words if the cell is empty.
def text_or(value, empty_text):
    if pd.isna(value) or str(value).strip() == "":
        return empty_text
    return str(value)


# Return a fraction as a percent, or "not available" if the cell is empty.
def pct_or_na(value):
    if pd.isna(value):
        return "not available"
    return utils.pct(value)


# Return the power text for one tier row, or "not filled yet" if there is no value.
def power_text(row):
    if pd.isna(row["power_value"]) or str(row["power_value"]).strip() == "":
        return "not filled yet"
    parts = [str(row["power_value"]), text_or(row["power_unit"], "")]
    return " ".join(p for p in parts if p)


# Put the tiers in order: Tier 1 first (cheapest), Tier 3 last (full set).
tiers = tiers.sort_values("tier").reset_index(drop=True)

# ---------- Tier table ----------
st.subheader("Tier table")

# Build a table with readable column names.
table = pd.DataFrame({
    "Tier": tiers["tier"],
    "Name": tiers["name"],
    "Sensor group": tiers["group"].map(utils.GROUPS).fillna(tiers["group"]),
    "K fold accuracy": tiers["kfold_acc"].map(pct_or_na),
    "LOSO accuracy": tiers["loso_acc"].map(pct_or_na),
    "Power": tiers.apply(power_text, axis=1),
    "Power mode": tiers["power_mode"].map(lambda v: text_or(v, "not filled yet")),
    "Datasheet": tiers["datasheet_ref"].map(lambda v: text_or(v, "not filled yet")),
    "Note": tiers["note"].map(lambda v: text_or(v, "")),
})
st.dataframe(table, hide_index=True, use_container_width=True)

# Note about where the power numbers come from.
st.caption(
    "Power numbers come from one chip's datasheet [5]. They show the size of the savings, "
    "not an exact value for any product.\n\n"
    "Time domain only and frequency domain only use the same sensors, so they change "
    "the amount of calculation, not the sensor power."
)

# ---------- Find the cheapest tier ----------
st.subheader("Find the cheapest tier")

# Controls: the model and how much accuracy we can lose.
model = st.selectbox(
    "Model", list(utils.MODELS.keys()), format_func=lambda code: utils.MODELS[code]
)
allowed_drop = st.slider(
    "How many points of accuracy can you lose compared with the full set?",
    0.0, 15.0, 5.0, 0.5,
)

# Keep only the rows of the chosen model that have an accuracy.
model_rows = summary[summary["model"] == model].dropna(subset=["kfold_acc_mean"])
acc_by_group = model_rows.set_index("group")["kfold_acc_mean"]

# The full set accuracy is the reference. Without it we cannot compare.
if "both" not in acc_by_group.index:
    st.info("There is no K fold accuracy for the full set (both) with this model.")
    st.stop()
full_acc = acc_by_group["both"] * 100

# Go from the cheapest tier up. Take the first reduced tier inside the allowed drop.
chosen = None
for _, row in tiers.iterrows():
    if row["group"] == "both" or row["group"] not in acc_by_group.index:
        continue
    acc = acc_by_group[row["group"]] * 100
    drop = full_acc - acc
    if drop <= allowed_drop + 1e-9:
        chosen = (row, acc, drop)
        break

# Show the result in a highlighted box.
if chosen is None:
    st.info(
        f"The full set is the only option at this setting. "
        f"No reduced tier stays within {allowed_drop:.1f} points of the full set "
        f"({full_acc:.1f}%)."
    )
else:
    row, acc, drop = chosen
    group_name = utils.GROUPS.get(row["group"], row["group"])
    st.success(
        f"**Cheapest tier that fits: {row['tier']} ({text_or(row['name'], '')})**\n\n"
        f"- Group: {group_name}\n"
        f"- K fold accuracy: {acc:.1f}%\n"
        f"- Drop from the full set ({full_acc:.1f}%): {drop:.1f} points\n"
        f"- Power: {power_text(row)}"
    )
