# Sensor Tiers page: find the cheapest sensor tier that is reliable for a new person.
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
summary_columns = ["model", "group", "kfold_acc_mean", "loso_acc_mean"]
missing = [c for c in summary_columns if c not in summary.columns]
if missing:
    st.error("The file results/results_summary.csv is missing these columns: " + ", ".join(missing))
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


# Return "yes" or "no" for a table cell.
def yes_no(value):
    return "yes" if value else "no"


# Put the tiers in order: Tier 1 first, Tier 3 last (full set).
tiers = tiers.sort_values("tier").reset_index(drop=True)

# ---------- The rule and the controls ----------
st.subheader("The rule")
st.write(
    "A setup is reliable if both are true for the same model, on the new person (LOSO) test: "
    "its accuracy is at least the minimum accuracy, and it is at most the allowed drop below "
    f"the full set. If a reliable setup is less than {utils.RULE_BORDERLINE:.0f} points above "
    "the minimum accuracy, it is borderline."
)

# The model comes first in the list, so logistic regression is the default.
model = st.selectbox(
    "Model", list(utils.MODELS.keys()), format_func=lambda code: utils.MODELS[code]
)
col_a, col_b = st.columns(2)
min_acc = col_a.slider("Minimum accuracy (%)", 50, 100, utils.RULE_MIN_ACC, 1)
allowed_drop = col_b.slider("Allowed drop from the full set (points)", 0.0, 15.0, utils.RULE_MAX_DROP, 0.5)

# Accuracy of the chosen model for each group, taken from results_summary.csv.
model_rows = summary[summary["model"] == model].drop_duplicates(subset=["group"]).set_index("group")
kfold_by_group = model_rows["kfold_acc_mean"]
loso_by_group = model_rows["loso_acc_mean"]

# The full set LOSO accuracy is the reference. Without it we cannot apply the rule.
if "both" not in loso_by_group.index or pd.isna(loso_by_group["both"]):
    st.info("There is no LOSO accuracy for the full set (both) with this model.")
    st.stop()
full_loso = loso_by_group["both"] * 100

# Add the LOSO accuracy (in percent), the drop and the two rule results to each tier.
tiers["loso_pct"] = tiers["group"].map(lambda g: loso_by_group.get(g, float("nan")) * 100)
tiers["below_full"] = full_loso - tiers["loso_pct"]
checks = [utils.check_reliable(a, full_loso, min_acc, allowed_drop) for a in tiers["loso_pct"]]
tiers["reliable"] = [c[0] for c in checks]
tiers["borderline"] = [c[1] for c in checks]

# ---------- Tier table ----------
st.subheader("Tier table")
st.write(f"Accuracy shown for: {utils.MODELS[model]}")

# Build a table with readable column names.
table = pd.DataFrame({
    "Tier": tiers["tier"],
    "Name": tiers["name"],
    "Sensor group": tiers["group"].map(utils.GROUPS).fillna(tiers["group"]),
    "5 fold accuracy": tiers["group"].map(kfold_by_group).map(pct_or_na),
    "LOSO accuracy": (tiers["loso_pct"] / 100).map(pct_or_na),
    "Points below the full set": tiers["below_full"].map(
        lambda v: "not available" if pd.isna(v) else f"{v:.1f}"
    ),
    "Reliable": tiers["reliable"].map(yes_no),
    "Borderline": tiers["borderline"].map(yes_no),
    "Power": tiers.apply(power_text, axis=1),
    "Power mode": tiers["power_mode"].map(lambda v: text_or(v, "not filled yet")),
    "Datasheet": tiers["datasheet_ref"].map(lambda v: text_or(v, "not filled yet")),
    "Note": tiers["note"].map(lambda v: text_or(v, "")),
})
st.dataframe(table, hide_index=True, use_container_width=True)

# Note about where the power numbers come from.
st.caption(
    "Sensor chip only (LSM6DSL, low-power mode, 52 Hz), from the datasheet. "
    "The gyroscope value (265 uA) is calculated as 290 minus 25. "
    "One chip is an example, not a rule. It is not the power of a whole device."
)

# ---------- Find the cheapest tier ----------
st.subheader("Find the cheapest tier")

# The cheapest tier is the reliable tier with the lowest power value.
tiers["power_num"] = pd.to_numeric(tiers["power_value"], errors="coerce")
candidates = tiers[tiers["reliable"] & tiers["power_num"].notna()]

# Show the result in a highlighted box.
if candidates.empty:
    st.info(
        "No tier is reliable with these settings. "
        f"Try a lower minimum accuracy or a larger allowed drop. Full set accuracy: {full_loso:.1f}%."
    )
else:
    best = candidates.loc[candidates["power_num"].idxmin()]
    group_name = utils.GROUPS.get(best["group"], best["group"])
    details = (
        f"- Group: {group_name}\n"
        f"- LOSO accuracy: {best['loso_pct']:.1f}%\n"
        f"- Points below the full set ({full_loso:.1f}%): {best['below_full']:.1f}\n"
        f"- Power: {power_text(best)}"
    )
    if best["borderline"]:
        st.warning(
            f"**Cheapest tier: {best['tier']} ({text_or(best['name'], '')}), reliable, but borderline**\n\n"
            + details
        )
    else:
        st.success(
            f"**Cheapest reliable tier: {best['tier']} ({text_or(best['name'], '')})**\n\n" + details
        )
