"""
Step 5: Executive dashboard.
Shows risk segments, key churn drivers, and a retention-spend ROI
calculator, mirroring the kind of unified dashboard FischerJordan
builds for account managers / execs.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Retention Risk Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/scored_customers.csv")
    calib = pd.read_csv("data/calibration_curve.csv")
    return df, calib

df, calib = load_data()

st.title("Customer Retention Risk Dashboard")
st.caption(
    "Predicts churn probability per customer, calibrates it against actual "
    "outcomes, and ranks customers by expected value of retention spend."
)

# ---- top metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers scored", len(df))
col2.metric("Base churn rate", f"{df['Churn'].mean():.1%}")
col3.metric("High risk customers", int((df["risk_tier"] == "High").sum()))
col4.metric(
    "Expected value saved (positive-ROI targeting)",
    f"${df[df['expected_roi'] > 0]['expected_value_saved'].sum():,.0f}"
)

st.divider()

# ---- risk tier breakdown ----
left, right = st.columns([1, 1])

with left:
    st.subheader("Risk tier distribution")
    tier_counts = df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
    fig = px.bar(
        x=tier_counts.index, y=tier_counts.values,
        labels={"x": "Risk tier", "y": "Customers"},
        color=tier_counts.index,
        color_discrete_map={"Low": "#4C9A6A", "Medium": "#E8A33D", "High": "#C6473E"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn rate by contract type")
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values()
    fig2 = px.bar(
        x=contract_churn.values, y=contract_churn.index, orientation="h",
        labels={"x": "Churn rate", "y": "Contract type"},
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---- calibration check ----
st.subheader("Is the model trustworthy? (calibration check)")
st.caption(
    "If the model says '70% chance of churn' for a group of customers, "
    "roughly 70% of them should actually churn. This checks that."
)
fig3 = px.line(
    calib, x="predicted", y="actual", markers=True,
    labels={"predicted": "Predicted probability", "actual": "Actual churn rate"},
)
fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---- retention ROI calculator ----
st.subheader("Retention targeting: who's worth the spend")
st.caption(
    "Ranks customers by (churn probability x customer value), then only "
    "recommends targeting where expected value saved exceeds intervention cost."
)

intervention_cost = st.slider("Assumed intervention cost per customer ($)", 10, 200, 50, step=10)
retention_lift = st.slider("Assumed relative reduction in churn probability from intervention", 0.1, 0.6, 0.3, step=0.05)

df["expected_value_saved_live"] = df["churn_prob"] * retention_lift * df["clv_proxy"]
df["expected_roi_live"] = df["expected_value_saved_live"] - intervention_cost
targetable = df[df["expected_roi_live"] > 0]

c1, c2, c3 = st.columns(3)
c1.metric("Customers worth targeting", len(targetable))
c2.metric("Total intervention spend", f"${len(targetable) * intervention_cost:,.0f}")
c3.metric("Total expected value saved", f"${targetable['expected_value_saved_live'].sum():,.0f}")

st.dataframe(
    targetable.sort_values("expected_roi_live", ascending=False)[
        ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_prob", "risk_tier", "expected_roi_live"]
    ].head(50).rename(columns={"expected_roi_live": "expected_roi"}),
    use_container_width=True,
)

st.caption(
    "Note: customer value uses tenure x monthly charge as a CLV proxy, and "
    "intervention cost/lift are illustrative assumptions, not measured figures. "
    "In a real engagement these would come from client finance and A/B test data."
)
