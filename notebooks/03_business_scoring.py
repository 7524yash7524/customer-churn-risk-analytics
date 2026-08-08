"""
Step 4: Translate probabilities into a business decision.
Most churn projects stop at "here's who's at risk". A consulting-style
answer ranks customers by risk x value, so retention budget goes where
it actually pays off.
"""
import pandas as pd

df = pd.read_csv("../data/test_predictions.csv")

# risk score alone isn't enough - a $20/month customer at 90% risk
# isn't worth the same intervention as a $100/month customer at 60% risk
df["priority_score"] = df["churn_prob"] * df["clv_proxy"]

df["risk_tier"] = pd.cut(
    df["churn_prob"], bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
)

# simple retention ROI assumption: an intervention (discount, call, etc.)
# costs $50 and cuts churn probability by 30% relative, for illustration.
# this assumption should be stated explicitly as illustrative, not measured.
INTERVENTION_COST = 50
RETENTION_LIFT = 0.30

df["expected_value_saved"] = df["churn_prob"] * RETENTION_LIFT * df["clv_proxy"]
df["expected_roi"] = df["expected_value_saved"] - INTERVENTION_COST

# rank and only intervene where ROI is positive
priority_list = df.sort_values("priority_score", ascending=False)
worth_targeting = priority_list[priority_list["expected_roi"] > 0]

print("customers in test set:", len(df))
print("customers worth targeting (positive ROI):", len(worth_targeting))
print("total expected value saved if all positive-ROI customers targeted: $%.0f"
      % worth_targeting["expected_value_saved"].sum())
print("total intervention spend: $%.0f" % (len(worth_targeting) * INTERVENTION_COST))

df.to_csv("../data/scored_customers.csv", index=False)
print("saved scored customers")
