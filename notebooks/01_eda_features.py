"""
Step 1-2: Define objective + acquire/clean data
Client framing: A telecom provider is losing ~27% of customers annually.
Goal: predict who is likely to churn, and prioritize retention spend by
customer value, not just by risk.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/telco_churn.csv")

# TotalCharges has some blank strings for new customers (tenure=0), fix that
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn"] = (df["Churn"] == "Yes").astype(int)

print("rows:", len(df))
print("churn rate:", df["Churn"].mean().round(3))

# quick look at churn rate by contract type - this is usually the biggest driver
print(df.groupby("Contract")["Churn"].mean())

# customer lifetime value proxy - since we don't have real CLV, use
# tenure * monthly charges as a simple approximation. flagged clearly
# as a proxy in the writeup, not real revenue data.
df["clv_proxy"] = df["tenure"] * df["MonthlyCharges"]

# a few features that tend to matter for churn based on domain logic,
# not just throwing every column at the model
df["has_internet"] = (df["InternetService"] != "No").astype(int)
df["num_addon_services"] = df[
    ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
     "TechSupport", "StreamingTV", "StreamingMovies"]
].apply(lambda col: (col == "Yes").astype(int)).sum(axis=1)

df.to_csv("../data/telco_churn_clean.csv", index=False)
print("saved cleaned data:", df.shape)