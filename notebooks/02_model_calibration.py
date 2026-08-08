"""
Step 3: Build the calibrated risk model.
Same pattern as the chess win-predictor - Gradient Boosting, then check
if the predicted probabilities are actually trustworthy, not just accurate.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

df = pd.read_csv("../data/telco_churn_clean.csv")

target = "Churn"
drop_cols = ["customerID", "Churn", "clv_proxy"]  # keep clv_proxy out of the model, it's for scoring later
X = df.drop(columns=drop_cols)
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)

base_model = Pipeline([
    ("pre", pre),
    ("gb", GradientBoostingClassifier(random_state=42)),
])

base_model.fit(X_train, y_train)
raw_probs = base_model.predict_proba(X_test)[:, 1]

print("uncalibrated ROC-AUC:", round(roc_auc_score(y_test, raw_probs), 3))
print("uncalibrated Brier score:", round(brier_score_loss(y_test, raw_probs), 4))

# calibrate with isotonic regression, same as the chess project
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)
cal_probs = calibrated_model.predict_proba(X_test)[:, 1]

print("calibrated ROC-AUC:", round(roc_auc_score(y_test, cal_probs), 3))
print("calibrated Brier score:", round(brier_score_loss(y_test, cal_probs), 4))

# calibration curve - are predicted probabilities close to actual frequencies?
frac_pos, mean_pred = calibration_curve(y_test, cal_probs, n_bins=10)
calib_df = pd.DataFrame({"predicted": mean_pred, "actual": frac_pos})
calib_df.to_csv("../data/calibration_curve.csv", index=False)
print(calib_df)

# save everything the dashboard needs
joblib.dump(calibrated_model, "../data/churn_model.joblib")

results = df.loc[idx_test].copy()
results["churn_prob"] = cal_probs
results.to_csv("../data/test_predictions.csv", index=False)
print("saved model and predictions")
