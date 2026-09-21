### 🎯 Customer Retention Risk & Value-Based Targeting
Calibrated churn prediction model (Gradient Boosting + isotonic calibration, ROC-AUC 0.845) 
paired with a risk × customer-value targeting layer, modeled on boutique consulting 
methodology (Customer Analytics practice). Ranks customers by 
expected ROI of retention spend rather than risk alone.

**Live dashboard:** [Customer Retention Risk Dashboard](https://customer-churn-risk-analytics-bogmrxieudfevykczcxoht.streamlit.app/)
**Repo:** [customer-churn-risk-analytics](https://github.com/7524yash7524/customer-churn-risk-analytics)

# Customer Retention Risk & Value-Based Targeting

A Risk & Customer Analytics case study modeled on how boutique analytics
consultancies (e.g. FischerJordan) approach customer churn: not just
"predict who leaves," but "tell the business who's worth saving, and prove
the model can be trusted."

## Business objective

A telecom provider is losing ~27% of customers annually. Rather than
targeting every at-risk customer equally, the goal is to prioritize
retention spend where it actually pays off — ranking customers by
**churn probability x customer value**, not risk alone.

## Methodology (4-step)

1. **Define objective** — quantify the churn problem and the decision it should drive (who to target, not just who's at risk).
2. **Acquire & synthesize data** — IBM Telco Customer Churn dataset (7,043 customers), cleaned and feature-engineered around known churn drivers (contract type, tenure, add-on services).
3. **Build the model** — Gradient Boosting classifier, calibrated with isotonic regression so predicted probabilities reflect real-world frequencies (same calibration approach used in my [chess ML project](../chess-dashboard-improved)).
4. **Translate to action** — convert probabilities into a prioritized, ROI-filtered retention list and an executive dashboard.

## Key results

- **ROC-AUC: 0.845** on held-out test data
- **Calibration:** predicted probabilities track actual churn rates closely across risk deciles (see calibration chart in dashboard) — the model isn't just accurate, it's trustworthy at the probability level
- **Biggest driver:** month-to-month contracts churn at ~43% vs. ~3% for two-year contracts
- **Business impact (illustrative):** targeting only customers with positive expected ROI (vs. targeting everyone above a risk threshold) saves an estimated $157K in customer value against $34K in intervention spend on the test set

## Why this isn't "just another churn model"

Most churn portfolio projects stop at ROC-AUC. This one adds two things
FischerJordan-style consulting work requires:

- **Calibration, not just discrimination** — a risk score is only useful to a business if "70% risk" really means 70%, not just that riskier customers rank higher.
- **Value-weighted targeting** — flagging risk without weighing it against customer value wastes retention budget on low-value customers and misses high-value ones sitting at medium risk.

## Structure

```
notebooks/01_eda_features.py       - data cleaning, feature engineering
notebooks/02_model_calibration.py  - model training + calibration check
notebooks/03_business_scoring.py   - risk x value scoring, ROI logic
app/app.py                          - Streamlit executive dashboard
data/                                - cleaned data, model, scored outputs
```

## Assumptions (stated explicitly)

- Customer value uses tenure x monthly charge as a CLV proxy (no real revenue/margin data available in the public dataset).
- Intervention cost ($50) and retention lift (30%) are illustrative, adjustable in the dashboard — in a real engagement these would come from client finance data and A/B tests, not assumption.

## Run locally

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```
