# Customer Retention Risk & Value-Based Targeting

A calibrated churn model for a telecom provider that ranks customers by the **expected return on retention spend**, not by churn risk alone.

**Live dashboard:** [Customer Retention Risk Dashboard](https://customer-churn-risk-analytics-bogmrxieudfevykczcxoht.streamlit.app/)

## The business question

The telecom in this dataset loses about 27% of its customers (26.5% in the IBM Telco data, 7,043 customers). Contacting every at-risk customer wastes money. The goal is to decide **who is worth saving**, and to check that the model's probabilities can be trusted.

## What I built

1. **Data and features:** IBM Telco Customer Churn data, cleaned and feature-engineered (contract type, tenure, add-on services).
2. **Model:** Gradient Boosting classifier with isotonic calibration (5-fold), trained on 80% of customers and evaluated on a stratified 20% hold-out (1,409 customers).
3. **Targeting:** each customer gets `expected value saved = churn probability x 30% retention lift x customer value`, and the intervention costs $50. Only customers with positive expected ROI are targeted.
4. **Dashboard:** Streamlit app with the calibration chart, risk tiers and the targeting list. Cost and lift are adjustable.

## Results (1,409-customer test set)

| Metric | Value |
|---|---|
| ROC-AUC | 0.845 |
| Brier score | 0.135 (0.195 for always predicting the base rate) |
| Calibration | Average gap between predicted and actual churn is 4.4 percentage points across 10 risk bins. The worst bin is off by 10.7 points (predicted 85%, actual 75%) |
| Churn by contract | Month-to-month 42.7%, one year 11.3%, two year 2.8% |

**Targeting rules compared** (expected values from the model, using the assumptions below):

| Rule | Customers contacted | Expected value saved | Spend | Net |
|---|---:|---:|---:|---:|
| Everyone with churn probability of 30% or more | 529 | $98,270 | $26,450 | $71,820 |
| Everyone with churn probability of 50% or more | 299 | $42,114 | $14,950 | $27,164 |
| **Positive expected ROI (this project)** | **686** | **$157,295** | **$34,300** | **$122,995** |

Targeting by expected ROI gives the largest net gain, because it includes valuable customers at medium risk and skips low-value customers at high risk.

## Assumptions

- Customer value is `tenure x monthly charge`, a proxy, because the public dataset has no revenue or margin data.
- The $50 intervention cost and 30% retention lift are **illustrative**, not measured. In a real project they would come from finance data and A/B tests.
- The dollar figures above are model expectations, not observed outcomes.

## Structure

```
notebooks/01_eda_features.py       data cleaning and feature engineering
notebooks/02_model_calibration.py  training and calibration check
notebooks/03_business_scoring.py   risk x value scoring and ROI logic
app/app.py                         Streamlit dashboard
data/                              cleaned data, model, scored outputs
```

## Run locally

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

Data: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Author

Yash Chitransh, M.Sc. Mathematics & Scientific Computing, NIT Warangal. [LinkedIn](https://www.linkedin.com/in/yash-chitransh-4b108b2aa)

## License

MIT, see `LICENSE`.
