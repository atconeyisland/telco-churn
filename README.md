# Telco Customer Churn - Supervised Learning (CA-4)

Binary classification: predict which telecom customers will churn.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[app,dev]"
```

Download the dataset from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
and save it as `data/raw/Telco-Customer-Churn.csv`.

## Run
```bash
pytest                                                                    # sanity tests
jupytext --to ipynb notebooks/churn_analysis.py --output notebooks/churn_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/churn_analysis.ipynb
streamlit run app/streamlit_app.py
```

The notebook writes figures to `reports/figures/`, the final pipeline and model card
to `models/`, and caches grid searches in `artifacts/` (delete it to re-tune).

## Method summary
- 70/30 stratified split, fixed seed. All preprocessing lives inside the pipeline (no leakage).
- Baselines: Dummy, Logistic Regression, Decision Tree, KNN.
- Advanced: Random Forest, Gradient Boosting, XGBoost x {no handling, class weights, SMOTE}.
- GridSearchCV with 5-fold stratified CV (ROC-AUC refit; PR-AUC, F1, recall also tracked).
- Final model chosen on CV ROC-AUC only. Threshold tuned on out-of-fold train predictions.
  The test set is used once for the final report.
- Interpretation: built-in importance, permutation importance, SHAP, decile lift, calibration.
