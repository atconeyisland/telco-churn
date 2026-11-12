# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Telco Customer Churn - Supervised Learning Case Study (CA-4)
#
# **Task:** binary classification - predict whether a telecom customer will churn.
# **Dataset:** Telco Customer Churn (IBM sample data, Kaggle `blastchar/telco-customer-churn`).
#
# **Rubric map**
# 1. Problem understanding and EDA (missing values, outliers, distributions, correlations, encoding, 70/30 split)
# 2. Baseline models (Logistic Regression, Decision Tree, KNN) with Accuracy / Precision / Recall / F1 / Confusion Matrix
# 3. Advanced models (Random Forest, Gradient Boosting, XGBoost), GridSearchCV, cross-validation, ROC/AUC, SMOTE and class weights
# 4. Interpretation: final model justification, feature importance, business insights, limitations, future work, deployment demo

# %%
import warnings

import joblib
import numpy as np
import pandas as pd
from IPython.display import display

from churn import artifacts, config, data, eda, evaluate, interpret, models, plots, thresholds, tuning

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:,.4f}".format)
plots.set_style()

# %% [markdown]
# ## 1. Problem understanding
#
# Acquiring a new customer costs far more than retaining an existing one, so a churn model is only
# useful if it **ranks at-risk customers well** and lets the business choose **how many to contact**.
# Because only ~1 in 4 customers churns, plain accuracy is misleading (always predicting "No" scores ~73%).
# We therefore judge models on ROC-AUC, PR-AUC, recall and F1, and tune the decision threshold explicitly.

# %% [markdown]
# ## 2. Data exploration
# ### 2.1 Dataset size, feature types, target

# %%
raw = data.load_raw()
print(f"Rows: {raw.shape[0]:,} | Columns: {raw.shape[1]}")
display(raw.head())
display(eda.overview(raw))

# %% [markdown]
# `TotalCharges` is stored as text. Pandas does not flag it as missing, so check for blank strings explicitly.

# %%
blank_total = raw["TotalCharges"].astype(str).str.strip().eq("")
print(f"TotalCharges dtype: {raw['TotalCharges'].dtype} | blank entries: {blank_total.sum()}")
display(raw.loc[blank_total, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# %% [markdown]
# All blank rows have `tenure == 0`: brand-new customers that have not been billed yet.
# The correct value is therefore `0.0` (a deterministic rule applied in `data.clean_data`), not a statistical imputation.

# %%
df = data.clean_data(raw)

print(f"Unique customerID: {df['customerID'].is_unique} | duplicate rows: {df.duplicated().sum()}")
print(f"Remaining missing values: {int(df.isna().sum().sum())}")

feature_types = pd.Series(
    {
        "numeric (raw)": config.RAW_NUMERIC,
        "categorical": config.CATEGORICAL,
        "identifier (dropped)": [config.ID_COL],
        "target": [config.TARGET],
    }
)
display(feature_types.to_frame("columns"))
display(eda.overview(df))

# %% [markdown]
# ### 2.2 Target variable

# %%
display(
    df[config.TARGET]
    .value_counts()
    .to_frame("customers")
    .assign(share=lambda t: t["customers"] / t["customers"].sum())
)
plots.plot_target_distribution(df[config.TARGET], name="01_target_distribution")

# %% [markdown]
# ### 2.3 Distribution analysis

# %%
numeric_summary = df[config.RAW_NUMERIC].describe().T.join(df[config.RAW_NUMERIC].skew().rename("skew"))
display(numeric_summary)
plots.plot_numeric_distributions(df, config.RAW_NUMERIC, name="02_numeric_distributions")

# %% [markdown]
# ### 2.4 Outlier detection (Tukey IQR fences)

# %%
display(eda.iqr_outliers(df, config.RAW_NUMERIC))
plots.plot_boxplots(df, config.RAW_NUMERIC, name="03_boxplots")

# %% [markdown]
# Check the outlier table above. If few or no points fall outside the fences, no rows need removing: a long right
# tail in `TotalCharges` is genuine (long-tenure, high-spend customers). We keep all rows and use `StandardScaler`,
# which tree models ignore anyway.

# %% [markdown]
# ### 2.5 Churn drivers by category

# %%
df_eda = df.assign(TenureBucket=eda.tenure_bucket(df["tenure"]))
plots.plot_churn_rate_grid(df_eda, ["TenureBucket", *config.CATEGORICAL], name="04_churn_rate_by_category")

# %%
display(eda.churn_rate_by(df_eda, "Contract"))
display(eda.churn_rate_by(df_eda, "PaymentMethod"))
display(eda.churn_rate_by(df_eda, "TenureBucket"))

# %% [markdown]
# ### 2.6 Association with the target (chi-square and Cramér's V)

# %%
association = eda.chi2_table(df, config.CATEGORICAL, config.TARGET)
display(association)
plots.plot_ranked_bar(
    association["cramers_v"], "Association with churn (Cramér's V)", "Cramér's V", name="05_cramers_v"
)

# %% [markdown]
# ### 2.7 Correlation analysis

# %%
corr_frame = df[config.RAW_NUMERIC].assign(Churn=data.encode_target(df[config.TARGET]))
plots.plot_correlation_heatmap(corr_frame.corr(), name="06_correlation_numeric")

# %% [markdown]
# `tenure` and `TotalCharges` are strongly correlated (TotalCharges is roughly tenure x MonthlyCharges).
# This is harmless for tree ensembles and tolerable for regularised Logistic Regression, but it is worth noting in the report.

# %% [markdown]
# ## 3. Feature engineering, encoding and 70/30 split
#
# Two stateless features are added: `n_addons` (number of internet add-on services) and `is_autopay`.
# Encoding (one-hot, binary columns collapse to one indicator) and scaling are part of the model pipeline,
# so they are fitted on training data only. The split is stratified to preserve the churn ratio.

# %%
df_model = data.add_features(df)
X_train, X_test, y_train, y_test = data.make_splits(df_model)

split_summary = pd.DataFrame(
    {
        "rows": [len(X_train), len(X_test)],
        "share": [len(X_train) / len(df_model), len(X_test) / len(df_model)],
        "churn_rate": [y_train.mean(), y_test.mean()],
    },
    index=["train", "test"],
)
display(split_summary)

# %%
preprocessor = models.build_preprocessor().fit(X_train)
X_train_encoded = pd.DataFrame(
    preprocessor.transform(X_train), columns=preprocessor.get_feature_names_out(), index=X_train.index
)
print(f"Encoded training matrix: {X_train_encoded.shape}")
display(X_train_encoded.head())

target_corr = X_train_encoded.corrwith(y_train).dropna()
plots.plot_ranked_bar(
    target_corr, "Correlation of encoded features with churn (train)", "Pearson r", name="07_target_correlation"
)

# %% [markdown]
# ## 4. Baseline models
#
# Defaults only (no tuning), evaluated on the held-out 30% test set at the standard 0.5 threshold.
# A `DummyClassifier` (always predicts the majority class) provides the floor every model must beat.

# %%
baselines = models.baseline_models()
for estimator in baselines.values():
    estimator.fit(X_train, y_train)

baseline_results = evaluate.evaluate_models(baselines, X_test, y_test)
display(baseline_results)

# %%
plots.plot_confusion_matrices(baselines, X_test, y_test, name="08_baseline_confusion_matrices")

# %% [markdown]
# **Interpretation (write after running; cover these points with your actual numbers)**
# - Dummy accuracy equals the majority-class share, with recall 0: accuracy alone cannot be trusted here.
# - Logistic Regression: best ranking (ROC-AUC), but recall on churners at 0.5 is well below precision.
# - Decision Tree (unconstrained): near-perfect on train, weaker on test, so it overfits (compare train vs test).
# - KNN: sensitive to the many one-hot dimensions, usually the weakest of the three on AUC.
# - Conclusion: the gap between accuracy and recall motivates imbalance handling and threshold tuning.

# %% [markdown]
# ### 4.1 Cross-validated baselines (5-fold stratified, training set)

# %%
display(tuning.cv_summary(baselines, X_train, y_train))

# %% [markdown]
# ## 5. Advanced modelling
#
# Random Forest, Gradient Boosting and XGBoost, each under three imbalance strategies:
# `none`, `weights` (class_weight / scale_pos_weight) and `smote` (inside the pipeline, so synthetic samples are
# created only from the training portion of each CV fold). Every combination is tuned with `GridSearchCV`
# (5-fold stratified CV, refit on ROC-AUC; PR-AUC, F1 and recall are tracked too).
#
# Grid searches are cached in `artifacts/`. Delete the file to re-run them.

# %%
config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_file = config.CACHE_DIR / "grid_searches.joblib"

if cache_file.exists():
    searches = joblib.load(cache_file)
    print(f"Loaded cached grid searches from {cache_file.name}")
else:
    searches = tuning.tune_all(X_train, y_train)
    joblib.dump(searches, cache_file, compress=3)

summary = tuning.summarize(searches)
display(summary)

# %% [markdown]
# ### 5.1 Imbalance strategy comparison (CV, training data only)

# %%
for metric, title in [
    ("cv_roc_auc", "CV ROC-AUC"),
    ("cv_pr_auc", "CV PR-AUC"),
    ("cv_f1", "CV F1 at 0.5 threshold"),
    ("cv_recall", "CV recall at 0.5 threshold"),
]:
    pivot = summary.pivot(index="model", columns="strategy", values=metric)
    display(pivot.style.set_caption(title).format("{:.4f}"))

plots.plot_heatmap(
    summary.pivot(index="model", columns="strategy", values="cv_f1"),
    "CV F1 at 0.5 by imbalance strategy",
    name="09_strategy_f1",
)
plots.plot_heatmap(
    summary.pivot(index="model", columns="strategy", values="cv_roc_auc"),
    "CV ROC-AUC by imbalance strategy",
    name="10_strategy_auc",
)

# %% [markdown]
# ROC-AUC is threshold-free, so imbalance handling barely moves it. Its effect shows up in **recall and F1 at the
# default 0.5 threshold**, where class weights and SMOTE trade precision for recall. The threshold analysis in section 6
# shows that a well-chosen threshold can recover most of this effect for any model.

# %% [markdown]
# ### 5.2 Tuned models on the test set

# %%
tuned = tuning.collect_best(searches, summary)
display(
    tuning.best_per_model(summary)[
        ["model", "strategy", "cv_roc_auc", "cv_pr_auc", "cv_f1", "cv_recall", "best_params"]
    ]
)

tuned_results = evaluate.evaluate_models(tuned, X_test, y_test)
comparison = pd.concat([baseline_results.drop(index="Dummy"), tuned_results])
display(
    comparison[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].sort_values(
        "roc_auc", ascending=False
    )
)

# %%
curve_models = {k: v for k, v in baselines.items() if k == "Logistic Regression"} | tuned
plots.plot_roc_curves(curve_models, X_test, y_test, name="11_roc_curves")
plots.plot_pr_curves(curve_models, X_test, y_test, name="12_pr_curves")

# %%
plots.plot_confusion_matrices(tuned, X_test, y_test, name="13_tuned_confusion_matrices")

# %% [markdown]
# ## 6. Final model selection and threshold tuning
#
# The final model is the (model, strategy) pair with the best **cross-validated ROC-AUC on the training set**.
# The test set plays no part in the choice. The decision threshold is then tuned on **out-of-fold training
# predictions** and the test set is used once, to report final performance.

# %%
final_model, final_strategy = tuning.select_final(summary)
final_label = f"{final_model} ({final_strategy})"
final_search = searches[(final_model, final_strategy)]
final_pipe = final_search.best_estimator_
print(f"Final model: {final_label}")
print(f"Best params: {final_search.best_params_}")

# %%
oof = thresholds.out_of_fold_proba(final_pipe, X_train, y_train)

t_f1, _ = thresholds.best_threshold(y_train, oof, beta=1.0)
t_f2, _ = thresholds.best_threshold(y_train, oof, beta=2.0)
curve = thresholds.cost_curve(y_train, oof, config.COST_FALSE_NEGATIVE, config.COST_FALSE_POSITIVE)
t_cost = thresholds.cost_optimal_threshold(curve)

marks = {"F1-optimal": t_f1, "F2-optimal": t_f2, "cost-optimal": t_cost}
plots.plot_threshold_tradeoff(y_train, oof, marks, name="14_threshold_tradeoff")
plots.plot_cost_curve(curve, {"cost-optimal": t_cost, "F1-optimal": t_f1}, name="15_cost_curve")

# %%
operating_points = {
    "default 0.50": 0.5,
    "F1-optimal": t_f1,
    "F2-optimal": t_f2,
    "cost-optimal": t_cost,
}
threshold_table = evaluate.threshold_comparison(final_pipe, X_test, y_test, operating_points)
display(threshold_table)

# %% [markdown]
# We use the **F1-optimal threshold** as the default operating point (balanced precision and recall).
# If the retention offer is cheap relative to the value of a saved customer, the cost-optimal or F2-optimal
# threshold is preferable. The cost figures in `config.py` are illustrative assumptions and should be replaced
# with real business numbers.

# %%
threshold = t_f1
plots.plot_confusion_matrices(
    {final_label: final_pipe}, X_test, y_test, thresholds={final_label: threshold}, name="16_final_confusion_matrix"
)
plots.plot_calibration({final_label: final_pipe}, X_test, y_test, name="17_calibration")

# %% [markdown]
# ### 6.1 Lift and gains (how many churners do we catch by contacting the top-risk customers?)

# %%
test_prob = evaluate.positive_proba(final_pipe, X_test)
display(evaluate.decile_table(y_test, test_prob))

# %% [markdown]
# ## 7. Interpretation
# ### 7.1 Feature importance

# %%
builtin = interpret.builtin_importance(final_pipe)
display(builtin.head(15).to_frame("importance"))
plots.plot_ranked_bar(
    builtin, f"{final_model}: built-in importance (encoded features)", "Importance", name="18_builtin_importance"
)

# %%
perm = interpret.permutation_table(final_pipe, X_test, y_test, scoring="roc_auc", n_repeats=20)
display(perm)
plots.plot_ranked_bar(
    perm["importance_mean"],
    "Permutation importance (drop in test ROC-AUC)",
    "Mean decrease in ROC-AUC",
    name="19_permutation_importance",
)

# %%
try:
    shap_values, shap_X = interpret.compute_shap(final_pipe, X_test)
    plots.plot_shap_summary(shap_values, shap_X, name="20_shap_summary")
except Exception as exc:  # SHAP is an optional extra; never block the notebook on it
    print(f"SHAP skipped: {type(exc).__name__}: {exc}")

# %% [markdown]
# ### 7.2 Business insights

# %%
segments = {
    "Month-to-month contract": df["Contract"] == "Month-to-month",
    "Two-year contract": df["Contract"] == "Two year",
    "Tenure <= 12 months": df["tenure"] <= 12,
    "Fiber optic internet": df["InternetService"] == "Fiber optic",
    "Electronic check payment": df["PaymentMethod"] == "Electronic check",
    "No tech support": df["TechSupport"] == "No",
    "High-risk combo: M2M + fiber + tenure <= 12": (
        (df["Contract"] == "Month-to-month")
        & (df["InternetService"] == "Fiber optic")
        & (df["tenure"] <= 12)
    ),
}
display(eda.segment_table(df, segments))

# %% [markdown]
# **Insights to write up (confirm each against the tables and plots above)**
# - Contract type and tenure dominate: month-to-month customers in their first year churn at several times the base rate.
# - Fiber optic customers churn more despite paying more. This points to a price/value or service-quality problem, not just a pricing one.
# - Customers without tech support or online security churn more; bundling or promoting these add-ons is a candidate retention lever.
# - Auto-pay customers churn less than electronic-check customers.
# - `gender` shows no meaningful association with churn, so it should not drive any targeting.
# - The decile table quantifies how much better model-based targeting is than random outreach.

# %% [markdown]
# ## 8. Save the final model for the demo app

# %%
final_metrics = threshold_table.loc["F1-optimal"].to_dict()
artifacts.save_model(
    final_pipe,
    model_name=final_model,
    strategy=final_strategy,
    threshold=threshold,
    metrics=final_metrics,
    params=final_search.best_params_,
)

reloaded, card = artifacts.load_model()
assert np.allclose(reloaded.predict_proba(X_test)[:, 1], test_prob), "Reloaded model disagrees with in-memory model"
print("Saved and verified. Operating threshold:", round(card["threshold"], 3))

# %% [markdown]
# ## 9. Final model justification
#
# - **Why this model:** highest cross-validated ROC-AUC on training data (selected without touching the test set),
#   with test performance confirming it generalises (see section 5.2).
# - **Why these metrics:** with ~26% churners, ROC-AUC and PR-AUC measure ranking quality independent of threshold,
#   while recall/F1 at a tuned threshold measure the actual retention campaign.
# - **Why this threshold:** tuned on out-of-fold training predictions, then reported once on the test set.

# %% [markdown]
# ## 10. Limitations
# - Single snapshot, no time dimension: no real forecasting horizon, and no way to detect concept drift. Churn drivers may change with pricing or competition.
# - Features are account-level only (no usage, support-call, complaint or network-quality data), which caps achievable performance.
# - The dataset is a synthetic IBM sample, so absolute performance will not transfer to a real operator.
# - SMOTE interpolates one-hot and scaled features, creating synthetic points that are not valid customers; its benefit should be judged on CV, as done here.
# - Predicting churn is not the same as predicting who is *persuadable*. Ranking by churn probability may spend offers on customers who would have stayed.
# - Cost figures are assumptions. Test set has ~2,100 rows, so metric differences of about 1-2 points are within noise.
# - Fairness: `gender` and `SeniorCitizen` are protected-style attributes. A production system needs a fairness review before use.
#
# ## 11. Future improvements
# - Uplift modelling or A/B-tested retention offers to target persuadable customers.
# - Behavioural time-series features (usage trends, support tickets, billing disputes) and survival analysis for *time to churn*.
# - Probability calibration (isotonic/Platt) if probabilities will be used for expected-value calculations.
# - Cost-sensitive learning with real customer-lifetime-value figures; monitoring for drift; scheduled retraining.
