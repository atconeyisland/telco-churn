from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missing counts and cardinality."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    )


def iqr_outliers(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Tukey-fence outlier summary for numeric columns."""
    rows = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "lower_fence": lower,
                "upper_fence": upper,
                "n_outliers": n_out,
                "pct_outliers": 100 * n_out / len(s),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def tenure_bucket(tenure: pd.Series) -> pd.Series:
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013) between two categorical variables."""
    table = pd.crosstab(x, y)
    r, k = table.shape
    if r < 2 or k < 2:
        return 0.0
    n = table.to_numpy().sum()
    chi2 = chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def chi2_table(df: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    """Chi-square test and Cramer's V of each categorical feature against the target."""
    rows = []
    for col in cols:
        table = pd.crosstab(df[col], df[target])
        chi2, p_value, dof, _ = chi2_contingency(table)
        rows.append(
            {
                "feature": col,
                "chi2": chi2,
                "p_value": p_value,
                "dof": dof,
                "cramers_v": cramers_v(df[col], df[target]),
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("cramers_v", ascending=False)


def churn_rate_by(df: pd.DataFrame, col: str, target: str = "Churn") -> pd.DataFrame:
    grouped = df.groupby(col, observed=True)[target].agg(
        customers="size", churn_rate=lambda s: (s == "Yes").mean()
    )
    return grouped.sort_values("churn_rate", ascending=False)


def segment_table(
    df: pd.DataFrame, segments: dict[str, pd.Series], target: str = "Churn"
) -> pd.DataFrame:
    """Size, churn rate and lift (vs overall churn rate) for boolean-mask segments."""
    base = (df[target] == "Yes").mean()
    rows = []
    for name, mask in segments.items():
        rate = (df.loc[mask, target] == "Yes").mean()
        rows.append(
            {
                "segment": name,
                "customers": int(mask.sum()),
                "share_of_base": mask.mean(),
                "churn_rate": rate,
                "lift_vs_overall": rate / base,
            }
        )
    return pd.DataFrame(rows).set_index("segment")
