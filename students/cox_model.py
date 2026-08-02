"""
cox_model.py

Students implement Cox Proportional Hazards regression.
"""

from typing import Any, Dict, List, Tuple
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import pandas as pd
import numpy as np


def fit_cox_model(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: List[str],
) -> Any:

    if len(covariates) < 3:
        raise ValueError("Cox model must include at least 3 covariates")

    # Keep only the variables used in the model
    df = data[[time_col, event_col] + covariates].copy()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    cph = CoxPHFitter()
    cph.fit(df, duration_col=time_col, event_col=event_col)

    # Save the processed dataframe for PH testing
    cph.training_df = df

    return cph


def extract_cox_summary(cox_model) :
    summary = cox_model.summary.reset_index()

    return summary.rename(columns={
        "covariate": "covariate",
        "coef": "coef",
        "exp(coef)": "exp(coef)",
        "se(coef)": "se(coef)",
        "z": "z",
        "p": "p",
        "exp(coef) lower 95%": "lower 95%",
        "exp(coef) upper 95%": "upper 95%"
    })
   

def test_proportional_hazards(
    cox_model,
    data,
    time_col,
    event_col,
):

    # Use the same dataframe that was used to fit the model
    results = proportional_hazard_test(
        cox_model,
        cox_model.training_df,
        time_transform="rank"
    )

    ph = {}

    for cov in results.summary.index:
        ph[cov] = {
            "test_statistic": float(results.summary.loc[cov, "test_statistic"]),
            "p_value": float(results.summary.loc[cov, "p"])
        }

    return ph
