"""
kaplan_meier.py

Students implement Kaplan-Meier survival analysis and log-rank test.
"""
from typing import Any, Dict
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def fit_kaplan_meier(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
) -> Dict[str, Any]:
    
    km_models = {}

    for group in data[group_col].dropna().unique():
        subset = data[data[group_col] == group]
        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], event_observed=subset[event_col], label=str(group))
        km_models[group] = kmf
    return km_models

def compute_logrank_test(
    data: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
) -> Dict[str, float]:
    
    groups = data[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Log-rank test requires exactly 2 groups")
    
    group1 = data[data[group_col] == groups[0]]
    group2 = data[data[group_col] == groups[1]]
    
    result = logrank_test(
        group1[time_col], group2[time_col],
        event_observed_A=group1[event_col],
        event_observed_B=group2[event_col]
    )
    
    return {
        "test_statistic": result.test_statistic,
        "p_value": result.p_value
    }
    

def plot_km_curves(
    km_models: Dict[str, Any],
    filename: str = "km_survival_plot.png",
    title: str = "Kaplan-Meier Survival Curves",
) -> None:
    plt.figure(figsize=(8, 6))
    for model in km_models.values():
        model.plot_survival_function(ci_show=True)
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend(title="Group")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename,dpi=300)
    plt.close()
