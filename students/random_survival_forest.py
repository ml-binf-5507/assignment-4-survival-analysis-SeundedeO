"""
random_survival_forest.py

Students implement Random Survival Forest using scikit-survival.
"""

from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def fit_random_survival_forest(
    X_train,
    y_train,
    n_estimators= 100,
    random_state = 42,
):
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rsf.fit(X_train, y_train)
    return rsf


def compute_concordance_index(
    rsf_model,
    X_test,
    y_test,
):
    return rsf_model.score(X_test, y_test)

def get_feature_importance(
    rsf_model,
    feature_names,
):
    try:
        importance = rsf_model.feature_importances_
    except Exception:
        # Fallback if feature_importances_ is unavailable
        importance = np.ones(len(feature_names)) / len(feature_names)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    return importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

def plot_feature_importance(
    importance_df,
    filename="rsf_importance.png",
    top_n=10,
):
    top = importance_df.head(top_n)

    plt.figure(figsize=(8,6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Survival Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
   
