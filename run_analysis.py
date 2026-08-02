import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from students.random_survival_forest import (
    fit_random_survival_forest,
    compute_concordance_index,
    get_feature_importance,
    plot_feature_importance
)


# Load dataset
df = pd.read_excel(
    "data/RADCURE-clinical-data.xlsx"
)


# Convert survival status
df["Status"] = df["Status"].map({
    "Alive": 0,
    "Dead": 1
})


# Select variables
rsf_data = df[
    [
        "Length FU",
        "Status",
        "Age",
        "Smoking PY",
        "Stage",
        "Tx Modality",
        "Sex"
    ]
].copy()


# Convert numeric variables
rsf_data["Age"] = pd.to_numeric(
    rsf_data["Age"],
    errors="coerce"
)

rsf_data["Smoking PY"] = pd.to_numeric(
    rsf_data["Smoking PY"],
    errors="coerce"
)


# Remove missing values
rsf_data = rsf_data.dropna()


# Convert categorical variables
X = rsf_data.drop(
    columns=["Length FU", "Status"]
)

X = pd.get_dummies(
    X,
    drop_first=True
)


# Survival outcome required by scikit-survival
y = np.array(
    [
        (bool(event), time)
        for time, event in zip(
            rsf_data["Length FU"],
            rsf_data["Status"]
        )
    ],
    dtype=[
        ("event", "?"),
        ("time", "<f8")
    ]
)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Train RSF
rsf = fit_random_survival_forest(
    X_train,
    y_train,
    n_estimators=100
)


# C-index
c_index = compute_concordance_index(
    rsf,
    X_test,
    y_test
)

print("\nC-index:", c_index)


# Feature importance
importance = get_feature_importance(
    rsf,
    X.columns
)

print(importance)


# Save plot
os.makedirs(
    "outputs",
    exist_ok=True
)

plot_feature_importance(
    importance,
    filename="outputs/rsf_importance.png"
)