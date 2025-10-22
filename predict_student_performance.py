"""
Predicting Student Performance
Author: Azizul Hakim (you can change the author line)
File: predict_student_performance.py

Usage:
    1) Put `datastudents_performance.csv.csv` in the same folder as this script.
    2) Install requirements: pip install -r requirements.txt
    3) Run: python predict_student_performance.py

What it does (summary):
    - Loads the dataset
    - Creates a binary target "high_performer" (average score >= 70)
    - Performs EDA prints (shape, class balance, basic stats)
    - Preprocesses features with a ColumnTransformer (one-hot for categoricals + scaler)
    - Trains Logistic Regression and RandomForest, compares them via cross-validation & on test set
    - Saves the best model to outputs/model.joblib
    - Exports feature importances (when model supports)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
print("Current working directory:", os.getcwd())


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: xgboost if installed (fast and often high-performing)
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False

# ----------------------------
# Configuration
# ----------------------------
DATA_FILE = "datastudents_performance.csv.csv"  # <- ensure this file exists in same folder
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1) Load data
# ----------------------------
print("Loading dataset:", DATA_FILE)
df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head(5).to_string(index=False))

# ----------------------------
# 2) Quick EDA
# ----------------------------
print("\n--- Basic info ---")
print(df.info())

print("\n--- Descriptive statistics (numeric) ---")
print(df.describe().T)

# create a target: average of math, reading, writing
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# Binary target: high performer if avg >= 70
df["high_performer"] = (df["avg_score"] >= 70).astype(int)

print("\nTarget distribution (high_performer):")
print(df["high_performer"].value_counts(normalize=True))

# ----------------------------
# 3) Feature selection & preprocessing
# ----------------------------
# We'll use some categorical features + numeric scores if desired.
# But to predict whether a student will be high performer from background features
# we exclude the raw scores as predictors when predicting high_performer.
features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    # If you want to predict the avg from scores, include numeric; here we predict class so exclude scores
]

# Confirm features exist
for f in features:
    if f not in df.columns:
        raise ValueError(f"Feature {f} not found in dataset")

X = df[features].copy()
y = df["high_performer"].copy()

# Show a quick categorical value counts
print("\nCategorical value counts (sample):")
for col in features:
    print(f"\n{col}:\n", df[col].value_counts().to_string())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# Preprocessing: OneHot encode categoricals
categorical_cols = features  # all features here are categorical
numeric_cols = []  # none in this setup

# OneHotEncoder with handle_unknown so pipeline is robust
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
        # ("num", StandardScaler(), numeric_cols)  # add if numeric features exist
    ],
    remainder="drop",
    sparse_threshold=0,
)

# ----------------------------
# 4) Modeling pipelines
# ----------------------------
models = {
    "logistic": Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]
    ),
}

if has_xgb:
    models["xgboost"] = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)),
        ]
    )

# ----------------------------
# 5) Cross-validation comparison
# ----------------------------
print("\nCross-validation (5-fold) results:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    cv_results[name] = scores.mean()
    print(f"  {name:15s} ROC-AUC mean: {scores.mean():.4f} (std {scores.std():.4f})")

# Choose best model by cv ROC-AUC
best_model_name = max(cv_results, key=cv_results.get)
print(f"\nBest model by CV ROC-AUC: {best_model_name}")

best_pipeline = models[best_model_name]
best_pipeline.fit(X_train, y_train)

# ----------------------------
# 6) Evaluate on test set
# ----------------------------
print("\nEvaluating on test set:")
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

if y_proba is not None:
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot ROC curve if probabilities available
if y_proba is not None:
    try:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve ({best_model_name})")
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{best_model_name}.png"))
        plt.close()
        print(f"ROC curve saved to {os.path.join(OUTPUT_DIR, f'roc_{best_model_name}.png')}")
    except Exception:
        pass

# ----------------------------
# 7) Feature importance (if available)
# ----------------------------
feature_names = None
try:
    # Extract feature names from OneHotEncoder
    ohe = best_pipeline.named_steps["pre"].named_transformers_["cat"]
    # When sparse=False used, get feature names
    try:
        cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    except Exception:
        # older OneHotEncoder versions
        cat_feature_names = []
        for i, col in enumerate(categorical_cols):
            cats = ohe.categories_[i]
            cat_feature_names.extend([f"{col}__{c}" for c in cats])
    feature_names = cat_feature_names  # no numeric features here
except Exception:
    feature_names = None

# If the classifier supports feature_importances_ (RandomForest / XGB)
try:
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_") and feature_names is not None:
        importances = clf.feature_importances_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
        print(f"Feature importances saved to {os.path.join(OUTPUT_DIR, 'feature_importances.csv')}")
except Exception:
    pass

# ----------------------------
# 8) Save model
# ----------------------------
model_path = os.path.join(OUTPUT_DIR, "model.joblib")
dump(best_pipeline, model_path)
print(f"Saved trained model to: {model_path}")

# ----------------------------
# 9) Quick inference demo
# ----------------------------
print("\nQuick inference with 3 examples from test set:")
sample = X_test.head(3).copy()
print(sample.to_string(index=False))
preds = best_pipeline.predict(sample)
probs = best_pipeline.predict_proba(sample)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
for i, pred in enumerate(preds):
    print(f"  Example {i+1} -> predicted high_performer: {pred}, prob: {probs[i] if probs is not None else 'n/a'}")

print("\nAll done. Check the outputs/ folder for model and any plots or feature importance.")
