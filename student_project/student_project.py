# student_project/student_project.py
"""
Student starter (broken by design) for CS5100 Phase 1.
This file is intentionally TODO-heavy. Students must implement the functions
below to pass the public and hidden tests.

Design goals:
- Clear error messages (NotImplementedError) instead of silent wrong types.
- Helpful guidance in docstrings about expected behavior.
- Safe to import (no heavy compute at import time).
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
from collections import Counter

# -------------------------
# Section A: Data Loading
# -------------------------
def load_data(path=None):
    """
    Load the student dataset.

    Behavior expected by autograder / tests:
    - If path is None:
        prefer "student-mat-mini.csv" in repo root (fast), else
        prefer "datasets/student-mat-mini.csv", else
        fall back to "datasets/student-mat.csv" or "student-mat.csv".
    - Return: pandas.DataFrame

    NOTE TO STUDENTS: Implement this to read CSV using the correct separator
    (UCI full dataset uses ';'). If you don't have the full dataset locally,
    run the provided generator script to create the mini CSV.

    Currently this function is left as a TODO for you to implement.
    """
    # Helpful explicit error rather than returning None
    # Students should implement the loading logic described above.
    raise NotImplementedError(
        "load_data() is not implemented. Please load 'student-mat-mini.csv' (repo root) "
        "or 'datasets/student-mat-mini.csv' and return a pandas DataFrame. "
        "See scripts/generate_mini_dataset.py for generating the mini CSV."
    )


# -------------------------
# Section B: Exploratory / Preprocessing helpers
# -------------------------
def summary_stats():
    """
    Return a dictionary of summary statistics, e.g.:
        {"mean_G3": ..., "median_absences": ...}

    TODO: implement using load_data().
    """
    raise NotImplementedError("summary_stats() not implemented. Compute mean_G3 and median_absences.")


def compute_correlations():
    """
    Compute and return a pandas DataFrame of correlations (df.corr()) for numeric columns.

    TODO: implement using load_data().
    """
    raise NotImplementedError("compute_correlations() not implemented. Return df.corr(numeric_only=True).")


def preprocess_data(df):
    """
    Preprocess the provided DataFrame and return a processed DataFrame ready for modeling.

    Expected contract (must meet autograder checks):
    - Create target column 'at_risk' as: (df['G3'] < 10).astype(int)
    - Drop grade columns (G1, G2, G3) from the feature matrix to avoid leakage
    - Encode categorical variables (one-hot or similar) so NO object dtypes remain
    - Impute missing values
    - Scale numeric columns to [0,1] range
    - Return a pandas DataFrame that includes 'at_risk' and only numeric columns otherwise

    NOTE: Hidden tests assert target is exactly (G3 < 10) and will fail if you change it.
    """
    raise NotImplementedError("preprocess_data(df) not implemented. See docstring for expected contract.")


# -------------------------
# Section B: Gradient Boosting Pipeline (Broken starter)
# -------------------------
def train_gb_pipeline(X_train=None, y_train=None):
    """
    Build and fit a sklearn Pipeline that includes:
      ("preprocessor", ColumnTransformer(...)) and ("classifier", GradientBoostingClassifier)

    - Must return a fitted sklearn-like pipeline with .predict() and preferably .predict_proba()
    - Hidden tests expect a named step "preprocessor" to exist (if you return a Pipeline)

    TODO: implement. The starter returns an unfitted pipeline / raises if sklearn is missing.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception:
        raise NotImplementedError("sklearn not available in the environment; install dependencies.")

    # Intentionally bare-bones pipeline (not fitted) so students must build preprocessor + fit
    model = Pipeline([("classifier", GradientBoostingClassifier())])
    # The correct implementation should construct and include a named "preprocessor"
    # and call model.fit(X_train, y_train) before returning.
    raise NotImplementedError(
        "train_gb_pipeline() not implemented. Build a pipeline with a 'preprocessor' step and fit it."
    )


# -------------------------
# Section C: Random Forest (From Scratch) skeleton
# -------------------------
class DecisionTree:
    def __init__(self, max_depth=None):
        """
        Simple DecisionTree skeleton. Students may implement any reasonable tree
        representation (tuple/dict/class with 'predict' method).
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Student: implement recursive split building and store in self.tree"""
        raise NotImplementedError("DecisionTree.fit not implemented (student task)")

    def predict(self, X):
        """Student: implement prediction traversal using self.tree"""
        raise NotImplementedError("DecisionTree.predict not implemented (student task)")


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, sample_size=None, random_state=42):
        """
        RandomForest skeleton (bagging). Students must implement fit/predict.
        The autograder expects:
         - fit(X, y): populates self.trees (list of DecisionTree instances)
         - predict(X): returns a list/array of labels (same length as X)
        """
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """Student: implement bagging + DecisionTree training"""
        raise NotImplementedError("RandomForest.fit not implemented (student task)")

    def predict(self, X):
        """Student: implement majority-vote across self.trees"""
        raise NotImplementedError("RandomForest.predict not implemented (student task)")
