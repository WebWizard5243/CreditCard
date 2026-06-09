"""
model.py — Feature Fusion & Random Forest Classifier
=====================================================
Responsibilities:
  1. Merge graph-derived features with processed transactional features
  2. Drop remaining identifiers (cc_num, merchant, zip, trans_date_trans_time)
  3. Train a Random Forest Classifier with class_weight='balanced'
  4. Evaluate with Precision, Recall, F1, Confusion Matrix, AUC-ROC
  5. Support cross-validation for robustness checks
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class FraudClassifier:
    """
    Random Forest-based fraud classifier with built-in feature fusion,
    class-imbalance handling, and evaluation utilities.
    """

    # Columns that must NOT enter the feature matrix
    _DROP_BEFORE_TRAIN: list[str] = [
        "cc_num", "merchant", "zip",
    ]

    TARGET = "is_fraud"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",      # handles extreme class imbalance
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # 1. Feature Fusion
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Drop identifiers, separate features (X) from label (y).

        Returns
        -------
        X : pd.DataFrame   — feature matrix
        y : pd.Series       — binary target
        """
        df = df.copy()

        cols_to_drop = [c for c in self._DROP_BEFORE_TRAIN if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

        if self.TARGET in df.columns:
            y = df.pop(self.TARGET)
        else:
            y = None
        
        X = df

        # Ensure no object columns remain (safety net)
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            logger.warning("Dropping leftover object columns: %s", obj_cols)
            X = X.drop(columns=obj_cols)

        self._feature_names = X.columns.tolist()
        logger.info("Feature matrix ready — %d features, %d samples.", X.shape[1], X.shape[0])
        return X, y

    # ------------------------------------------------------------------
    # 2. Train / Evaluate
    # ------------------------------------------------------------------
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        80/20 stratified split → train → evaluate → cross-validate.

        Returns a dict of evaluation artefacts consumed by the Streamlit UI.
        """
        # ---- Split --------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y,
        )
        logger.info("Train: %d | Test: %d  (fraud rate train=%.3f%%, test=%.3f%%)",
                     len(X_train), len(X_test),
                     y_train.mean() * 100, y_test.mean() * 100)

        # ---- Train --------------------------------------------------------
        logger.info("Training Random Forest …")
        self.model.fit(X_train, y_train)

        # ---- Predict -------------------------------------------------------
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # ---- Metrics -------------------------------------------------------
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(pr_recall, pr_precision)

        report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
        logger.info("\n%s", report)

        # ---- Cross-Validation  (stratified) --------------------------------
        logger.info("Running %d-fold stratified cross-validation …", cv_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1", n_jobs=1)
        logger.info("CV F1 scores: %s  |  mean=%.4f", cv_scores, cv_scores.mean())

        # ---- Feature Importance -------------------------------------------
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "pr_precision": pr_precision,
            "pr_recall": pr_recall,
            "classification_report": report,
            "cv_f1_scores": cv_scores,
            "cv_f1_mean": cv_scores.mean(),
            "feature_importance": feat_imp,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    # ------------------------------------------------------------------
    # 3. Predict on new data
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 / 1)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each sample."""
        return self.model.predict_proba(X)[:, 1]
