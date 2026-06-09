"""
data_pipeline.py — Data Preprocessing Module
=============================================
Handles:
  1. Loading raw CSV data
  2. Missing value imputation
  3. Label Encoding of categorical columns (category, gender, job)
  4. Min-Max Scaling of numerical columns (amt, unix_time)
  5. Dropping identifier columns that would leak info (first, last, street, trans_num, dob, cc_num as raw)

Design decision: We keep `cc_num` and `merchant` around *temporarily* because the
GraphEngine needs them to build the transaction graph. They are dropped only at the
Feature-Fusion stage inside model.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class DataPipeline:
    """End-to-end data preprocessing pipeline with strict OOP design."""

    # Columns that carry *no* predictive value / risk of overfitting
    _ID_COLUMNS: List[str] = [
        "Unnamed: 0", "first", "last", "street", "dob", "trans_num",
        "trans_date_trans_time",  # parsed from unix_time; raw string is not useful
    ]

    # Categorical columns to encode
    _CAT_COLUMNS: List[str] = ["category", "gender", "job", "state", "city"]

    # Numerical columns to scale via Min-Max
    _NUM_COLUMNS: List[str] = ["amt", "unix_time"]

    def __init__(self) -> None:
        # Fitted encoders / scalers — stored so they can be reused on new data
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame and perform initial sanity checks."""
        logger.info("Loading dataset from %s …", filepath)
        df = pd.read_csv(filepath, low_memory=False)
        logger.info("Loaded %d rows × %d columns.", *df.shape)
        return df

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe (straight from CSV).
        fit : bool
            If True, fit encoders/scalers on this data (training).
            If False, use already-fitted transformers (inference).

        Returns
        -------
        pd.DataFrame
            Cleaned, encoded, and scaled dataframe.
        """
        df = df.copy()

        # Step 1 — Drop identifier columns
        df = self._drop_identifiers(df)

        # Step 2 — Handle missing values
        df = self._handle_missing(df)

        # Step 3 — Encode categorical features
        df = self._encode_categoricals(df, fit=fit)

        # Step 4 — Scale numerical features
        df = self._scale_numericals(df, fit=fit)

        if fit:
            self._is_fitted = True

        logger.info("Preprocessing complete. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _drop_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that should not be used as features."""
        cols_to_drop = [c for c in self._ID_COLUMNS if c in df.columns]
        if cols_to_drop:
            logger.info("Dropping identifier columns: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)
        return df

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values.
        - Numerical columns → median
        - Categorical columns → mode (most frequent)
        """
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if cols_with_missing.empty:
            logger.info("No missing values detected.")
            return df

        logger.info("Missing values found:\n%s", cols_with_missing)
        for col in cols_with_missing.index:
            if df[col].dtype in ("float64", "int64", "float32", "int32"):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode specified categorical columns."""
        for col in self._CAT_COLUMNS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                # Handle unseen labels gracefully by fitting on all unique values
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                self._label_encoders[col] = le
                logger.info("Label-encoded '%s' (%d classes).", col, len(le.classes_))
            else:
                le = self._label_encoders[col]
                df[col] = df[col].astype(str)
                # Map unseen labels to -1 to avoid crashes on new data
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x, _k=known, _le=le: _le.transform([x])[0] if x in _k else -1)
        return df

    def _scale_numericals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Min-Max scale specified numerical columns."""
        cols_present = [c for c in self._NUM_COLUMNS if c in df.columns]
        if not cols_present:
            return df
        if fit:
            df[cols_present] = self._scaler.fit_transform(df[cols_present])
            logger.info("Min-Max scaled columns: %s", cols_present)
        else:
            df[cols_present] = self._scaler.transform(df[cols_present])
        return df
