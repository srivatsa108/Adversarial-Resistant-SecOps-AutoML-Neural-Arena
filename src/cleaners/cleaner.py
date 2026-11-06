"""Data cleaning, junk metadata filtering, and neural network readiness module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import GLOBAL_CONFIG
from src.interfaces.cleaner import BaseDataCleaner


@dataclass
class CleaningReport:
    """Summary metrics of data cleaning transformations performed on the cybersecurity dataset."""

    initial_row_count: int
    initial_column_count: int
    cleaned_row_count: int
    cleaned_column_count: int
    duplicate_rows_dropped: int
    dropped_columns: List[str] = field(default_factory=list)
    dropped_reasons: Dict[str, str] = field(default_factory=dict)
    imputed_columns: Dict[str, str] = field(default_factory=dict)
    imputed_values_count: int = 0
    plain_english_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "initial_row_count": self.initial_row_count,
            "initial_column_count": self.initial_column_count,
            "cleaned_row_count": self.cleaned_row_count,
            "cleaned_column_count": self.cleaned_column_count,
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "dropped_columns": self.dropped_columns,
            "dropped_reasons": self.dropped_reasons,
            "imputed_columns": self.imputed_columns,
            "imputed_values_count": self.imputed_values_count,
            "plain_english_summary": self.plain_english_summary,
        }


class DataCleaner(BaseDataCleaner):
    """Production data cleaner enforcing deduplication, junk pruning, and dynamic imputation."""

    def __init__(
        self,
        variance_threshold: float = GLOBAL_CONFIG.DEFAULT_VARIANCE_THRESHOLD,
        numeric_impute_strategy: Literal["median", "mean"] = "median",
    ) -> None:
        """Initialize the cleaner.

        Args:
            variance_threshold: Threshold below which numeric variance is considered zero/near-zero.
            numeric_impute_strategy: 'median' (robust to outliers) or 'mean'.
        """
        self._variance_threshold = variance_threshold
        self._numeric_strategy = numeric_impute_strategy

    def clean(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform end-to-end cleaning and junk filtering on the DataFrame in-memory.

        Args:
            df: Raw input DataFrame.
            target_column: Optional target column name to protect from being pruned as junk.

        Returns:
            Tuple of (cleaned pd.DataFrame, report dictionary).
        """
        cleaned_df = df.copy()
        initial_rows, initial_cols = cleaned_df.shape

        report = CleaningReport(
            initial_row_count=initial_rows,
            initial_column_count=initial_cols,
            cleaned_row_count=initial_rows,
            cleaned_column_count=initial_cols,
            duplicate_rows_dropped=0,
        )

        if cleaned_df.empty:
            report.plain_english_summary = "Empty dataset provided. No records to clean."
            return cleaned_df, report.to_dict()

        # 1. Drop rows containing completely duplicate data
        before_dedup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(keep="first")
        duplicates_dropped = before_dedup - len(cleaned_df)
        report.duplicate_rows_dropped = duplicates_dropped

        # 2. Drop columns with zero or near-zero variance (junk metadata tags)
        cleaned_df, dropped_cols, dropped_reasons = self._filter_junk_columns(
            cleaned_df, target_column
        )
        report.dropped_columns = dropped_cols
        report.dropped_reasons = dropped_reasons

        # 3. Dynamic missing value imputation
        cleaned_df, imputed_summary, imputed_count = self._impute_missing_values(cleaned_df)
        report.imputed_columns = imputed_summary
        report.imputed_values_count = imputed_count

        # Final counts
        report.cleaned_row_count = len(cleaned_df)
        report.cleaned_column_count = len(cleaned_df.columns)

        # 4. Generate plain-English SecOps summary
        summary_parts = []
        if duplicates_dropped > 0:
            summary_parts.append(f"Purged {duplicates_dropped} redundant/duplicate log records")
        else:
            summary_parts.append("No duplicate records detected")

        if dropped_cols:
            col_list_str = ", ".join(f"'{c}'" for c in dropped_cols[:3])
            if len(dropped_cols) > 3:
                col_list_str += f" (+{len(dropped_cols) - 3} more)"
            summary_parts.append(f"stripped {len(dropped_cols)} uninformative/zero-variance metadata columns ({col_list_str})")
        else:
            summary_parts.append("retained all feature columns with positive variance")

        if imputed_count > 0:
            summary_parts.append(
                f"repaired {imputed_count} missing values across {len(imputed_summary)} columns via dynamic {self._numeric_strategy}/mode imputation"
            )
        else:
            summary_parts.append("dataset is 100% complete with zero null entries")

        report.plain_english_summary = (
            f"Zero-Trust Sanitization complete: {'; '.join(summary_parts)}. "
            f"Active analysis space: {report.cleaned_row_count} records × {report.cleaned_column_count} features."
        )

        return cleaned_df, report.to_dict()

    def _filter_junk_columns(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
        """Identify and drop zero/near-zero variance columns (junk metadata tags)."""
        dropped_cols: List[str] = []
        reasons_map: Dict[str, str] = {}
        cols_to_keep: List[str] = []

        for col in df.columns:
            if target_column and col == target_column:
                cols_to_keep.append(col)
                continue

            series = df[col].dropna()

            # If entirely NaN, drop
            if len(series) == 0:
                dropped_cols.append(col)
                reasons_map[col] = "All values are missing/NaN"
                continue

            # Check unique values count (0 variance)
            n_unique = series.nunique()
            if n_unique <= 1:
                dropped_cols.append(col)
                reasons_map[col] = f"Zero variance (single constant value: '{series.iloc[0] if len(series) > 0 else 'empty'}')"
                continue

            # Check numeric variance
            if pd.api.types.is_numeric_dtype(df[col]):
                var = float(series.var(ddof=0))
                if np.isnan(var) or var <= self._variance_threshold:
                    dropped_cols.append(col)
                    reasons_map[col] = f"Near-zero numeric variance ({var:.2e} <= {self._variance_threshold})"
                    continue
            else:
                # For categorical/string columns: near-zero variance if single class dominates > 99.9% in large sets
                if len(series) > 100:
                    top_freq = series.value_counts(normalize=True).iloc[0]
                    if top_freq >= 0.999:
                        dropped_cols.append(col)
                        reasons_map[col] = f"Near-zero categorical variance (dominant class = {top_freq:.2%})"
                        continue

            cols_to_keep.append(col)

        return df[cols_to_keep], dropped_cols, reasons_map

    def _impute_missing_values(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, str], int]:
        """Dynamically impute missing values: numeric -> mean/median, categorical -> mode."""
        df_imputed = df.copy()
        imputed_summary: Dict[str, str] = {}
        total_imputed = 0

        for col in df_imputed.columns:
            nan_count = df_imputed[col].isna().sum()
            if nan_count == 0:
                continue

            total_imputed += int(nan_count)

            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                if self._numeric_strategy == "mean":
                    val = df_imputed[col].mean()
                    imputed_summary[col] = f"mean ({val:.4f})"
                else:
                    val = df_imputed[col].median()
                    imputed_summary[col] = f"median ({val:.4f})"

                if pd.isna(val):
                    val = 0.0
                df_imputed[col] = df_imputed[col].fillna(val)
            else:
                mode_series = df_imputed[col].mode(dropna=True)
                val = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
                imputed_summary[col] = f"mode ('{val}')"
                df_imputed[col] = df_imputed[col].fillna(val)

        return df_imputed, imputed_summary, total_imputed

    @staticmethod
    def prepare_feature_matrix(
        X: pd.DataFrame,
        fit_scaler: bool = True,
        scaler: Optional[StandardScaler] = None,
        max_categories_per_col: int = 15,
        max_total_features: int = 128,
    ) -> Tuple[np.ndarray, List[str], StandardScaler]:
        """Preprocess feature DataFrame for Neural Network and ML readiness.

        Safely handles high-cardinality text/logs and categorical features without
        memory exhaustion (ArrayMemoryError) by using frequency binning, string-length
        feature extraction, and float32 memory bounds.

        Returns:
            Tuple of (scaled feature matrix np.ndarray, feature names list, fitted StandardScaler).
        """
        if X.empty:
            fitted_scaler = StandardScaler() if scaler is None else scaler
            return np.zeros((0, 1), dtype=np.float32), ["dummy_feature"], fitted_scaler

        processed_parts: List[pd.DataFrame] = []
        n_rows = len(X)

        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                med = float(s.median()) if not s.dropna().empty else 0.0
                num_series = s.fillna(med).astype(np.float32)
                processed_parts.append(num_series.to_frame(name=col))
            else:
                n_unique = s.nunique()
                cardinality_ratio = (n_unique / n_rows) if n_rows > 0 else 0.0

                # High-cardinality text, raw logs, UUIDs, or timestamps (>30 unique or >15% of dataset)
                if n_unique > 30 or cardinality_ratio > 0.15:
                    str_s = s.astype(str)
                    len_feature = str_s.str.len().astype(np.float32).to_frame(name=f"{col}_len")
                    digit_feature = str_s.str.count(r"\d").astype(np.float32).to_frame(name=f"{col}_digits")
                    processed_parts.extend([len_feature, digit_feature])
                else:
                    # Low-to-medium cardinality: keep top N categories + 'OTHER'
                    top_cats = s.value_counts().nlargest(max_categories_per_col).index
                    binned_s = s.where(s.isin(top_cats), other="OTHER")
                    dummies = pd.get_dummies(binned_s, prefix=col, drop_first=True, dtype=np.float32)
                    if not dummies.empty:
                        processed_parts.append(dummies)

        if not processed_parts:
            # Fallback if no valid features extracted
            X_encoded = pd.DataFrame({"constant_bias": np.ones(n_rows, dtype=np.float32)})
        else:
            X_encoded = pd.concat(processed_parts, axis=1)

        # Cap total features to safe memory bound
        if X_encoded.shape[1] > max_total_features:
            variances = X_encoded.var().sort_values(ascending=False)
            top_cols = variances.head(max_total_features).index
            X_encoded = X_encoded[top_cols]

        feature_names = list(X_encoded.columns)
        X_values = X_encoded.values.astype(np.float32)

        if fit_scaler or scaler is None:
            fitted_scaler = StandardScaler(copy=False)
            X_scaled = fitted_scaler.fit_transform(X_values)
        else:
            fitted_scaler = scaler
            X_scaled = fitted_scaler.transform(X_values)

        return X_scaled.astype(np.float32), feature_names, fitted_scaler
