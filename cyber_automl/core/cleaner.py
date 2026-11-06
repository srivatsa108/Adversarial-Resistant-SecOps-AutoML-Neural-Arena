"""Data cleaning and junk metadata filtering module."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from cyber_automl.config import GLOBAL_CONFIG


@dataclass
class CleaningReport:
    """Summary metrics of cleaning transformations performed on the dataset."""

    initial_row_count: int
    initial_column_count: int
    cleaned_row_count: int
    cleaned_column_count: int
    duplicate_rows_dropped: int
    dropped_columns: List[str] = field(default_factory=list)
    dropped_reasons: Dict[str, str] = field(default_factory=dict)
    imputed_columns: Dict[str, str] = field(default_factory=dict)
    imputed_values_count: int = 0


class DataCleaner:
    """Production data cleaner enforcing deduplication, junk column pruning, and dynamic imputation."""

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
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Perform end-to-end cleaning and junk filtering on the DataFrame in-memory.

        Args:
            df: Raw input DataFrame.
            target_column: Optional target column name to protect from being pruned as junk.

        Returns:
            Tuple of (cleaned pd.DataFrame, CleaningReport summary).
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
            return cleaned_df, report

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

        return cleaned_df, report

    def _filter_junk_columns(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
        """Identify and drop zero/near-zero variance columns (junk metadata tags)."""
        dropped_cols: List[str] = []
        dropped_reasons: Dict[str, str] = []
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
                reasons_map[col] = f"Zero variance (single constant value: {series.iloc[0] if len(series) > 0 else 'empty'})"
                continue

            # Check numeric variance
            if pd.api.types.is_numeric_dtype(df[col]):
                var = float(series.var(ddof=0))
                if np.isnan(var) or var <= self._variance_threshold:
                    dropped_cols.append(col)
                    reasons_map[col] = f"Near-zero variance (variance = {var:.2e} <= {self._variance_threshold})"
                    continue
            else:
                # For categorical/string columns: near-zero variance if top value dominates (> 99.9%)
                top_freq = series.value_counts(normalize=True).iloc[0]
                if top_freq >= 0.999 and len(series) > 100:
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
