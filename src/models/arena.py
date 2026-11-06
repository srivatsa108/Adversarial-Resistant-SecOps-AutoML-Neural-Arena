"""AutoML Arena orchestrator comparing classical ML and Deep Neural Networks."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cleaners.cleaner import DataCleaner
from src.config import GLOBAL_CONFIG
from src.interfaces.trainer import BaseModelTrainer
from src.models.autoencoder import AnomalyDetectionResult, DeepAutoencoderTrainer
from src.models.supervised import (
    DeepMLPClassifierTrainer,
    DeepMLPRegressorTrainer,
    LinearRegressionTrainer,
    LogisticRegressionTrainer,
    RandomForestClassifierTrainer,
    RandomForestRegressorTrainer,
    XGBoostClassifierTrainer,
)


@dataclass
class ArenaResult:
    """Consolidated outcome of the AutoML & Neural Arena benchmark."""

    problem_type: str  # 'classification', 'regression', or 'unsupervised'
    winning_model_name: str
    winning_model: Any
    leaderboard_df: pd.DataFrame
    model_metrics: Dict[str, Dict[str, float]]
    loss_curves: Dict[str, Dict[str, List[float]]]
    feature_names: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    anomaly_result: Optional[AnomalyDetectionResult] = None
    scaler: Optional[Any] = None


class AutoMLArena:
    """Orchestrates multi-model competitive arena for cybersecurity datasets."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = GLOBAL_CONFIG.DEFAULT_RANDOM_STATE,
        contamination: float = GLOBAL_CONFIG.DEFAULT_CONTAMINATION_RATE,
    ) -> None:
        self._test_size = test_size
        self._random_state = random_state
        self._contamination = contamination

    def determine_task_type(self, df: pd.DataFrame, target_column: Optional[str]) -> str:
        """Infer whether problem is Unsupervised Anomaly Detection, Classification, or Regression."""
        if not target_column or target_column.strip().lower() in ("none", "unlabeled", "none / unlabeled", ""):
            return "unsupervised"

        if target_column not in df.columns:
            return "unsupervised"

        target_series = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(target_series):
            unique_count = target_series.nunique()
            if unique_count <= GLOBAL_CONFIG.CLASSIFICATION_CARDINALITY_THRESHOLD:
                return "classification"
            return "regression"
        else:
            return "classification"

    def run(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        test_size: Optional[float] = None,
        contamination: Optional[float] = None,
    ) -> ArenaResult:
        """Execute automated model search, neural network training, and benchmark."""
        t_size = test_size if test_size is not None else self._test_size
        contam = contamination if contamination is not None else self._contamination

        task_type = self.determine_task_type(df, target_column)

        if task_type == "unsupervised":
            return self._run_unsupervised(df, contam)
        elif task_type == "classification":
            return self._run_classification(df, target_column, t_size)
        else:
            return self._run_regression(df, target_column, t_size)

    def _run_unsupervised(self, df: pd.DataFrame, contamination: float) -> ArenaResult:
        """Train Deep Autoencoder for zero-day log anomaly detection."""
        # Convert all features into scaled numeric space
        X_scaled, feat_names, scaler = DataCleaner.prepare_feature_matrix(df, fit_scaler=True)
        autoencoder = DeepAutoencoderTrainer(
            contamination=contamination, random_state=self._random_state
        )
        autoencoder.fit(X_scaled)
        anomaly_res = autoencoder.detect_anomalies(X_scaled, df, contamination=contamination)

        # Leaderboard representation for consistency
        leaderboard_data = [{
            "Model": "Deep Autoencoder",
            "Anomalies Flagged": anomaly_res.anomaly_count,
            "Contamination Rate": f"{contamination * 100:.1f}%",
            "Threshold MSE": f"{anomaly_res.reconstruction_threshold:.4f}",
            "Mean Deviation": f"+{np.mean(anomaly_res.deviation_percentages):.1f}%",
            "Status": "Optimal Neural Anomaly Detector",
        }]
        leaderboard_df = pd.DataFrame(leaderboard_data)

        # Build dummy split representation for downstream explainability & testing
        split_idx = int(len(X_scaled) * (1 - self._test_size))
        X_df = pd.DataFrame(X_scaled, columns=feat_names)

        return ArenaResult(
            problem_type="unsupervised",
            winning_model_name="Deep Autoencoder",
            winning_model=autoencoder,
            leaderboard_df=leaderboard_df,
            model_metrics={"Deep Autoencoder": {
                "anomaly_count": float(anomaly_res.anomaly_count),
                "total_logs": float(anomaly_res.total_logs),
                "contamination": contamination,
                "mean_deviation_pct": float(np.mean(anomaly_res.deviation_percentages)),
            }},
            loss_curves={"Deep Autoencoder": autoencoder.get_loss_history()},
            feature_names=feat_names,
            X_train=X_df.iloc[:split_idx],
            X_test=X_df.iloc[split_idx:],
            y_train=None,
            y_test=None,
            anomaly_result=anomaly_res,
            scaler=scaler,
        )

    def _run_classification(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float,
    ) -> ArenaResult:
        """Train and evaluate candidate classifiers and Deep Neural Networks."""
        y = df[target_column].copy()
        X_raw = df.drop(columns=[target_column])

        # Prepare dummy/scaled features
        X_scaled, feat_names, scaler = DataCleaner.prepare_feature_matrix(X_raw, fit_scaler=True)
        X_df = pd.DataFrame(X_scaled, columns=feat_names)

        # Stratified train/test split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=test_size, random_state=self._random_state, stratify=y
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=test_size, random_state=self._random_state
            )

        candidates: List[BaseModelTrainer] = [
            LogisticRegressionTrainer(random_state=self._random_state),
            RandomForestClassifierTrainer(n_estimators=50, max_depth=10, random_state=self._random_state),
            XGBoostClassifierTrainer(n_estimators=50, max_depth=5, random_state=self._random_state),
            DeepMLPClassifierTrainer(random_state=self._random_state),
        ]

        leaderboard_rows: List[Dict[str, Any]] = []
        metrics_dict: Dict[str, Dict[str, float]] = {}
        loss_curves: Dict[str, Dict[str, List[float]]] = {}
        trained_models: Dict[str, BaseModelTrainer] = {}

        best_score = -1.0
        winning_name = ""
        winning_model = None

        for trainer in candidates:
            try:
                trainer.fit(X_train, y_train)
                eval_metrics = trainer.evaluate(X_test, y_test)
                metrics_dict[trainer.name] = eval_metrics
                trained_models[trainer.name] = trainer

                if hasattr(trainer, "get_loss_history"):
                    history = trainer.get_loss_history()
                    if history and history.get("train_loss"):
                        loss_curves[trainer.name] = history

                f1 = eval_metrics.get("f1", 0.0)
                acc = eval_metrics.get("accuracy", 0.0)
                prec = eval_metrics.get("precision", 0.0)
                rec = eval_metrics.get("recall", 0.0)
                roc = eval_metrics.get("roc_auc", 0.5)

                leaderboard_rows.append({
                    "Model": trainer.name,
                    "F1 Score": f1,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "ROC-AUC": roc,
                })

                if f1 > best_score:
                    best_score = f1
                    winning_name = trainer.name
                    winning_model = trainer
            except Exception as e:
                leaderboard_rows.append({
                    "Model": trainer.name,
                    "F1 Score": 0.0,
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "ROC-AUC": 0.5,
                    "Error": str(e),
                })

        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

        return ArenaResult(
            problem_type="classification",
            winning_model_name=winning_name,
            winning_model=winning_model,
            leaderboard_df=leaderboard_df,
            model_metrics=metrics_dict,
            loss_curves=loss_curves,
            feature_names=feat_names,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
        )

    def _run_regression(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float,
    ) -> ArenaResult:
        """Train and evaluate candidate regressors and Deep Neural Networks."""
        y = df[target_column].astype(float).copy()
        X_raw = df.drop(columns=[target_column])

        X_scaled, feat_names, scaler = DataCleaner.prepare_feature_matrix(X_raw, fit_scaler=True)
        X_df = pd.DataFrame(X_scaled, columns=feat_names)

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=self._random_state
        )

        candidates: List[BaseModelTrainer] = [
            LinearRegressionTrainer(random_state=self._random_state),
            RandomForestRegressorTrainer(random_state=self._random_state),
            DeepMLPRegressorTrainer(random_state=self._random_state),
        ]

        leaderboard_rows: List[Dict[str, Any]] = []
        metrics_dict: Dict[str, Dict[str, float]] = {}
        loss_curves: Dict[str, Dict[str, List[float]]] = {}

        best_score = -1e9
        winning_name = ""
        winning_model = None

        for trainer in candidates:
            try:
                trainer.fit(X_train, y_train)
                eval_metrics = trainer.evaluate(X_test, y_test)
                metrics_dict[trainer.name] = eval_metrics

                if hasattr(trainer, "get_loss_history"):
                    history = trainer.get_loss_history()
                    if history and history.get("train_loss"):
                        loss_curves[trainer.name] = history

                r2 = eval_metrics.get("r2", 0.0)
                mae = eval_metrics.get("mae", 0.0)
                rmse = eval_metrics.get("rmse", 0.0)
                mse = eval_metrics.get("mse", 0.0)

                leaderboard_rows.append({
                    "Model": trainer.name,
                    "R² Score": r2,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MSE": mse,
                })

                if r2 > best_score:
                    best_score = r2
                    winning_name = trainer.name
                    winning_model = trainer
            except Exception as e:
                leaderboard_rows.append({
                    "Model": trainer.name,
                    "R² Score": -1.0,
                    "MAE": 999.0,
                    "RMSE": 999.0,
                    "MSE": 999.0,
                    "Error": str(e),
                })

        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(by="R² Score", ascending=False).reset_index(drop=True)

        return ArenaResult(
            problem_type="regression",
            winning_model_name=winning_name,
            winning_model=winning_model,
            leaderboard_df=leaderboard_df,
            model_metrics=metrics_dict,
            loss_curves=loss_curves,
            feature_names=feat_names,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
        )
