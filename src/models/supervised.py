"""Supervised Machine Learning & Deep Neural Network trainers for SecOps AutoML."""

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.interfaces.neural_net import BaseNeuralNetEngine
from src.interfaces.trainer import BaseModelTrainer
from src.utils.metrics import (
    evaluate_classification_metrics,
    evaluate_regression_metrics,
)


class LogisticRegressionTrainer(BaseModelTrainer):
    """L2-regularized Logistic Regression classifier for cyber threat detection."""

    def __init__(self, C: float = 1.0, max_iter: int = 500, random_state: int = 42) -> None:
        self._model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self._name = "Logistic Regression"
        self._problem_type = "classification"

    @property
    def name(self) -> str:
        return self._name

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogisticRegressionTrainer":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return self._model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        prob = self.predict_proba(X_test)
        return evaluate_classification_metrics(y_test.values, preds, prob)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def raw_model(self) -> Any:
        return self._model


class RandomForestClassifierTrainer(BaseModelTrainer):
    """Random Forest Classifier with decision tree ensemble for threat vector classification."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )
        self._name = "Random Forest Classifier"
        self._problem_type = "classification"

    @property
    def name(self) -> str:
        return self._name

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestClassifierTrainer":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return self._model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        prob = self.predict_proba(X_test)
        return evaluate_classification_metrics(y_test.values, preds, prob)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def raw_model(self) -> Any:
        return self._model


class XGBoostClassifierTrainer(BaseModelTrainer):
    """XGBoost Classifier for gradient-boosted decision trees."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42) -> None:
        self._name = "XGBoost Classifier"
        self._problem_type = "classification"
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
        self._model: Any = None
        self._label_map: Dict[Any, int] = {}
        self._inv_label_map: Dict[int, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostClassifierTrainer":
        # Encode string labels to integers if necessary
        unique_labels = sorted(pd.Series(y_train).unique())
        self._label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        self._inv_label_map = {i: lbl for i, lbl in enumerate(unique_labels)}
        y_encoded = pd.Series(y_train).map(self._label_map).values

        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier(
                n_estimators=self._params["n_estimators"],
                max_depth=self._params["max_depth"],
                learning_rate=self._params["learning_rate"],
                random_state=self._params["random_state"],
                eval_metric="logloss",
            )
            self._model.fit(X_train, y_encoded)
        except Exception:
            # Fallback to GradientBoostingClassifier from sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier(
                n_estimators=self._params["n_estimators"],
                max_depth=self._params["max_depth"],
                learning_rate=self._params["learning_rate"],
                random_state=self._params["random_state"],
            )
            self._model.fit(X_train, y_encoded)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_preds = self._model.predict(X)
        if self._inv_label_map:
            return np.array([self._inv_label_map.get(int(p), p) for p in raw_preds])
        return raw_preds

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        prob = self.predict_proba(X_test)
        return evaluate_classification_metrics(y_test.values, preds, prob)

    def get_params(self) -> Dict[str, Any]:
        return self._params

    @property
    def raw_model(self) -> Any:
        return self._model


class DeepMLPClassifierTrainer(BaseModelTrainer, BaseNeuralNetEngine):
    """Deep Multi-Layer Perceptron Neural Network for classification with PyTorch / MLP."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64),
        max_iter: int = 50,
        dropout_rate: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self._name = "Deep MLP Neural Network"
        self._problem_type = "classification"
        self._hidden_sizes = hidden_layer_sizes
        self._max_iter = max_iter
        self._dropout = dropout_rate
        self._random_state = random_state
        self._loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._torch_model: Any = None
        self._sklearn_fallback: Any = None
        self._classes: np.ndarray = np.array([])
        self._label_map: Dict[Any, int] = {}
        self._inv_label_map: Dict[int, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def architecture_name(self) -> str:
        return "Deep PyTorch MLP (ReLU + Dropout 0.2 + Adam)"

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        return self._train_torch(X_train, y_train, X_val, y_val, epochs, batch_size, lr)

    def _train_torch(
        self,
        X_train: np.ndarray,
        y_train: Any,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            torch.manual_seed(self._random_state)
            input_dim = X_train.shape[1]
            num_classes = len(self._classes)

            # Build PyTorch Module
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for h_dim in self._hidden_sizes:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self._dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, num_classes))

            self._torch_model = nn.Sequential(*layers)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=lr)

            # Tensor conversion
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self._loss_history = {"train_loss": [], "val_loss": []}

            for epoch in range(epochs):
                self._torch_model.train()
                epoch_loss = 0.0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = self._torch_model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(batch_x)

                epoch_train_loss = epoch_loss / len(X_train)
                self._loss_history["train_loss"].append(float(epoch_train_loss))

                # Validation loss if provided
                if X_val is not None and y_val is not None:
                    self._torch_model.eval()
                    with torch.no_grad():
                        val_out = self._torch_model(torch.tensor(X_val, dtype=torch.float32))
                        val_loss_val = criterion(val_out, torch.tensor(y_val, dtype=torch.long)).item()
                        self._loss_history["val_loss"].append(float(val_loss_val))

            return self._loss_history
        except Exception:
            # Fallback to sklearn MLP
            self._sklearn_fallback = MLPClassifier(
                hidden_layer_sizes=self._hidden_sizes,
                max_iter=epochs,
                random_state=self._random_state,
            )
            self._sklearn_fallback.fit(X_train, y_train)
            if hasattr(self._sklearn_fallback, "loss_curve_"):
                self._loss_history["train_loss"] = [float(l) for l in self._sklearn_fallback.loss_curve_]
            return self._loss_history

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "DeepMLPClassifierTrainer":
        self._classes = np.unique(y_train)
        self._label_map = {lbl: i for i, lbl in enumerate(self._classes)}
        self._inv_label_map = {i: lbl for i, lbl in enumerate(self._classes)}
        y_enc = np.array([self._label_map[y] for y in y_train])

        X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        self._train_torch(X_arr, y_enc, epochs=self._max_iter)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)
        if prob is not None:
            max_idx = np.argmax(prob, axis=1)
            return np.array([self._inv_label_map[i] for i in max_idx])
        return np.array([])

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if self._torch_model is not None:
            import torch
            import torch.nn.functional as F
            self._torch_model.eval()
            with torch.no_grad():
                logits = self._torch_model(torch.tensor(X_arr, dtype=torch.float32))
                probs = F.softmax(logits, dim=1).cpu().numpy()
                return probs
        elif self._sklearn_fallback is not None:
            return self._sklearn_fallback.predict_proba(X)
        return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        prob = self.predict_proba(X_test)
        return evaluate_classification_metrics(y_test.values, preds, prob)

    def get_loss_history(self) -> Dict[str, List[float]]:
        return self._loss_history

    def get_params(self) -> Dict[str, Any]:
        return {
            "hidden_layers": self._hidden_sizes,
            "dropout": self._dropout,
            "max_iter": self._max_iter,
        }

    @property
    def raw_model(self) -> Any:
        return self._torch_model if self._torch_model is not None else self._sklearn_fallback


# Regression Models
class LinearRegressionTrainer(BaseModelTrainer):
    """Ridge Linear Regression for continuous cyber threat metric forecasting."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self._model = Ridge(alpha=alpha, random_state=random_state)
        self._name = "Ridge Linear Regression"
        self._problem_type = "regression"

    @property
    def name(self) -> str:
        return self._name

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionTrainer":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        return evaluate_regression_metrics(y_test.values, preds)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def raw_model(self) -> Any:
        return self._model


class RandomForestRegressorTrainer(BaseModelTrainer):
    """Random Forest Regressor for continuous risk telemetry."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )
        self._name = "Random Forest Regressor"
        self._problem_type = "regression"

    @property
    def name(self) -> str:
        return self._name

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestRegressorTrainer":
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        return evaluate_regression_metrics(y_test.values, preds)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def raw_model(self) -> Any:
        return self._model


class DeepMLPRegressorTrainer(BaseModelTrainer, BaseNeuralNetEngine):
    """Deep Multi-Layer Perceptron Neural Network for regression."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64),
        max_iter: int = 50,
        random_state: int = 42,
    ) -> None:
        self._name = "Deep MLP Regressor"
        self._problem_type = "regression"
        self._hidden_sizes = hidden_layer_sizes
        self._max_iter = max_iter
        self._random_state = random_state
        self._loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._model: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def architecture_name(self) -> str:
        return "Deep PyTorch/Sklearn MLP Regressor (ReLU + Adam)"

    @property
    def problem_type(self) -> str:
        return self._problem_type

    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        self.fit(pd.DataFrame(X_train), pd.Series(y_train if y_train is not None else []))
        return self._loss_history

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "DeepMLPRegressorTrainer":
        self._model = MLPRegressor(
            hidden_layer_sizes=self._hidden_sizes,
            max_iter=self._max_iter,
            random_state=self._random_state,
        )
        self._model.fit(X_train, y_train)
        if hasattr(self._model, "loss_curve_"):
            self._loss_history["train_loss"] = [float(l) for l in self._model.loss_curve_]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        return evaluate_regression_metrics(y_test.values, preds)

    def get_loss_history(self) -> Dict[str, List[float]]:
        return self._loss_history

    def get_params(self) -> Dict[str, Any]:
        return {"hidden_layers": self._hidden_sizes, "max_iter": self._max_iter}

    @property
    def raw_model(self) -> Any:
        return self._model
