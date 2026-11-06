"""Deep Autoencoder Neural Network for unsupervised zero-day cyber log anomaly detection."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.interfaces.neural_net import BaseNeuralNetEngine
from src.utils.metrics import compute_reconstruction_error


@dataclass
class AnomalyDetectionResult:
    """Detailed output of Deep Autoencoder log anomaly detection."""

    total_logs: int
    anomaly_count: int
    contamination_rate: float
    reconstruction_threshold: float
    mean_baseline_error: float
    reconstruction_errors: np.ndarray
    anomaly_indices: List[int]
    anomalies_df: pd.DataFrame
    deviation_percentages: List[float]
    plain_english_summary: str
    secops_threat_level: str


class DeepAutoencoderTrainer(BaseNeuralNetEngine):
    """Deep Autoencoder neural network detecting zero-day cybersecurity anomalies."""

    def __init__(
        self,
        bottleneck_ratio: float = 0.35,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._bottleneck_ratio = bottleneck_ratio
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._contamination = contamination
        self._random_state = random_state
        self._loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._torch_model: Any = None
        self._sklearn_fallback: Any = None
        self._input_dim: int = 0
        self._feature_names: List[str] = []

    @property
    def architecture_name(self) -> str:
        return f"Deep Autoencoder (Compression ratio: {self._bottleneck_ratio:.2f}, Adam, ReLU)"

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
        return self.fit(X_train, epochs=epochs, batch_size=batch_size, lr=lr)

    def fit(
        self,
        X: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """Fit the autoencoder to compress and reconstruct unlabelled log vectors."""
        n_epochs = epochs or self._epochs
        b_size = batch_size or self._batch_size
        l_rate = lr or self._lr

        self._input_dim = X.shape[1]
        bottleneck_dim = max(2, int(self._input_dim * self._bottleneck_ratio))
        hidden_dim = max(bottleneck_dim + 2, min(128, self._input_dim * 2))

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            torch.manual_seed(self._random_state)

            # PyTorch Deep Autoencoder: Symmetric Encoder-Decoder
            class AutoencoderNet(nn.Module):
                def __init__(self, in_d: int, hid_d: int, bot_d: int) -> None:
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(in_d, hid_d),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hid_d, bot_d),
                        nn.ReLU(),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(bot_d, hid_d),
                        nn.ReLU(),
                        nn.Linear(hid_d, in_d),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.decoder(self.encoder(x))

            self._torch_model = AutoencoderNet(self._input_dim, hidden_dim, bottleneck_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=l_rate)

            x_tensor = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, x_tensor)
            loader = DataLoader(dataset, batch_size=b_size, shuffle=True)

            self._loss_history = {"train_loss": []}

            for epoch in range(n_epochs):
                self._torch_model.train()
                total_loss = 0.0
                for bx, target in loader:
                    optimizer.step()
                    optimizer.zero_grad()
                    out = self._torch_model(bx)
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(bx)

                avg_loss = total_loss / len(X)
                self._loss_history["train_loss"].append(float(avg_loss))

            return self._loss_history

        except Exception:
            # Fallback to Scikit-Learn MLPRegressor as an Autoencoder (predicting X from X)
            from sklearn.neural_network import MLPRegressor

            self._sklearn_fallback = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, bottleneck_dim, hidden_dim),
                max_iter=n_epochs,
                random_state=self._random_state,
            )
            self._sklearn_fallback.fit(X, X)
            if hasattr(self._sklearn_fallback, "loss_curve_"):
                self._loss_history["train_loss"] = [float(l) for l in self._sklearn_fallback.loss_curve_]
            return self._loss_history

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct the feature matrix through the bottleneck."""
        if self._torch_model is not None:
            import torch
            self._torch_model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                recon = self._torch_model(tensor_x).cpu().numpy()
                return recon
        elif self._sklearn_fallback is not None:
            return self._sklearn_fallback.predict(X)
        else:
            raise RuntimeError("Autoencoder model has not been trained.")

    def detect_anomalies(
        self,
        X_scaled: np.ndarray,
        original_df: pd.DataFrame,
        contamination: Optional[float] = None,
    ) -> AnomalyDetectionResult:
        """Identify high-risk zero-day log anomalies based on reconstruction errors."""
        rate = contamination if contamination is not None else self._contamination
        reconstructed = self.reconstruct(X_scaled)
        errors = compute_reconstruction_error(X_scaled, reconstructed)

        # Dynamic cutoff threshold at (100 - contamination * 100) percentile
        percentile_cutoff = (1.0 - rate) * 100.0
        threshold = float(np.percentile(errors, percentile_cutoff))

        # Identify anomalies
        anomaly_mask = errors >= threshold
        anomaly_indices = list(np.where(anomaly_mask)[0])

        # Baseline error of normal traffic
        normal_errors = errors[~anomaly_mask]
        baseline_mean = float(np.mean(normal_errors)) if len(normal_errors) > 0 else float(np.mean(errors))
        if baseline_mean <= 1e-8:
            baseline_mean = 1e-4

        # Calculate deviation percentage for each anomaly
        deviations: List[float] = []
        for idx in anomaly_indices:
            row_err = errors[idx]
            dev_pct = float(((row_err - baseline_mean) / baseline_mean) * 100.0)
            deviations.append(max(0.0, dev_pct))

        # Extract anomalies dataframe with added diagnostic columns
        anomalies_df = original_df.iloc[anomaly_indices].copy()
        anomalies_df["_reconstruction_error_mse"] = errors[anomaly_indices]
        anomalies_df["_baseline_deviation_pct"] = deviations

        mean_dev = float(np.mean(deviations)) if deviations else 0.0

        threat_level = "CRITICAL" if mean_dev >= 300.0 or rate > 0.08 else "HIGH" if mean_dev >= 150.0 else "MEDIUM"

        summary = (
            f"Deep Autoencoder flagged {len(anomaly_indices)} zero-day anomaly events "
            f"({rate * 100:.1f}% contamination quota) surpassing reconstruction error threshold of {threshold:.4f}. "
            f"Anomalous telemetry exhibits an average +{mean_dev:.1f}% structural deviation from normal baseline traffic."
        )

        return AnomalyDetectionResult(
            total_logs=len(original_df),
            anomaly_count=len(anomaly_indices),
            contamination_rate=rate,
            reconstruction_threshold=threshold,
            mean_baseline_error=baseline_mean,
            reconstruction_errors=errors,
            anomaly_indices=anomaly_indices,
            anomalies_df=anomalies_df,
            deviation_percentages=deviations,
            plain_english_summary=summary,
            secops_threat_level=threat_level,
        )

    def get_loss_history(self) -> Dict[str, List[float]]:
        return self._loss_history
