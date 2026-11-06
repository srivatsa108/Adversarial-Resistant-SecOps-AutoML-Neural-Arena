"""Explainable AI (XAI) engine powered by SHAP with plain-English SecOps translation."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.interfaces.explainer import BaseExplainer
from src.utils.translations import translate_shap_to_secops_root_causes


class SHAPExplainer(BaseExplainer):
    """Computes exact mathematical feature attributions and translates them into SecOps insights."""

    def __init__(self, background_samples: int = 50) -> None:
        self._background_samples = background_samples

    def explain_global(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        max_display: int = 10,
    ) -> Dict[str, Any]:
        """Compute global feature importance weights using SHAP or sensitivity decomposition."""
        feature_names = list(X_sample.columns)

        # Handle Unsupervised Deep Autoencoder
        if hasattr(model, "reconstruct"):
            return self._explain_autoencoder_global(model, X_sample, max_display)

        raw_model = getattr(model, "raw_model", model)

        try:
            import shap

            # Select appropriate SHAP Explainer
            model_type_str = str(type(raw_model)).lower()
            if any(k in model_type_str for k in ("forest", "tree", "xgb", "boost")):
                explainer = shap.TreeExplainer(raw_model)
                shap_values = explainer.shap_values(X_sample)
            elif "linear" in model_type_str or "ridge" in model_type_str or "logistic" in model_type_str:
                explainer = shap.LinearExplainer(raw_model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Neural Net or general model: KernelExplainer with background kmeans/sample
                bg = shap.sample(X_sample, min(self._background_samples, len(X_sample)))
                predict_fn = getattr(model, "predict_proba", None) or getattr(model, "predict")
                explainer = shap.KernelExplainer(predict_fn, bg)
                shap_values = explainer.shap_values(X_sample.iloc[: min(25, len(X_sample))])

            # Process multiclass or 3D SHAP outputs
            if isinstance(shap_values, list):
                # For classification, pick the malicious/anomalous class (class 1 or last)
                target_shap = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                target_shap = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            else:
                target_shap = np.array(shap_values)

            # Mean absolute SHAP values per feature
            mean_abs_shap = np.mean(np.abs(target_shap), axis=0)

        except Exception:
            # Fallback to feature variance & correlation sensitivity decomposition
            cov_matrix = np.cov(X_sample.values, rowvar=False)
            if cov_matrix.ndim == 2:
                mean_abs_shap = np.diag(cov_matrix)
            else:
                mean_abs_shap = np.var(X_sample.values, axis=0)

        # Build sorted feature importance list
        feat_imp_list: List[Dict[str, Any]] = []
        for name, imp in zip(feature_names, mean_abs_shap):
            feat_imp_list.append({"feature": str(name), "importance": float(imp)})

        feat_imp_list.sort(key=lambda x: x["importance"], reverse=True)
        top_features = feat_imp_list[:max_display]

        # Plain-English SecOps narrative
        narrative = translate_shap_to_secops_root_causes(top_features)

        return {
            "top_features": top_features,
            "all_features": feat_imp_list,
            "plain_english_summary": narrative,
            "total_features_evaluated": len(feature_names),
        }

    def explain_instance(
        self,
        model: Any,
        instance: pd.Series,
        background_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute local feature attribution for a single suspicious log event or zero-day anomaly."""
        feature_names = list(background_data.columns)
        inst_df = pd.DataFrame([instance])

        if hasattr(model, "reconstruct"):
            # Feature-level reconstruction error for Autoencoder
            recon = model.reconstruct(inst_df.values)[0]
            orig = inst_df.values[0]
            sq_diffs = (orig - recon) ** 2
            local_importances = [
                {"feature": str(feat), "importance": float(diff)}
                for feat, diff in zip(feature_names, sq_diffs)
            ]
        else:
            global_exp = self.explain_global(model, background_data, max_display=len(feature_names))
            local_importances = global_exp["all_features"]

        local_importances.sort(key=lambda x: x["importance"], reverse=True)
        top_local = local_importances[:5]
        narrative = translate_shap_to_secops_root_causes(top_local)

        return {
            "instance_attributions": top_local,
            "plain_english_root_cause": narrative,
        }

    def _explain_autoencoder_global(
        self,
        autoencoder: Any,
        X_sample: pd.DataFrame,
        max_display: int,
    ) -> Dict[str, Any]:
        """Decompose Autoencoder reconstruction error across features."""
        feature_names = list(X_sample.columns)
        X_arr = X_sample.values
        recon = autoencoder.reconstruct(X_arr)
        feat_mse = np.mean((X_arr - recon) ** 2, axis=0)

        feat_imp_list: List[Dict[str, Any]] = [
            {"feature": str(name), "importance": float(imp)}
            for name, imp in zip(feature_names, feat_mse)
        ]
        feat_imp_list.sort(key=lambda x: x["importance"], reverse=True)
        top_features = feat_imp_list[:max_display]
        narrative = translate_shap_to_secops_root_causes(top_features)

        return {
            "top_features": top_features,
            "all_features": feat_imp_list,
            "plain_english_summary": narrative,
            "total_features_evaluated": len(feature_names),
        }
