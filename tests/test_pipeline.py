"""Comprehensive test suite for the Adversarial-Resistant SecOps AutoML Platform."""

import io
import json
import numpy as np
import pandas as pd
import pytest

from src.adversarial.tester import AdversarialRobustnessTester
from src.cleaners.cleaner import DataCleaner
from src.mitre.generator import MitreRuleGenerator
from src.models.arena import AutoMLArena
from src.parsers.factory import ParserFactory
from src.security.privacy import PrivacyMasker
from src.security.sanitizer import sanitize_dataframe_text_vectors, sanitize_scalar_string
from src.security.validator import BufferValidator, SecurityValidationError
from src.xai.explainer import SHAPExplainer


def test_validator_and_security():
    """Verify in-memory buffer validation, file size limits, and sanitization."""
    validator = BufferValidator()

    # Empty buffer rejected
    with pytest.raises(SecurityValidationError):
        validator.validate(io.BytesIO(b""), "test.csv")

    # Disallowed extension rejected
    with pytest.raises(SecurityValidationError):
        validator.validate(io.BytesIO(b"test"), "test.exe")

    # Valid CSV buffer accepted
    valid_buf = io.BytesIO(b"col1,col2\n1,2\n3,4")
    ext = validator.validate(valid_buf, "data.csv", "text/csv")
    assert ext == "csv"

    # CSV formula injection neutralization
    assert sanitize_scalar_string("=cmd|'/C calc'!A0").startswith("'=")
    assert sanitize_scalar_string("+SUM(A1:A10)").startswith("'+")
    assert sanitize_scalar_string("<script>alert('xss')</script>") == ""


def test_zero_trust_privacy_masker():
    """Verify cryptographic masking and local de-masking for GDPR/SOC2 compliance."""
    masker = PrivacyMasker()

    sample_log = (
        "Incident: Compromised host 192.168.1.50 contacted external C2 at 203.0.113.19 "
        "using MAC 00:1B:44:11:3A:B7 and leaked to admin@corp.internal via https://intranet.corp.internal/upload"
    )

    masked_log = masker.mask_text(sample_log)

    # Assert internal entities are concealed
    assert "192.168.1.50" not in masked_log
    assert "INTERNAL_HOST" in masked_log
    assert "00:1B:44:11:3A:B7" not in masked_log
    assert "MAC_ADDR" in masked_log
    assert "admin@corp.internal" not in masked_log
    assert "ANON_USER" in masked_log
    assert "https://intranet.corp.internal" not in masked_log

    # Assert local de-masking restores original data
    demasked_log = masker.demask_text(masked_log)
    assert "192.168.1.50" in demasked_log
    assert "admin@corp.internal" in demasked_log
    assert "00:1B:44:11:3A:B7" in demasked_log


def test_parsers_in_memory():
    """Verify in-memory parsing for CSV and JSON without disk leaks."""
    # CSV parser
    csv_bytes = b"ip,port,action\n10.0.0.1,80,ALLOW\n10.0.0.2,443,ALLOW\n"
    csv_parser = ParserFactory.get_parser("csv")
    df_csv = csv_parser.parse(io.BytesIO(csv_bytes), "firewall.csv")
    assert len(df_csv) == 2
    assert "port" in df_csv.columns

    # JSON parser
    json_bytes = json.dumps([
        {"user": "alice", "score": 95.5},
        {"user": "bob", "score": 42.0},
    ]).encode("utf-8")
    json_parser = ParserFactory.get_parser("json")
    df_json = json_parser.parse(io.BytesIO(json_bytes), "logs.json")
    assert len(df_json) == 2
    assert "user" in df_json.columns


def test_cleaner_junk_filtering():
    """Verify deduplication, zero-variance pruning, and dynamic imputation."""
    cleaner = DataCleaner()

    raw_data = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 2.0, np.nan],
        "constant_junk": ["CONSTANT", "CONSTANT", "CONSTANT", "CONSTANT", "CONSTANT"],
        "category_col": ["TCP", "UDP", "TCP", "TCP", None],
    })
    # Add duplicate row
    raw_data = pd.concat([raw_data, raw_data.iloc[[0]]], ignore_index=True)

    cleaned_df, report = cleaner.clean(raw_data)

    assert report["duplicate_rows_dropped"] == 1
    assert "constant_junk" in report["dropped_columns"]
    assert "constant_junk" not in cleaned_df.columns
    assert cleaned_df["feature_a"].isna().sum() == 0
    assert cleaned_df["category_col"].isna().sum() == 0


def test_automl_arena_supervised_classification():
    """Verify Supervised Classification arena with ML and Deep Neural Net."""
    np.random.seed(42)
    df = pd.DataFrame({
        "url_len": np.random.randint(20, 150, 100),
        "special_chars": np.random.randint(0, 10, 100),
        "entropy": np.random.uniform(2.0, 5.0, 100),
        "label": ["Phishing" if i % 2 == 0 else "Legitimate" for i in range(100)],
    })

    arena = AutoMLArena(test_size=0.2, random_state=42)
    result = arena.run(df, target_column="label")

    assert result.problem_type == "classification"
    assert result.winning_model is not None
    assert len(result.leaderboard_df) >= 3
    assert result.winning_model_name in ("Logistic Regression", "Random Forest Classifier", "XGBoost Classifier", "Deep MLP Neural Network")


def test_automl_arena_unsupervised_autoencoder():
    """Verify Unsupervised Deep Autoencoder Anomaly Detection."""
    np.random.seed(42)
    normal_traffic = np.random.normal(loc=10.0, scale=2.0, size=(95, 4))
    # 5 extreme anomaly rows
    outlier_traffic = np.random.normal(loc=150.0, scale=20.0, size=(5, 4))
    data = np.vstack([normal_traffic, outlier_traffic])

    df = pd.DataFrame(data, columns=["bytes", "pkts", "duration", "rate"])

    arena = AutoMLArena(test_size=0.2, contamination=0.05, random_state=42)
    result = arena.run(df, target_column=None)

    assert result.problem_type == "unsupervised"
    assert result.winning_model_name == "Deep Autoencoder"
    assert result.anomaly_result is not None
    assert result.anomaly_result.anomaly_count == 5
    assert len(result.anomaly_result.deviation_percentages) == 5
    assert np.mean(result.anomaly_result.deviation_percentages) > 100.0


def test_adversarial_tester_and_xai():
    """Verify adversarial robustness evaluation and SHAP feature explanations."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feat1": np.random.randn(80),
        "feat2": np.random.randn(80),
        "label": [1 if i % 2 == 0 else 0 for i in range(80)],
    })

    arena = AutoMLArena(test_size=0.2, random_state=42)
    res = arena.run(df, target_column="label")

    # Adversarial test
    adv_tester = AdversarialRobustnessTester(epsilons=[0.0, 0.1, 0.2])
    adv_res = adv_tester.evaluate_evasion_robustness(res.winning_model, res.X_test, res.y_test)

    assert "evasion_resistance_score" in adv_res
    assert 0.0 <= adv_res["evasion_resistance_score"] <= 100.0
    assert "secops_warning" in adv_res

    # XAI SHAP explanation
    explainer = SHAPExplainer()
    xai_res = explainer.explain_global(res.winning_model, res.X_test)
    assert len(xai_res["top_features"]) > 0
    assert "plain_english_summary" in xai_res


def test_mitre_rule_generator():
    """Verify automated MITRE mapping, YARA, Snort, and iptables synthesis."""
    generator = MitreRuleGenerator()

    summary = {"winning_model": "Random Forest", "problem_type": "classification"}
    top_features = [{"feature": "destination_port_3389", "importance": 0.65}, {"feature": "packet_bytes", "importance": 0.25}]

    output = generator.generate(summary, top_features)

    assert len(output["mitre_mappings"]) > 0
    assert "rule" in output["yara_rule"].lower()
    assert "alert" in output["snort_signature"].lower()
    assert len(output["firewall_commands"]) > 0
    assert any("iptables" in cmd for cmd in output["firewall_commands"])
