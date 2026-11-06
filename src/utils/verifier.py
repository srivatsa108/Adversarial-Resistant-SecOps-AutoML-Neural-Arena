"""Automated System & Invariant Verification Suite for AegisOps SecOps Platform."""

import time
from typing import Any, Dict, List

from tests.test_api import (
    test_health_endpoint,
    test_ingest_and_clean_endpoint,
)
from tests.test_pipeline import (
    test_adversarial_tester_and_xai,
    test_automl_arena_supervised_classification,
    test_automl_arena_unsupervised_autoencoder,
    test_cleaner_junk_filtering,
    test_mitre_rule_generator,
    test_parsers_in_memory,
    test_validator_and_security,
    test_zero_trust_privacy_masker,
)

VERIFICATION_CASES = [
    {
        "id": "SEC-01",
        "name": "BufferValidator & Quota Defense",
        "description": "In-memory BytesIO validation, 1 GB quota enforcement, magic-byte inspection, injection neutralization",
        "fn": test_validator_and_security,
    },
    {
        "id": "PRV-02",
        "name": "Zero-Trust Cryptographic Anonymizer",
        "description": "Deterministic RFC1918 IPv4/IPv6, MAC, and email substitution; in-memory vault with bidirectional de-masking",
        "fn": test_zero_trust_privacy_masker,
    },
    {
        "id": "ING-03",
        "name": "Multi-Format Ingestion Streaming",
        "description": "Native in-memory parsing of CSV, JSON, PDF tables, and PCAP/PCAPNG network packets",
        "fn": test_parsers_in_memory,
    },
    {
        "id": "CLN-04",
        "name": "Sanitization & Junk Pruning",
        "description": "Deduplication, zero-variance column dropping, dynamic median/mode imputation, formula sanitization",
        "fn": test_cleaner_junk_filtering,
    },
    {
        "id": "ML-05",
        "name": "Supervised Neural Arena",
        "description": "AutoML training of Logistic Regression, Random Forest, XGBoost, and PyTorch Deep MLP with Dropout",
        "fn": test_automl_arena_supervised_classification,
    },
    {
        "id": "ANO-06",
        "name": "Unsupervised Deep Autoencoder",
        "description": "Bottleneck PyTorch Autoencoder reconstruction error (MSE) outlier detection for zero-day threat discovery",
        "fn": test_automl_arena_unsupervised_autoencoder,
    },
    {
        "id": "ADV-07",
        "name": "Adversarial FGSM & Laplacian Stress Tester",
        "description": "Perturbation injection across epsilon range [0.0, 0.3]; Evasion Resistance Score calculation",
        "fn": test_adversarial_tester_and_xai,
    },
    {
        "id": "XAI-08",
        "name": "SHAP Explainability & Root Cause Mapping",
        "description": "Exact Shapley value attribution converted to plain-English operational security explanations",
        "fn": test_mitre_rule_generator,
    },
    {
        "id": "API-09",
        "name": "Decoupled FastAPI Health & Probe",
        "description": "Asynchronous REST OpenAPI probes verifying buffer quotas, CORS, and engine readiness",
        "fn": test_health_endpoint,
    },
    {
        "id": "API-10",
        "name": "End-to-End Ingest & Clean REST Pipeline",
        "description": "Full multipart in-memory upload, privacy masking, and sanitization cycle via REST endpoint",
        "fn": test_ingest_and_clean_endpoint,
    },
]


def run_all_verifications() -> List[Dict[str, Any]]:
    """Run all 10 invariant and security verification suites with precise millisecond profiling."""
    results = []
    for case in VERIFICATION_CASES:
        t0 = time.perf_counter()
        status = "PASSED"
        err_msg = ""
        try:
            case["fn"]()
        except Exception as e:
            status = "FAILED"
            err_msg = str(e)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        results.append({
            "id": case["id"],
            "name": case["name"],
            "description": case["description"],
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "error": err_msg,
        })
    return results
