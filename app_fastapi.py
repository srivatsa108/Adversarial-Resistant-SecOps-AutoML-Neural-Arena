"""Decoupled FastAPI Backend for Adversarial-Resistant SecOps AutoML Platform.

Provides fully documented asynchronous OpenAPI/Swagger endpoints with CORS
support for modern React/Next.js/Vue frontends.
"""

import io
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.adversarial.tester import AdversarialRobustnessTester
from src.cleaners.cleaner import DataCleaner
from src.config import GLOBAL_CONFIG
from src.mitre.generator import MitreRuleGenerator
from src.models.arena import AutoMLArena
from src.parsers.factory import ParserFactory
from src.security.privacy import PrivacyMasker
from src.security.validator import BufferValidator, SecurityValidationError
from src.xai.explainer import SHAPExplainer

app = FastAPI(
    title="SecOps AutoML & Deep Neural Network Platform API",
    description=(
        "Production-ready, Adversarial-Resistant, Explainable Automated Machine Learning (AutoML) "
        "and Deep Neural Network engine for cybersecurity telemetry, integrated with Google Antigravity Agent."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for modern web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory singletons and session cache
validator = BufferValidator()
privacy_engine = PrivacyMasker()
data_cleaner = DataCleaner()
arena = AutoMLArena()
adversarial_tester = AdversarialRobustnessTester()
explainer = SHAPExplainer()
mitre_gen = MitreRuleGenerator(privacy_masker=privacy_engine)

# In-memory session store (keyed by session_id or 'default')
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str = "default") -> Dict[str, Any]:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = {
            "df": None,
            "cleaned_df": None,
            "target_col": None,
            "arena_result": None,
            "privacy_engine": PrivacyMasker(),
        }
    return _SESSION_STORE[session_id]


# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    zero_trust_privacy: str = "active"
    adversarial_defense: str = "active"
    hardware_acceleration: str = "cpu/mps"


class TrainRequest(BaseModel):
    session_id: str = "default"
    target_column: Optional[str] = None
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    contamination: float = Field(0.05, ge=0.01, le=0.25)


class AdversarialTestRequest(BaseModel):
    session_id: str = "default"
    epsilons: Optional[List[float]] = None


class MitreGenerateRequest(BaseModel):
    session_id: str = "default"
    api_key: Optional[str] = None


class DemaskRequest(BaseModel):
    session_id: str = "default"
    payload: Dict[str, Any]


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System Health"])
async def health_check():
    """Check API health and operational status."""
    return HealthResponse()


@app.post("/api/v1/ingest", tags=["Data Ingestion & Privacy"])
async def ingest_file(
    file: UploadFile = File(...),
    enable_zero_trust_masking: bool = Form(True),
    session_id: str = Form("default"),
):
    """Securely ingest CSV, JSON, or PDF files entirely in-memory with Zero-Trust privacy masking."""
    try:
        content = await file.read()
        buffer = io.BytesIO(content)

        # In-memory validation
        ext = validator.validate(buffer, file.filename or "uploaded_data.csv", file.content_type)
        parser = ParserFactory.get_parser(ext)
        df = parser.parse(buffer, file.filename or "uploaded_data")

        session = get_session(session_id)
        local_masker = session["privacy_engine"]

        audit_info = {}
        if enable_zero_trust_masking:
            df, audit_info = local_masker.mask_dataframe(df)

        session["df"] = df

        preview_records = df.head(10).to_dict(orient="records")

        return {
            "status": "success",
            "filename": file.filename,
            "detected_format": ext,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "zero_trust_masking_applied": enable_zero_trust_masking,
            "privacy_audit": audit_info,
            "preview": preview_records,
        }
    except SecurityValidationError as sve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(sve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {e}")


@app.post("/api/v1/clean", tags=["Preprocessing & Sanitization"])
async def clean_dataset(
    session_id: str = Form("default"),
    target_column: Optional[str] = Form(None),
):
    """Deduplicate rows, purge zero-variance junk columns, and impute missing values."""
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No dataset uploaded in this session.")

    cleaned_df, report = data_cleaner.clean(session["df"], target_column)
    session["cleaned_df"] = cleaned_df
    session["target_col"] = target_column

    return {
        "status": "success",
        "cleaning_report": report,
        "active_features": list(cleaned_df.columns),
    }


@app.post("/api/v1/train", tags=["AutoML & Neural Arena"])
async def train_arena(req: TrainRequest):
    """Execute competitive AutoML & Deep Neural Network arena benchmark."""
    session = get_session(req.session_id)
    df = session["cleaned_df"] if session["cleaned_df"] is not None else session["df"]
    if df is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No dataset found. Ingest a file first.")

    try:
        result = arena.run(
            df=df,
            target_column=req.target_column,
            test_size=req.test_size,
            contamination=req.contamination,
        )
        session["arena_result"] = result
        session["target_col"] = req.target_column

        leaderboard = result.leaderboard_df.to_dict(orient="records")

        return {
            "status": "success",
            "problem_type": result.problem_type,
            "winning_model": result.winning_model_name,
            "leaderboard": leaderboard,
            "model_metrics": result.model_metrics,
            "loss_history": result.loss_curves,
            "features_used": result.feature_names,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Training failed: {e}")


@app.post("/api/v1/adversarial-test", tags=["Adversarial Stress Testing"])
async def run_adversarial_test(req: AdversarialTestRequest):
    """Subject winning model to FGSM/PGD evasion perturbation attacks."""
    session = get_session(req.session_id)
    res = session["arena_result"]
    if res is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No trained model in this session.")

    try:
        adv_res = adversarial_tester.evaluate_evasion_robustness(
            model=res.winning_model,
            X_test=res.X_test,
            y_test=res.y_test,
            epsilons=req.epsilons,
        )
        return {
            "status": "success",
            "evasion_resistance_score": adv_res["evasion_resistance_score"],
            "decay_curve": adv_res["decay_curve"],
            "secops_warning": adv_res["secops_warning"],
            "threat_badge": adv_res["threat_badge"],
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Adversarial test error: {e}")


@app.post("/api/v1/explain", tags=["Explainable AI (XAI)"])
async def explain_model(session_id: str = "default"):
    """Compute exact SHAP feature attributions and translate into plain-English root causes."""
    session = get_session(session_id)
    res = session["arena_result"]
    if res is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No trained model in this session.")

    try:
        xai_out = explainer.explain_global(res.winning_model, res.X_test)
        return {
            "status": "success",
            "top_features": xai_out["top_features"],
            "plain_english_summary": xai_out["plain_english_summary"],
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"XAI explanation failed: {e}")


@app.post("/api/v1/generate-mitre-rules", tags=["MITRE ATT&CK & Rule Synthesis"])
async def generate_mitre_rules(req: MitreGenerateRequest):
    """Synthesize MITRE ATT&CK mappings, YARA rules, Snort signatures, and iptables."""
    session = get_session(req.session_id)
    res = session["arena_result"]
    if res is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No trained model in this session.")

    try:
        xai_out = explainer.explain_global(res.winning_model, res.X_test)
        adv_out = adversarial_tester.evaluate_evasion_robustness(res.winning_model, res.X_test, res.y_test)

        anomalies_sample = []
        if res.anomaly_result and not res.anomaly_result.anomalies_df.empty:
            anomalies_sample = res.anomaly_result.anomalies_df.head(5).to_dict(orient="records")

        rules = mitre_gen.generate(
            model_summary={
                "winning_model": res.winning_model_name,
                "problem_type": res.problem_type,
            },
            top_features=xai_out["top_features"],
            anomalies_sample=anomalies_sample,
            robustness_metrics=adv_out,
            api_key=req.api_key,
        )
        return {"status": "success", **rules}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Rule generation failed: {e}")


@app.post("/api/v1/demask", tags=["Data Ingestion & Privacy"])
async def local_demask(req: DemaskRequest):
    """De-mask sensitive entities locally strictly for trusted local UI display."""
    session = get_session(req.session_id)
    local_masker = session["privacy_engine"]
    demasked = local_masker.demask_dict(req.payload)
    return {"status": "success", "demasked_payload": demasked}
